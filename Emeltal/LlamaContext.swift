import Foundation

final actor LlamaContext {
    private let model: OpaquePointer
    private let n_vocab: Int32
    private let candidateBuffer: UnsafeMutableBufferPointer<llama_token_data>
    private let context: OpaquePointer
    private var turns: [Turn]
    private let eosTokenId: Int32

    let n_ctx: UInt32
    let bosToken: String
    let manager: AssetManager

    func reset() {
        turns.removeAll()
        llama_kv_cache_clear(context)
    }

    func save(to url: URL) throws {
        let llmStatePath = url.appendingPathComponent("llmState.bin").path.cString(using: .utf8)
        llama_save_session_file(context, llmStatePath, nil, 0)

        let data = try JSONEncoder().encode(turns)
        try data.write(to: url.appendingPathComponent("turns.json"))
    }

    init(manager: AssetManager) async throws {
        self.manager = manager

        let asset = manager.asset
        let useGpu = asset.category.useGpuOnThisSystem

        llama_backend_init(false)

        var model_params = llama_model_default_params()
        model_params.use_mlock = false
        model_params.use_mmap = true
        if !useGpu {
            model_params.n_gpu_layers = 0
        }

        guard let model = llama_load_model_from_file(asset.localModelPath.path, model_params) else {
            throw "Could not initialise context"
        }

        self.model = model
        n_vocab = llama_n_vocab(model)

        let shouldAddBosToken = {
            let add_bos = llama_add_bos_token(model)
            if add_bos == -1 {
                return llama_vocab_type(model) == LLAMA_VOCAB_TYPE_SPM
            }
            return add_bos != 0
        }()
        if shouldAddBosToken {
            let bosTokenId = llama_token_bos(model)
            bosToken = String(cString: llama_token_get_text(model, bosTokenId))
        } else {
            bosToken = ""
        }

        eosTokenId = llama_token_eos(model)

        let mem = UnsafeMutablePointer<llama_token_data>.allocate(capacity: Int(n_vocab))
        candidateBuffer = UnsafeMutableBufferPointer(start: mem, count: Int(n_vocab))

        let threadCounts = useGpu ? 1 : UInt32(performanceCpuCount)

        var ctx_params = llama_context_default_params()
        ctx_params.n_ctx = 0
        ctx_params.n_batch = asset.category.maxBatch
        ctx_params.n_threads = threadCounts
        ctx_params.n_threads_batch = threadCounts
        ctx_params.seed = UInt32.random(in: UInt32.min ..< UInt32.max)
        ctx_params.logits_all = true
        ctx_params.offload_kqv = true

        guard let newContext = llama_new_context_with_model(model, ctx_params) else {
            throw "Could not initialise context"
        }

        context = newContext
        n_ctx = llama_n_ctx(newContext)
        turns = []
    }

    func restoreStateIfNeeded(from statePath: URL, template: Template) throws {
        let llmState = statePath.appendingPathComponent("llmState.bin")
        let turnStates = statePath.appendingPathComponent("turns.json")

        if FileManager.default.fileExists(atPath: llmState.path),
           FileManager.default.fileExists(atPath: turnStates.path) {
            let bosTokenId = llama_token_bos(model)
            var emptyData = [bosTokenId, eosTokenId]
            llama_decode(context, llama_batch_get_one(&emptyData, 2, 0, 0))
            llama_kv_cache_clear(context)
            log("Warmup complete - Loading state from \(llmState.path)")

            var loaded = 0
            llama_load_session_file(context, llmState.path.cString(using: .utf8), nil, 0, &loaded)

            let infoData = try Data(contentsOf: turnStates)
            turns = try JSONDecoder().decode([Turn].self, from: infoData)
        } else {
            let initial = template.text(for: .initial)
            if initial.isEmpty {
                log("No system prompt, skipping warmup")
            } else {
                log("Adding initial prompt: \(initial)")
                let tokens = tokenize(text: initial)
                let seq = Turn(id: 0)
                _ = seq.append(tokens: tokens, in: context, andPredict: false, offset: 0)
                turns.append(seq)
                log("Warmup complete")
            }
        }
    }

    deinit {
        llama_free(context)
        llama_free_model(model)
        llama_backend_free()
    }

    private var predictionTask: Task<Void, Never>?

    func cancelIfNeeded() async {
        if let predictionTask {
            predictionTask.cancel()
            await predictionTask.value
        }
    }

    func process(text: String, template: Template, turnIndex: Int) -> AsyncStream<String> {
        let promptText = template.text(for: .turn(text: text, index: turnIndex))
        log("Prompt: \(promptText)\n")

        let prediction = AsyncStream.makeStream(of: String.self, bufferingPolicy: .unbounded)
        predictionTask = Task {
            await process(initialText: promptText, to: prediction.continuation, template: template)
            predictionTask = nil
        }
        return prediction.stream
    }

    private final class Turn: Codable {
        let id: llama_seq_id
        var length = 0

        enum CodingKeys: CodingKey {
            case id
            case length
        }

        init(id: llama_seq_id) {
            self.id = id
        }

        func append(tokens: [llama_token], in context: OpaquePointer, andPredict: Bool, offset: Int) -> UnsafeMutablePointer<Float>? {
            let promptCount = tokens.count
            // print("Seq ID \(id): \(promptCount) tokens, offset: \(offset): ", terminator: "")
            var b = llama_batch_init(Int32(promptCount), 0, 1)
            defer {
                length += tokens.count
                llama_batch_free(b)
            }
            b.n_tokens = Int32(promptCount)
            for (i, token) in tokens.enumerated() {
                let pos = Int32(i + offset)
                // print("[\(pos): \(token)] ", terminator: "")
                b.token[i] = token
                b.pos[i] = pos
                b.n_seq_id[i] = 1
                b.seq_id[i]![0] = 0
                b.logits[i] = 0
            }
            // print()
            if andPredict {
                b.logits[promptCount - 1] = 1
                llama_decode(context, b)
                return llama_get_logits_ith(context, Int32(promptCount) - 1)
            } else {
                llama_decode(context, b)
                return nil
            }
        }

        func appendAndPredict(token: llama_token, in context: OpaquePointer, pos: Int) -> UnsafeMutablePointer<Float> {
            // print("+[\(pos): \(token)] ", terminator: "")
            var b = llama_batch_init(1, 0, 1)
            b.n_tokens = 1
            b.token[0] = token
            b.pos[0] = Int32(pos)
            b.n_seq_id[0] = 1
            b.seq_id[0]![0] = 0
            b.logits[0] = 1
            llama_decode(context, b)
            llama_batch_free(b)
            length += 1
            return llama_get_logits_ith(context, 0)!
        }
    }

    private var allTokensCount: Int {
        turns.reduce(0) { $0 + $1.length }
    }

    private func ensureCacheSpace(toFit count: Int) {
        guard turns.count > 2 else {
            return
        }
        while allTokensCount + count >= n_ctx {
            var evictedCount: Int32 = 0
            var idsToEvict = Set<llama_seq_id>()
            for turn in turns.suffix(from: 1) { // avoid evicting the system prompt
                evictedCount += Int32(turn.length)
                idsToEvict.insert(turn.id)
                if evictedCount >= count {
                    break
                }
            }

            if evictedCount < count {
                log("\nDropping all tokens from token window to fit new tokens")
                reset()
                return
            }

            turns.removeAll { idsToEvict.contains($0.id) }

            let evictStart = Int32(turns.first!.length)
            let evictEnd = evictStart + evictedCount

            llama_kv_cache_seq_rm(context, 0, evictStart, evictEnd)
            llama_kv_cache_seq_shift(context, 0, evictEnd, -1, -evictedCount)

            log("\nDropping \(evictedCount) tokens from the top of the context to make space for new ones. Tokens remaining after trim: \(allTokensCount)")
        }
    }

    private func tokenize(text: String) -> [llama_token] {
        let textLen = Int32(text.utf8.count)
        let maxTokens = max(128, textLen)
        var newTokens = [llama_token](repeating: 0, count: Int(maxTokens))
        let tokenisedCount = llama_tokenize(model, text, textLen, &newTokens, maxTokens, false, true)
        let newTokenLimit = Int(min(manager.asset.category.maxBatch, UInt32(tokenisedCount)))
        return Array(newTokens.prefix(newTokenLimit))
    }

    var turnCount: Int {
        turns.count
    }

    private func process(initialText: String, to continuation: AsyncStream<String>.Continuation, template: Template) async {
        let newTokens = tokenize(text: initialText)

        ensureCacheSpace(toFit: newTokens.count)

        let turnId = (turns.last?.id ?? -1) + 1
        let currentTurn = Turn(id: turnId)
        var logits = currentTurn.append(tokens: newTokens, in: context, andPredict: true, offset: allTokensCount)
        turns.append(currentTurn)

        while allTokensCount <= n_ctx, !Task.isCancelled {
            if logits == nil {
                fatalError()
            }

            for i in 0 ..< candidateBuffer.count {
                candidateBuffer[i] = llama_token_data(id: Int32(i), logit: logits![i], p: 0)
            }

            var candidates_p = llama_token_data_array(data: candidateBuffer.baseAddress, size: candidateBuffer.count, sorted: false)
            let params = manager.asset.params
            llama_sample_top_k(context, &candidates_p, Int32(params.topK), 1)
            llama_sample_top_p(context, &candidates_p, params.topP, 1)
            llama_sample_temp(context, &candidates_p, params.temperature)
            llama_sample_repetition_penalties(context, &candidates_p, newTokens,
                                              newTokens.count, // previous token count
                                              params.repeatPenatly, // repeat penalty
                                              params.frequencyPenatly, // freq penalty
                                              params.presentPenatly) // present penalty
            let new_token_id = llama_sample_token(context, &candidates_p)
            if new_token_id == eosTokenId {
                break
            }

            ensureCacheSpace(toFit: 1)

            logits = currentTurn.appendAndPredict(token: new_token_id, in: context, pos: allTokensCount)

            if let new_token_str = Self.text(from: new_token_id, in: model) {
                // log("Fragment: \(new_token_id) - '\(new_token_str)'")
                continuation.yield(new_token_str)
            } else {
                log("Warning, wordbuffer was invalid - token ID was \(new_token_id)")
            }

            await Task.yield()
        }

        log("Turn was \(currentTurn.length) tokens long")

        if Task.isCancelled {
            let cancelText = template.text(for: .cancel)
            if !cancelText.isEmpty {
                log("Prediction was cancelled, ensuring prediction text is capped gracefully")
                let newTokens = tokenize(text: cancelText)
                ensureCacheSpace(toFit: newTokens.count)
                _ = currentTurn.append(tokens: newTokens, in: context, andPredict: false, offset: allTokensCount)
            }
        }

        continuation.finish()
    }

    private static let wordBuffer = UnsafeMutablePointer<Int8>.allocate(capacity: 256)

    private static func text(from token: Int32, in model: OpaquePointer) -> String? {
        let written = Int(llama_token_to_piece(model, token, wordBuffer, 255))
        if written > 0 {
            wordBuffer[written] = 0
            return String(utf8String: wordBuffer)
        } else {
            return nil
        }
    }
}
