import Foundation

extension String {
    func grapheme(at charIndex: Int) -> Character? {
        if charIndex >= 0, count > charIndex {
            let i = index(startIndex, offsetBy: charIndex)
            return self[i]
        }
        return nil
    }
}

@GGMLActor
final class LlamaContext {
    private let model: OpaquePointer
    private let n_vocab: Int32
    private let candidateBuffer: UnsafeMutableBufferPointer<llama_token_data>
    private let context: OpaquePointer
    private let sampler: UnsafeMutablePointer<llama_sampler>
    private var turns: [Turn]
    private let eosTokenIds: Set<Int32>
    private let quotes: (String, String)?

    let n_ctx: UInt32
    let bosToken: String
    let asset: Model

    func clearAllTokens() {
        turns.removeAll()
        llama_kv_cache_clear(context)
    }

    func save(to url: URL) throws {
        let llmStatePath = url.appendingPathComponent("llmState.bin").path.cString(using: .utf8)
        llama_state_save_file(context, llmStatePath, nil, 0)

        let data = try JSONEncoder().encode(turns)
        try data.write(to: url.appendingPathComponent("turns.json"))
    }

    init(asset: Model) async throws(EmeltalError) {
        self.asset = asset

        llama_backend_init()

        var model_params = llama_model_default_params()
        model_params.use_mlock = false
        model_params.use_mmap = true

        let gpuUsage = await asset.memoryEstimate
        model_params.n_gpu_layers = Int32(gpuUsage.layersOffloaded)

        let modelPath = await asset.localModelPath.path
        guard let model = llama_model_load_from_file(modelPath, model_params) else {
            throw .message("Could not initialise context")
        }

        self.model = model
        n_vocab = llama_n_vocab(model)

        if llama_add_bos_token(model) {
            let bosTokenId = llama_token_bos(model)
            bosToken = String(cString: llama_token_get_text(model, bosTokenId))
        } else {
            bosToken = ""
        }

        eosTokenIds = asset.variant.eosOverrides ?? [llama_token_eos(model)]

        if let quote = asset.variant.quoteTag {
            quotes = ("<\(quote)>", "</\(quote)>")
        } else {
            quotes = nil
        }

        let mem = UnsafeMutablePointer<llama_token_data>.allocate(capacity: Int(n_vocab))
        candidateBuffer = UnsafeMutableBufferPointer(start: mem, count: Int(n_vocab))

        let threadCounts = Int32(performanceCpuCount)

        var ctx_params = llama_context_default_params()
        ctx_params.n_ctx = asset.variant.contextSize
        ctx_params.n_threads = threadCounts
        ctx_params.n_threads_batch = threadCounts
        ctx_params.flash_attn = true
        ctx_params.logits_all = true
        ctx_params.offload_kqv = gpuUsage.offloadKvCache

        guard let newContext = llama_new_context_with_model(model, ctx_params) else {
            throw .message("Could not initialise context")
        }

        let sparams = llama_sampler_chain_default_params()

        guard let newSampler = llama_sampler_chain_init(sparams) else {
            throw .message("Could not initialise sampling")
        }

        let params = await asset.params

        if params.topK != Int(Model.Params.Descriptors.topK.disabled) {
            llama_sampler_chain_add(newSampler, llama_sampler_init_top_k(Int32(params.topK)))
        }

        if params.topP != Model.Params.Descriptors.topP.disabled {
            llama_sampler_chain_add(newSampler, llama_sampler_init_top_p(params.topP, 1))
        }

        if params.temperature > 0 {
            if params.temperatureRange > 0 {
                llama_sampler_chain_add(newSampler, llama_sampler_init_temp_ext(params.temperature, params.temperatureRange, params.temperatureExponent))
            } else {
                llama_sampler_chain_add(newSampler, llama_sampler_init_temp(params.temperature))
            }
        } else {
            llama_sampler_chain_add(newSampler, llama_sampler_init_greedy())
        }

        llama_sampler_init_penalties(Int32(params.repeatCheckPenalty), params.repeatPenatly, params.frequencyPenatly, params.presentPenatly)

        let seed = UInt32.random(in: UInt32.min ..< UInt32.max)
        llama_sampler_chain_add(newSampler, llama_sampler_init_dist(seed))

        sampler = newSampler
        context = newContext
        n_ctx = llama_n_ctx(newContext)
        turns = []
    }

    private func emptyWarmup() {
        guard let eos = eosTokenIds.first else { return }
        log("LLM warmup")
        let bosTokenId = llama_token_bos(model)
        var emptyData = [bosTokenId, eos]
        llama_decode(context, llama_batch_get_one(&emptyData, 2))
        llama_kv_cache_clear(context)
    }

    func restoreStateIfNeeded(from statePath: URL, template: Template) throws {
        let llmState = statePath.appendingPathComponent("llmState.bin")
        let turnStates = statePath.appendingPathComponent("turns.json")

        if FileManager.default.fileExists(atPath: llmState.path),
           FileManager.default.fileExists(atPath: turnStates.path) {
            emptyWarmup()
            log("Warmup complete - Loading state from \(llmState.path)")

            var loaded = 0
            llama_state_load_file(context, llmState.path.cString(using: .utf8), nil, 0, &loaded)
            let infoData = try Data(contentsOf: turnStates)
            turns = try JSONDecoder().decode([Turn].self, from: infoData)
        } else {
            let initial = template.text(for: .initial)
            if initial.isEmpty {
                log("No system prompt")
                emptyWarmup()
            } else {
                log("Adding initial prompt: \(initial)")
                let tokens = tokenize(text: initial)
                let seq = Turn(id: 0)
                _ = seq.append(tokens: tokens, in: context, andPredict: false, offset: 0)
                turns.append(seq)
            }
            log("Warmup complete")
        }
    }

    func shutdown() {
        llama_sampler_free(sampler)
        llama_free(context)
        llama_model_free(model)
        llama_backend_free()
        candidateBuffer.deallocate()
    }

    deinit {
        wordBuffer.deallocate()
        log("Llama context deinit")
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

    private var allTokensCount: Int {
        turns.reduce(0) { $0 + $1.length }
    }

    private func ensureCacheSpace(toFit desiredFreeCount: Int) {
        guard turns.count > 2 else {
            return
        }
        while allTokensCount + desiredFreeCount >= n_ctx {
            var evictedCount: Int32 = 0
            var idsToEvict = Set<llama_seq_id>()
            for turn in turns.suffix(from: 1) { // avoid evicting the system prompt
                evictedCount += Int32(turn.length)
                idsToEvict.insert(turn.id)
                if evictedCount >= desiredFreeCount {
                    break
                }
            }

            if evictedCount < desiredFreeCount {
                log("\nDropping all tokens from token window to fit new tokens")
                clearAllTokens()
                return
            }

            turns.removeAll { idsToEvict.contains($0.id) }

            let evictStart = Int32(turns.first!.length)
            let evictEnd = evictStart + evictedCount

            llama_kv_cache_seq_rm(context, 0, evictStart, evictEnd)
            llama_kv_cache_seq_add(context, 0, evictEnd, -1, -evictedCount)

            log("\nDropping \(evictedCount) tokens from the top of the context to make space for new ones. Tokens remaining after trim: \(allTokensCount)")
        }
    }

    private func tokenize(text: String) -> [llama_token] {
        let textLen = Int32(text.utf8.count)
        let maxTokens = max(128, textLen)
        var newTokens = [llama_token](repeating: 0, count: Int(maxTokens))
        let tokenisedCount = llama_tokenize(model, text, textLen, &newTokens, maxTokens, false, true)
        return Array(newTokens.prefix(Int(tokenisedCount)))
    }

    var turnCount: Int {
        turns.count
    }

    struct FailsafeStopDetector {
        enum Result {
            case possible, detected, notDetected(pending: [Character])
        }

        let text: String
        private var pending = [Character]()

        init(text: String) {
            self.text = text
        }

        mutating func check(character: Character) -> Result {
            if let char = text.grapheme(at: pending.count), char == character {
                pending.append(character)
                if pending.count == text.count {
                    pending.removeAll(keepingCapacity: true)
                    return .detected
                } else {
                    return .possible
                }
            }
            pending.append(character)
            let p = pending
            pending.removeAll(keepingCapacity: true)
            return .notDetected(pending: p)
        }
    }

    private nonisolated(unsafe) let wordBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: 1024)

    private func process(initialText: String, to continuation: AsyncStream<String>.Continuation, template: Template) async {
        let start = Date.now

        let newTokens = tokenize(text: initialText)

        ensureCacheSpace(toFit: newTokens.count)

        let turnId = (turns.last?.id ?? -1) + 1
        let currentTurn = Turn(id: turnId)
        var logits = currentTurn.append(tokens: newTokens, in: context, andPredict: true, offset: allTokensCount)
        turns.append(currentTurn)

        var utf8Builder = UTF8Builder()
        var failsafeStopDetector: FailsafeStopDetector?
        if let failsafeStop = template.failsafeStop {
            failsafeStopDetector = FailsafeStopDetector(text: failsafeStop)
        }

        while allTokensCount <= n_ctx, !Task.isCancelled {
            if logits == nil {
                fatalError()
            }

            for i in 0 ..< candidateBuffer.count {
                candidateBuffer[i] = llama_token_data(id: Int32(i), logit: logits![i], p: 0)
            }

            let newTokenId: llama_token = llama_sampler_sample(sampler, context, -1)

            llama_sampler_accept(sampler, newTokenId)

            if llama_token_is_eog(model, newTokenId) {
                log("Text ended with EOS ID \(newTokenId) - Llama")
                break
            }

            if eosTokenIds.contains(newTokenId) {
                log("Text ended with EOS ID \(newTokenId) - Emeltal")
                break
            }

            let written = Int(llama_token_to_piece(model, newTokenId, wordBuffer, 1024, 0, true))
            if written > 0 {
                var outputString = ""
                for byteIndex in 0 ..< written {
                    let byte = wordBuffer[byteIndex]
                    switch utf8Builder.parseByte(byte) {
                    case .moreRequired:
                        log("Byte: \(newTokenId) / '\(byte)' - building unicode grapheme")

                    case let .result(completeString):
                        outputString.append(completeString)
                        log("Byte: \(newTokenId) / '\(completeString)' / [\(byte)] - completed unicode grapheme")
                    }
                }

                if !outputString.isEmpty {
                    log("Fragment: \(newTokenId) / '\(outputString)' / \(wordBufferBytes(written))")
                    if let quotes {
                        if outputString == quotes.0 {
                            outputString = "<div class='additional-info'>"
                        } else if outputString == quotes.1 {
                            outputString = "</div>"
                        }
                    }

                    if failsafeStopDetector == nil {
                        continuation.yield(outputString)

                    } else { // Scan for failsafe terminator
                        var detected = false
                        let finalChars = outputString.reduce([]) { existing, char -> [Character] in
                            if detected { return existing }
                            switch failsafeStopDetector!.check(character: char) {
                            case .detected:
                                detected = true
                                fallthrough
                            case .possible:
                                return existing
                            case let .notDetected(pending):
                                return existing + pending
                            }
                        }
                        if !finalChars.isEmpty {
                            let out = String(finalChars)
                            continuation.yield(out)
                        }
                        if detected {
                            log("Text ended with failsafe: [\(failsafeStopDetector?.text ?? "")]")
                            break
                        }
                    }
                }
            } else {
                #if DEBUG
                    fatalError("Warning, wordbuffer was zero length - token ID was \(newTokenId)")
                #else
                    log("Warning, wordbuffer was zero length - token ID was \(newTokenId)")
                #endif
            }

            ensureCacheSpace(toFit: 1)

            logits = currentTurn.appendAndPredict(token: newTokenId, in: context, pos: allTokensCount)

            await Task.yield()
        }

        log("Turn was \(currentTurn.length) tokens long, took \(-start.timeIntervalSinceNow) sec")

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
        llama_kv_cache_defrag(context)
        llama_kv_cache_update(context)
    }

    private func wordBufferBytes(_ len: Int) -> String {
        "[" + (0 ..< len).map { wordBuffer[$0] }.map { String($0) }.joined(separator: "][") + "]"
    }
}
