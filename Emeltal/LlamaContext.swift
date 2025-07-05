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
    private let vocab: OpaquePointer
    private let n_vocab: Int32
    private let candidateBuffer: UnsafeMutableBufferPointer<llama_token_data>
    private let context: OpaquePointer
    private let sampler: UnsafeMutablePointer<llama_sampler>
    private var turns: [Turn]
    private let eosTokenIds: Set<Int32>

    let n_ctx: UInt32
    let bosToken: String
    let eosManual: String?
    let asset: Model

    func clearAllTokens() {
        turns.removeAll()

        if let kv = llama_get_memory(context) {
            llama_memory_clear(kv, true)
        }
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
        vocab = llama_model_get_vocab(model)
        n_vocab = llama_vocab_n_tokens(vocab)

        if llama_vocab_get_add_bos(vocab) {
            let bosTokenId = llama_vocab_bos(vocab)
            bosToken = String(cString: llama_vocab_get_text(vocab, bosTokenId))
        } else {
            bosToken = ""
        }

        eosTokenIds = asset.variant.eosOverrides ?? [llama_vocab_eos(vocab)]
        eosManual = asset.variant.eosManual

        let mem = UnsafeMutablePointer<llama_token_data>.allocate(capacity: Int(n_vocab))
        candidateBuffer = UnsafeMutableBufferPointer(start: mem, count: Int(n_vocab))

        let threadCounts = Int32(performanceCpuCount)

        var ctx_params = llama_context_default_params()
        ctx_params.n_ctx = asset.variant.contextSize
        ctx_params.n_threads = threadCounts
        ctx_params.n_threads_batch = threadCounts
        ctx_params.flash_attn = true
        ctx_params.offload_kqv = gpuUsage.offloadKvCache

        guard let newContext = llama_init_from_model(model, ctx_params) else {
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
        let bosTokenId = llama_vocab_bos(vocab)
        var emptyData = [bosTokenId, eos]
        llama_decode(context, llama_batch_get_one(&emptyData, 2))
        if let kv = llama_get_memory(context) {
            llama_memory_clear(kv, true)
        }
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
            if initial.isEmpty || template.systemText == nil {
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

    func shutdown() async {
        await cancelIfNeeded()
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

    func process(text: String, template: Template, turnIndex: Int) -> AsyncStream<Character> {
        let promptText = template.text(for: .turn(text: text, index: turnIndex))
        log("Prompt: \(promptText)\n")

        let prediction = AsyncStream.makeStream(of: Character.self, bufferingPolicy: .unbounded)
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

            if let firstTurn = turns.first {
                let evictStart = Int32(firstTurn.length)
                evictKvTokens(at: evictStart, length: evictedCount)
            }

            log("\nDropping \(evictedCount) tokens from the top of the context to make space for new ones. Tokens remaining after trim: \(allTokensCount)")
        }
    }

    private func evictKvTokens(at evictStart: Int32, length: Int32) {
        if let kv = llama_get_memory(context) {
            let evictEnd = evictStart + length
            llama_memory_seq_rm(kv, 0, evictStart, evictEnd)
            llama_memory_seq_add(kv, 0, evictEnd, -1, -length)
        }
    }

    private func tokenize(text: String) -> [llama_token] {
        let textLen = Int32(text.utf8.count)
        let maxTokens = max(128, textLen)
        var newTokens = [llama_token](repeating: 0, count: Int(maxTokens))
        let tokenisedCount = llama_tokenize(vocab, text, textLen, &newTokens, maxTokens, false, true)
        return Array(newTokens.prefix(Int(tokenisedCount)))
    }

    var turnCount: Int {
        turns.count
    }

    private struct FailsafeStopDetector {
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

    private func process(initialText: String, to continuation: AsyncStream<Character>.Continuation, template: Template) async {
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

        var inThinkSection = false
        var tagBuffer: String?
        let muteTags = asset.variant.muteTokens?.flatMap { [$0, "/\($0)"] }

        func output(_ out: String) {
            for char in out {
                if let buffer = tagBuffer {
                    if char == ">" {
                        if let muteTags, muteTags.contains(buffer) {
                            log("Trimmed tag: \(buffer)")
                        } else {
                            continuation.yield("<")
                            log("Detected tag: \(buffer)")
                            switch buffer {
                            case "think":
                                inThinkSection = true
                            case "/think":
                                inThinkSection = false
                            default:
                                break
                            }
                            for c in buffer {
                                continuation.yield(c)
                            }
                            continuation.yield(">")
                        }
                        tagBuffer = nil
                    } else {
                        tagBuffer?.append(char)

                        // Tag is too long
                        if buffer.count > 10 {
                            continuation.yield("<")
                            for c in buffer {
                                continuation.yield(c)
                            }
                            continuation.yield(char)
                            tagBuffer = nil
                        }
                    }
                } else if char == "<" {
                    tagBuffer = ""
                } else {
                    continuation.yield(char)
                }
            }
        }

        sentence: while allTokensCount <= n_ctx, !Task.isCancelled {
            if logits == nil {
                fatalError()
            }

            for i in 0 ..< candidateBuffer.count {
                candidateBuffer[i] = llama_token_data(id: Int32(i), logit: logits![i], p: 0)
            }

            let newTokenId: llama_token = llama_sampler_sample(sampler, context, -1)

            if llama_vocab_is_eog(vocab, newTokenId) || eosTokenIds.contains(newTokenId) {
                log("Text ended with EOS ID \(newTokenId)")
                ensureCacheSpace(toFit: 1)
                _ = currentTurn.append(tokens: [newTokenId], in: context, andPredict: false, offset: allTokensCount)
                break
            }

            let written = Int(llama_token_to_piece(vocab, newTokenId, wordBuffer, 1024, 0, true))
            if written > 0 {
                var outputString = ""
                for byteIndex in 0 ..< written {
                    let byte = wordBuffer[byteIndex]
                    switch utf8Builder.parseByte(byte) {
                    case .moreRequired:
                        // log("Byte: \(newTokenId) / '\(byte)' - building unicode grapheme")
                        break

                    case let .result(completeString):
                        outputString.append(completeString)
                        // log("Byte: \(newTokenId) / '\(completeString)' / [\(byte)] - completed unicode grapheme")
                    }
                }

                if !outputString.isEmpty {
                    if outputString == eosManual {
                        log("Text ended with EOS String: \(outputString)")
                        break sentence
                    }
                }

                if !outputString.isEmpty {
                    log("Fragment: \(newTokenId) / '\(outputString)' / \(wordBufferBytes(written))")

                    if failsafeStopDetector == nil {
                        output(outputString)

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
                            output(out)
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

            logits = currentTurn.appendAndPredict(token: newTokenId, in: context, pos: allTokensCount, inThink: inThinkSection)

            await Task.yield()
        }

        if let buffer = tagBuffer {
            for c in buffer {
                continuation.yield(c)
            }
            tagBuffer = nil
        }

        if let hadThinkBlock = currentTurn.thinkBlockRange {
            log("Think block in turn \(currentTurn.id), logit range: \(hadThinkBlock), will trim related tokens from context")
            let currentTurnStart = turns.dropLast().reduce(0) { $0 + $1.length }
            let kvStart = Int32(currentTurnStart + hadThinkBlock.lowerBound)
            let kvLength = Int32(hadThinkBlock.startIndex.distance(to: hadThinkBlock.endIndex))
            evictKvTokens(at: kvStart, length: kvLength)
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
    }

    private func wordBufferBytes(_ len: Int) -> String {
        "[" + (0 ..< len).map { wordBuffer[$0] }.map { String($0) }.joined(separator: "][") + "]"
    }
}
