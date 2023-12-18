import Combine
import Foundation
import SwiftUI

@MainActor
@Observable
final class AppState: Identifiable {
    nonisolated var id: String { asset.id }

    var multiLineText = ""
    var messageLog = ""
    var mode: Mode = .startup {
        didSet {
            if oldValue != mode, let speaker {
                mode.audioFeedback(using: speaker)
            }
        }
    }

    var textOnly = Persisted._textOnly {
        didSet {
            Task {
                await speaker?.cancelIfNeeded()
            }
            Persisted._textOnly = textOnly
        }
    }

    enum ListenState {
        case notListening, pushButton, voiceActivated
    }

    var displayName: String {
        asset.displayName
    }

    var listenState = ListenState.notListening

    var statusMessage: String? = "Starting…"

    var floatingMode: Bool = Persisted._floatingMode {
        didSet {
            if oldValue == floatingMode {
                return
            }
            Persisted._floatingMode = floatingMode
            processFloatingMode(fromBoot: false)
        }
    }

    enum Mode: Equatable {
        static func == (lhs: Self, rhs: Self) -> Bool {
            switch lhs {
            case .startup:
                if case .startup = rhs {
                    return true
                }
            case .booting:
                if case .booting = rhs {
                    return true
                }
            case .warmup:
                if case .warmup = rhs {
                    return true
                }
            case .listening:
                if case .listening = rhs {
                    return true
                }
            case .loading:
                if case .loading = rhs {
                    return true
                }
            case .noting:
                if case .noting = rhs {
                    return true
                }
            case .replying:
                if case .replying = rhs {
                    return true
                }
            case .thinking:
                if case .thinking = rhs {
                    return true
                }
            case .waiting:
                if case .waiting = rhs {
                    return true
                }
            }
            return false
        }

        case startup, booting, warmup, loading(managers: [AssetManager]), waiting, listening(state: Mic.State), noting, thinking, replying

        func audioFeedback(using speaker: Speaker) {
            switch self {
            case .listening:
                Task {
                    await speaker.playEffect(speaker.startEffect)
                }
            case .noting:
                Task {
                    await speaker.playEffect(speaker.endEffect)
                }
            case .booting, .loading, .replying, .startup, .thinking, .waiting, .warmup:
                break
            }
        }

        var showGenie: Bool {
            switch self {
            case .noting, .replying, .thinking:
                true
            case .booting, .listening, .loading, .startup, .waiting, .warmup:
                false
            }
        }

        var showAlwaysOn: Bool {
            switch self {
            case .booting, .loading, .noting, .replying, .startup, .thinking, .warmup:
                false
            case .listening, .waiting:
                true
            }
        }
    }

    func reset() async throws {
        messageLog = ""
        await llamaContext?.reset()
        try await save()
    }

    let asset: Asset

    init(asset: Asset) {
        self.asset = asset
    }

    private var llamaContext: LlamaContext?
    private var whisperContext: WhisperContext?
    private var template: Template!
    private var speaker: Speaker?
    private let mic = Mic()
    private var micObservation: Cancellable?

    private func processFloatingMode(fromBoot: Bool) {
        if floatingMode {
            textOnly = false
            for window in NSApplication.shared.windows {
                let originalFrame = window.frame
                window.level = .floating
                window.styleMask = .borderless
                window.isOpaque = false
                window.backgroundColor = .clear
                if fromBoot {
                    window.setFrame(CGRect(x: originalFrame.minX, y: originalFrame.minY, width: assistantWidth, height: assistantHeight), display: true)
                } else {
                    window.setFrame(CGRect(x: originalFrame.maxX - assistantWidth - 32, y: originalFrame.minY + 191, width: assistantWidth, height: assistantHeight), display: true)
                }
            }
        } else {
            for window in NSApplication.shared.windows {
                window.level = .normal
                window.styleMask = [.titled, .closable, .miniaturizable]
                window.isOpaque = true
                window.backgroundColor = .windowBackgroundColor
                if !fromBoot {
                    let w: CGFloat = 800
                    let h: CGFloat = 600
                    if let bounds = NSApplication.shared.keyWindow?.screen?.frame.size {
                        let x = (bounds.width - w) * 0.5
                        let y = (bounds.height - h) * 0.5
                        window.setFrame(CGRect(x: x, y: y, width: w, height: h), display: true)
                    } else {
                        window.setFrame(CGRect(x: 100, y: 100, width: w, height: h), display: true)
                    }
                }
            }
        }
    }

    func boot() async throws {
        guard mode == .startup else {
            return
        }

        let llm = AssetManager(fetching: asset)
        let whisper = AssetManager(fetching: .whisper)

        mode = .loading(managers: [llm, whisper])

        for manager in [llm, whisper] {
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                if manager.phase.isOngoing {
                    manager.builderDone = { _ in
                        continuation.resume()
                    }
                } else {
                    continuation.resume()
                }
            }
            if case let .error(error) = manager.phase {
                throw error
            }
        }

        processFloatingMode(fromBoot: true)

        if hasSavedState {
            messageLog = (try? String(contentsOf: textPath)) ?? ""
            messageLog += "\n"
        } else if let speaker, await !speaker.havePreferredVoice {
            messageLog = "**The ideal voice for this app is the premium variant of \"Zoe\", which does not seem to be installed on your system. You can install it from your system settings and restart this app for the best experience.**\n\n"
        }

        let l = Task.detached { try await LlamaContext(manager: llm) }
        let w = Task.detached { let W = try await WhisperContext(manager: whisper); _ = await W.warmup(); return W }
        let s = Task.detached { try await Speaker() }

        micObservation = mic.statePublisher.receive(on: DispatchQueue.main).sink { [weak self] newState in
            guard let self else { return }
            if case let .listening(myState) = mode, newState != myState {
                if newState == .quiet(prefixBuffer: []), listenState == .voiceActivated {
                    Task {
                        await self.endMic(processOutput: true)
                    }
                }
                mode = .listening(state: newState)
            }
        }

        statusMessage = "Loading TTS…"
        speaker = try await s.value

        statusMessage = "Loading ASR…"
        whisperContext = try await w.value

        statusMessage = "Loading LLM…"
        llamaContext = try await l.value

        template = llm.asset.mlTemplate(in: llamaContext!)

        mode = .warmup
        statusMessage = "Warming up AI…"

        try await llamaContext?.restoreStateIfNeeded(from: statePath, template: template)

        shouldWaitOrListen()
        statusMessage = nil
    }

    /*
     private func buildTemplate() {
         /* OpenChat 3.5 */
         /*
          template = LlamaContext.Template(initial: "",
          turn: "<s>GPT4 Correct User: {text}<|end_of_turn|>GPT4 Correct Assistant:")
          */

         /*
          template = LlamaContext.Template(initial: "<s>GPT4 Correct User: Hello<|end_of_turn|>GPT4 Correct Assistant: Hi<|end_of_turn|></s>",
          turn: "<s>GPT4 Correct User: {text}<|end_of_turn|>GPT4 Correct Assistant: ")
          */

         /* Zephyr */
         /*
          template = LlamaContext.Template(initial: "<|system|>\nYou are a polite and intelligent assistant.</s>",
          turn: "\n<|user|>\n{text}</s>\n<|assistant|>\n",
          first: first)
          */

         /* Llama 2 / SunsetBoulevard / Mixtral */
         /*
          template = LlamaContext.Template(initial: "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
          turn: "\n[INST] {text} [/INST]\n",
          first: first)
          */

         /* Synthia-MixtralInst */
         /*
          template = LlamaContext.Template(initial: "",
          turn: "[INST] {text} [/INST] ",
          first: first)
          */
         /*
         template = LlamaContext.Template(format: .userAssistant,
                                          system: "You are an intelligent and cheerful chatbot.",
                                          bosToken: llamaContext!.bosToken)
          */
         /* Samantha 13B */
         /*
          template = LlamaContext.Template(initial: "\(bos)You are a caring and empathetic sentient AI companion named Samantha.\(nl)",
          turn: "\(nl)USER: {text}\(nl)ASSISTANT: ",
          first: first)
          */

         /* Mistral 7B */
         /*
          template = LlamaContext.Template(initial: "",
          turn: "\(bos)[INST]{text}[/INST]",
          first: first)
          */

         /* go-bruins */
         /*
          template = LlamaContext.Template(initial: "\(bos)You are a friendly, funny, and concise chatbot.\n\n",
          turn: "### Instruction:\n{text}\n\n### Response:\n",
          first: first)
          */
     }
      */

    private func shouldWaitOrListen() {
        Task {
            if listenState == .voiceActivated {
                await startMic()
            } else {
                withAnimation {
                    mode = .waiting
                }
            }
        }
    }

    func switchToPushButton() {
        if listenState == .voiceActivated {
            listenState = .notListening
            Task {
                await endMic(processOutput: false)
            }
        }
    }

    func switchToVoiceActivated() {
        if listenState == .pushButton {
            return
        }
        listenState = .voiceActivated
        Task {
            await startMic()
        }
    }

    func send() {
        let text = multiLineText.trimmingCharacters(in: .whitespacesAndNewlines)
        switch mode {
        case .booting, .listening, .loading, .replying, .startup, .thinking, .warmup:
            return
        case .noting, .waiting:
            break
        }
        if text.isEmpty {
            shouldWaitOrListen()
            return
        }

        withAnimation {
            mode = .thinking
            multiLineText = ""
        }
        Task.detached(priority: .userInitiated) {
            await self.complete(text: text)
        }
    }

    private func startMic() async {
        await speaker?.cancelIfNeeded()

        guard whisperContext != nil else { return }
        try? await mic.start()
        let micState = await mic.state
        withAnimation {
            mode = .listening(state: micState)
        }
    }

    private func endMic(processOutput: Bool) async {
        guard let whisperContext, let samples = try? await mic.stop() else {
            return
        }

        guard processOutput else {
            shouldWaitOrListen()
            return
        }

        if samples.count > 8000 {
            withAnimation {
                mode = .noting
            }
            let result = await Task.detached(priority: .userInitiated) { await whisperContext.transcribe(samples: samples).trimmingCharacters(in: .whitespacesAndNewlines) }.value
            if result.count > 1 {
                multiLineText = result
                send()

            } else {
                log("Transcription too short, will ignore it")
                shouldWaitOrListen()
            }
        } else {
            log("Sound clip too short, will ignore it")
            shouldWaitOrListen()
        }
    }

    func pushButtonUp() {
        guard case .pushButton = listenState else {
            return
        }
        listenState = .notListening
        Task {
            await endMic(processOutput: true)
        }
    }

    func pushButtonDown() {
        switch listenState {
        case .notListening:
            listenState = .pushButton
            Task {
                await startMic()
            }
        case .pushButton:
            return
        case .voiceActivated:
            Task {
                await speaker?.cancelIfNeeded()
            }
        }
    }

    func complete(text: String) async {
        guard let llamaContext else {
            return
        }

        let sanitisedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        messageLog += "\n#### \(sanitisedText)\n"

        let stream = llamaContext.process(text: sanitisedText, template: template)

        var sentenceBuffer = ""
        var inQuote = false
        var started = false
        var backtickCount = 0
        for await fragment in stream {
            // log(fragment)
            if !started {
                withAnimation {
                    mode = .replying
                }
                started = true
            }
            for c in fragment.enumerated() {
                if c.element == "`" {
                    backtickCount += 1
                    if backtickCount == 3 {
                        backtickCount = 0
                        inQuote.toggle()
                    }
                } else {
                    backtickCount = 0
                }
            }

            messageLog += fragment // .replacingOccurrences(of: "\n", with: "<br>")

            if !(textOnly || inQuote) {
                sentenceBuffer += fragment
                if let range = sentenceBuffer.ranges(of: #/[\.|\!|\?|\n|\r|\,|\;]\ /#).first {
                    var sentence = String(sentenceBuffer[sentenceBuffer.startIndex ..< range.upperBound])
                    sentence = sentence.replacingOccurrences(of: "...", with: "…").replacingOccurrences(of: "'s", with: "ߴs")
                    await speaker?.add(text: sentence)
                    sentenceBuffer = String(sentenceBuffer[range.upperBound ..< sentenceBuffer.endIndex])
                }
            }
        }

        if !sentenceBuffer.isEmpty {
            if !textOnly {
                sentenceBuffer = sentenceBuffer.replacingOccurrences(of: "...", with: "…").replacingOccurrences(of: "'s", with: "ߴs")
                await speaker?.add(text: sentenceBuffer)
            }
        }
        messageLog += "\n"

        try? await save()
        await speaker?.waitForCompletion()
        shouldWaitOrListen()
    }

    private var statePath: URL {
        asset.localStatePath
    }

    private var textPath: URL {
        statePath.appendingPathComponent("text.txt")
    }

    var hasSavedState: Bool {
        let fm = FileManager.default
        return fm.fileExists(atPath: textPath.path)
    }

    func save() async throws {
        if messageLog.isEmpty {
            let fm = FileManager.default
            if fm.fileExists(atPath: textPath.path) {
                try fm.removeItem(at: textPath)
                try fm.removeItem(at: statePath)
                log("Cleared state at \(statePath.path)")
            }

        } else {
            try messageLog.write(toFile: textPath.path, atomically: true, encoding: .utf8)
            try await llamaContext?.save(to: statePath)
            log("Saved state to \(statePath.path)")
        }
    }
}
