import Combine
import Foundation
import Network
import SwiftUI

@MainActor
@Observable
final class AppState: Identifiable, ModeProvider {
    nonisolated var id: String { asset.id }

    var multiLineText = ""
    var messageLog = ""
    var mode: AppMode = .startup {
        didSet {
            if oldValue != mode {
                if let speaker {
                    mode.audioFeedback(using: speaker)
                }
                Task {
                    await remote.send(.appMode, content: mode.data)
                }
            }
        }
    }

    var activationState = ActivationState.button {
        didSet {
            if oldValue != activationState {
                Task {
                    await remote.send(.appActivationState, content: activationState.data)
                }
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

    var displayName: String {
        asset.category.displayName
    }

    var statusMessage: String? = "Starting…"

    var isRemoteConnected = false {
        didSet {
            Task {
                await mic.setRemoteMode(isRemoteConnected)
                await speaker?.setMute(isRemoteConnected)
            }
        }
    }

    var floatingMode: Bool = Persisted._floatingMode {
        didSet {
            if oldValue == floatingMode {
                return
            }
            Persisted._floatingMode = floatingMode
            processFloatingMode(fromBoot: false)
        }
    }

    func reset() async throws {
        messageLog = ""
        await llamaContext?.reset()
        try await save()
        try await chatInit()
    }

    let asset: Asset

    init(asset: Asset) {
        self.asset = asset

        connectionStateObservation = remote.statePublisher.receive(on: DispatchQueue.main).sink { [weak self] state in
            guard let self else { return }
            isRemoteConnected = state.isConnectionActive
        }
    }

    private var llamaContext: LlamaContext?
    private var whisperContext: WhisperContext?
    private var template: Template!
    private var speaker: Speaker?
    private let mic = Mic()
    private var micObservation: Cancellable?

    private let remote = EmeltalConnector()
    private var connectionStateObservation: Cancellable!

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
        let whisper = AssetManager(fetching: Asset(defaultFor: .whisper))

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
        let s = Task.detached { try Speaker() }

        micObservation = mic.statePublisher.receive(on: DispatchQueue.main).sink { [weak self] newState in
            guard let self else { return }
            if case let .listening(micState) = mode, newState != micState {
                if newState == .quiet(prefixBuffer: []), activationState == .voiceActivated {
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

        try await chatInit()

        Task {
            await startServer()
        }
    }

    private func startServer() async {
        try? await Task.sleep(for: .seconds(1))
        log("Listening for remote connections")
        let stream = await remote.startServer()
        for await nibble in stream {
            log("From client: \(nibble)")

            switch nibble.payload {
            case .appActivationState, .appMode, .generatedSentence, .heartbeat, .unknown:
                break

            case .recordedSpeech:
                if let speech = nibble.data {
                    await mic.addToBuffer(speech)
                }

            case .recordedSpeechDone:
                await endMic(processOutput: true)

            case .buttonDown:
                pushButtonDown()

            case .toggleListeningMode:
                if activationState == .voiceActivated {
                    switchToPushButton()
                } else {
                    switchToVoiceActivated()
                }
            }
        }
    }

    private func chatInit() async throws {
        try await llamaContext?.restoreStateIfNeeded(from: statePath, template: template)
        shouldWaitOrListen()
        statusMessage = nil
    }

    private func shouldWaitOrListen() {
        Task {
            if activationState == .voiceActivated {
                await startMic()
            } else {
                withAnimation {
                    mode = .waiting
                }
            }
        }
    }

    func switchToPushButton() {
        if activationState == .button {
            return
        }
        activationState = .button
        Task {
            await endMic(processOutput: false)
        }
    }

    func switchToVoiceActivated() {
        if activationState == .voiceActivated {
            return
        }
        activationState = .voiceActivated
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
        if case .voiceActivated = activationState {
            return
        }
        Task {
            await endMic(processOutput: true)
        }
    }

    func pushButtonDown() {
        Task {
            await speaker?.cancelIfNeeded()
            if case .voiceActivated = activationState {
                return
            }
            await startMic()
        }
    }

    func complete(text: String) async {
        guard let llamaContext else {
            return
        }

        let sanitisedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        messageLog += "\n#### \(sanitisedText)\n"

        let index = await max(0, llamaContext.turnCount - 1)
        let stream = llamaContext.process(text: sanitisedText, template: template, turnIndex: index)

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
            sentenceBuffer += fragment
            if let range = sentenceBuffer.ranges(of: #/[\.|\!|\?|\n|\r|\,|\;\:]\ /#).first {
                let sentence = String(sentenceBuffer[sentenceBuffer.startIndex ..< range.upperBound])
                await handleText(sentence, inQuote: inQuote)
                sentenceBuffer = String(sentenceBuffer[range.upperBound ..< sentenceBuffer.endIndex])
            }
        }

        if !sentenceBuffer.isEmpty {
            await handleText(sentenceBuffer, inQuote: inQuote)
        }
        messageLog += "\n"

        try? await save()
        await speaker?.waitForCompletion()
        shouldWaitOrListen()
    }

    private func handleText(_ text: String, inQuote: Bool) async {
        let spoken = text.replacingOccurrences(of: "...", with: "…")
        if !(textOnly || inQuote) {
            await speaker?.add(text: spoken)
        }

        if !inQuote {
            if let data = spoken.data(using: .utf8) {
                await remote.send(.generatedSentence, content: data)
            }
        }
    }

    private var statePath: URL {
        asset.localStatePath
    }

    private var textPath: URL {
        statePath.appendingPathComponent("text.txt")
    }

    var hasSavedState: Bool {
        FileManager.default.fileExists(atPath: textPath.path)
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
