import AVFoundation
import Foundation
import Network
import SwiftUI

@MainActor
@Observable
final class AppState: Identifiable, ModeProvider {
    nonisolated var id: String { asset.id }

    var multiLineText = ""

    var messageLog = MessageLog(path: nil)

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

    var statusMessage: String? = "Starting"

    var isRemoteConnected = false {
        didSet {
            Task { [isRemoteConnected] in
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

    var shouldPromptForIdealVoice = false

    func reset() async throws {
        mode = .warmup
        await llamaContext?.cancelIfNeeded()
        await speaker?.cancelIfNeeded()
        await llamaContext?.clearAllTokens()
        messageLog.reset()
        try await save()
        try await chatInit(hasSavedState: false)
    }

    let asset: Asset

    init(asset: Asset) {
        self.asset = asset

        connectionStateObservation = Task {
            for await state in remote.stateStream.stream {
                isRemoteConnected = state.isConnectionActive
            }
        }

        Task {
            do {
                try await mainLoop()
            } catch {
                log("Error in main loop: \(error.localizedDescription)")
                fatalError(error.localizedDescription)
            }
        }
    }

    func shutdown() async {
        log("Shutting down app state…")
        micObservation?.cancel()
        connectionStateObservation.cancel()
        await remote.shutdown()
        await speaker?.shutdown()
        await llamaContext?.shutdown()
        await whisperContext?.shutdown()
    }

    deinit {
        log("App state deinit")
    }

    private var llamaContext: LlamaContext?
    private var whisperContext: WhisperContext?
    private var template: Template!
    private var speaker: Speaker?
    private let mic = Mic()
    private var micObservation: Task<Void, Never>?

    private let remote = ServerConnector()
    private var connectionStateObservation: Task<Void, Never>!

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

    private func sendMessageLog(value: String, initial: Bool) {
        guard let data = value.data(using: .utf8) else { return }
        Task {
            if initial {
                if data.isEmpty {
                    await remote.send(.textInitial, content: emptyData)
                } else {
                    await remote.send(.textInitial, content: data)
                }
            } else {
                await remote.send(.textDiff, content: data)
            }
        }
    }

    private func appendToMessageLog(_ text: String) {
        messageLog.appendText(text)
        sendMessageLog(value: text, initial: false)
    }

    func mainLoop() async throws {
        let llm = await AssetManager(fetching: asset)
        let whisper = await AssetManager(fetching: Asset(defaultFor: .whisper))

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

        let hasSavedState = FileManager.default.fileExists(atPath: textPath.path)
        if hasSavedState {
            messageLog = MessageLog(path: textPath)
        }

        let ctxs = Task.detached {
            let l = try await LlamaContext(manager: llm)
            let w = try await WhisperContext(manager: whisper)
            _ = try await w.warmup()
            return (l, w)
        }

        let audioTask = Task.detached { [weak self] () -> Speaker? in
            guard let self else { return nil }

            await AVCaptureDevice.requestAccess(for: .audio)

            let spk = try Speaker()
            try await spk.warmup()

            await mic.warmup()
            await setupMicObservation()

            return spk
        }

        statusMessage = "Audio Setup"
        speaker = try await audioTask.value
        shouldPromptForIdealVoice = await !(speaker!.havePreferredVoice)

        statusMessage = "ML Setup"
        (llamaContext, whisperContext) = try await ctxs.value

        template = llm.asset.mlTemplate(in: llamaContext!)

        mode = .warmup
        statusMessage = "Warming Up"

        try await chatInit(hasSavedState: hasSavedState)

        log("Listening for remote connections")
        let stream = await remote.startServer()

        for await nibble in stream {
            log("From client: \(nibble)")

            switch nibble.payload {
            case .appActivationState, .appMode, .heartbeat, .responseDone, .spokenSentence, .textDiff, .textInitial, .unknown:
                break

            case .hello:
                await remote.send(.appActivationState, content: activationState.data)
                await remote.send(.appMode, content: mode.data)

                let allText = messageLog.history + messageLog.newText
                sendMessageLog(value: allText, initial: true)

            case .requestReset:
                try? await reset()

            case .textInput:
                if let textData = nibble.data, let text = String(data: textData, encoding: .utf8) {
                    multiLineText = text
                    send()
                }

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

        log("Main loop done")
    }

    private func setupMicObservation() {
        micObservation = Task {
            for await newState in mic.stateStream.stream {
                if case let .listening(micState) = mode, newState != micState {
                    if newState == .quiet(prefixBuffer: []), activationState == .voiceActivated {
                        Task {
                            await endMic(processOutput: true)
                        }
                    }
                    mode = .listening(state: newState)
                }
            }
        }
    }

    private func chatInit(hasSavedState: Bool) async throws {
        if !hasSavedState, let systemText = template.systemText {
            messageLog = MessageLog(string: "> *\"\(systemText)\"*\n")
        }

        try await llamaContext?.restoreStateIfNeeded(from: statePath, template: template)
        shouldWaitOrListen()
        statusMessage = nil

        if let (used, maximum, system) = asset.category.memoryStrings {
            log("Startup complete, GPU usage: \(used) / \(maximum), system total: \(system)")
        }
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
        try? await mic.start(detectVoice: activationState == .voiceActivated)
        let micState = await mic.state
        withAnimation {
            mode = .listening(state: micState)
        }
    }

    private func endMic(processOutput: Bool) async {
        guard let whisperContext, let samples = try? await mic.stop(temporary: false) else {
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
            let result = try? await Task.detached(priority: .userInitiated) { try await whisperContext.transcribe(samples: samples).trimmingCharacters(in: .whitespacesAndNewlines) }.value
            if let result, result.count > 1 {
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
            if case .waiting = mode {
                await startMic()
            } else {
                await llamaContext?.cancelIfNeeded()
            }
        }
    }

    func complete(text: String) async {
        guard let llamaContext else {
            return
        }

        let sanitisedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        appendToMessageLog("\n#### \(sanitisedText)\n")

        let index = await max(0, llamaContext.turnCount - 1)
        let stream = await llamaContext.process(text: sanitisedText, template: template, turnIndex: index)

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

            appendToMessageLog(fragment)
            sentenceBuffer += fragment
            if let range = sentenceBuffer.ranges(of: #/[\.|\!|\?|\n|\r|\,|\;\:]\s/#).first, mode.nominal {
                let sentence = String(sentenceBuffer[sentenceBuffer.startIndex ..< range.upperBound])
                await handleText(sentence, inQuote: inQuote)
                sentenceBuffer = String(sentenceBuffer[range.upperBound ..< sentenceBuffer.endIndex])
            }
        }

        if mode.nominal {
            if !sentenceBuffer.isEmpty {
                await handleText(sentenceBuffer, inQuote: inQuote)
            }
            appendToMessageLog("\n")

            try? await save()
            await speaker?.waitForCompletion()
            shouldWaitOrListen()
        }
    }

    private func handleText(_ text: String, inQuote: Bool) async {
        let finalText = text.replacingOccurrences(of: "...", with: "…").trimmingCharacters(in: .whitespacesAndNewlines)
        log("Handling generated sentence: \(finalText)")

        if !(textOnly || inQuote) {
            await speaker?.add(text: finalText)
        }

        if !inQuote, let data = finalText.data(using: .utf8) {
            await remote.send(.spokenSentence, content: data)
        }
    }

    private var statePath: URL {
        asset.localStatePath
    }

    private var textPath: URL {
        statePath.appendingPathComponent("text.txt")
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
            messageLog.commitNewText()
            await remote.send(.responseDone, content: emptyData)
            try messageLog.save(to: textPath)
            try await llamaContext?.save(to: statePath)
            log("Saved state to \(statePath.path)")
        }
    }
}
