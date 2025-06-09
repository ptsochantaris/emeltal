import AVFoundation
import Foundation
import Network
import SwiftUI

// Stub
final class AssetManager {}

@MainActor
@Observable
final class LinkState: ModeProvider, ConversationHandler {
    private let remote = ClientConnector()
    private let speaker = try! Speaker()
    private let mic = Mic()

    private var connectionStateObservation: Task<Void, Never>!
    private var micObservation: Task<Void, Never>!

    let messageLog = MessageLog()
    var multiLineText = ""
    var shouldPromptForIdealVoice = false
    var connectionState = EmeltalConnector.State.boot

    var textOnly = false {
        didSet {
            Task { [textOnly] in
                await speaker.setMute(textOnly)
            }
        }
    }

    var remoteActivationState = ActivationState.button {
        didSet {
            if oldValue != remoteActivationState {
                log("New remote activation state: \(remoteActivationState)")
            }
        }
    }

    var buttonPushed = false {
        didSet {
            guard oldValue != buttonPushed, mode.pushButtonActive else { return }
            let pushed = buttonPushed
            Task {
                if pushed {
                    await speaker.cancelIfNeeded()
                    await remote.send(.buttonTap, content: emptyData)
                } else {
                    if case .voiceActivated = remoteActivationState {
                        return
                    }
                    await endMic(sendData: true)
                }
            }
        }
    }

    var mode: AppMode { remoteAppMode }

    var remoteAppMode = AppMode.booting {
        didSet {
            if oldValue != remoteAppMode {
                Task {
                    log("New remote state: \(remoteAppMode)")
                    if case .listening = remoteAppMode {
                        await self.startMic()
                    } else if case .waiting = remoteAppMode, case .listening = oldValue {
                        await self.endMic(sendData: false)
                    } else if case .booting = remoteAppMode {
                        await endMic(sendData: false)
                    }
                    remoteAppMode.audioFeedback(using: speaker)
                    if remoteAppMode == .waiting, oldValue != .waiting {
                        Task {
                            await speaker.play(effect: .startListening)
                        }
                    }
                }
            }
        }
    }

    func invalidateConnectionIfNotNeeded() async {
        if case .voiceActivated = remoteActivationState {
            return
        }
        await remote.shutdown()
    }

    func send() async {
        if let textData = multiLineText.data(using: .utf8), !textData.isEmpty {
            await remote.send(.textInput, content: textData)
        }
        multiLineText = ""
    }

    private func boot() async {
        log("First boot")

        #if canImport(AppKit)
            await AVCaptureDevice.requestAccess(for: .audio)
        #else
            let av = AVAudioSession.sharedInstance()
            #if os(iOS)
                try? av.setCategory(.playAndRecord, options: [.duckOthers, .defaultToSpeaker])
                try? av.setPreferredInputNumberOfChannels(1)
            #else
                try? av.setCategory(.playAndRecord, options: [.duckOthers])
            #endif
            try? av.setActive(true)
            await AVAudioApplication.requestRecordPermission()
        #endif

        shouldPromptForIdealVoice = await !speaker.havePreferredVoice

        setupMicObservation()

        setupConnectionObservation()

        try? await withThrowingDiscardingTaskGroup { group in
            group.addTask { [weak self] in
                await self?.mic.warmup()
            }
            group.addTask { [weak self] in
                try await self?.speaker.warmup()
            }
        }
    }

    private var booted = false
    func restoreConnectionIfNeeded() async {
        if !booted {
            booted = true
            await boot()
        }

        if case .connected = connectionState {
            return
        }

        Task { @NetworkActor in
            await connectionLoop()
        }
    }

    func shutdown() async {
        log("Shutting down app state…")
        micObservation?.cancel()
        connectionStateObservation.cancel()
        await remote.shutdown()
        await speaker.shutdown()
    }

    deinit {
        log("LinkState deinit")
    }

    func requestReset() {
        Task {
            await remote.send(.requestReset, content: emptyData)
        }
    }

    private func startMic() async {
        await speaker.cancelIfNeeded()
        try? await mic.start(detectVoice: remoteActivationState == .voiceActivated)
    }

    private func endMic(sendData: Bool) async {
        let floats = if remoteActivationState == .voiceActivated {
            try? await mic.pause()
        } else {
            try? await mic.stop()
        }
        if sendData {
            if let floats, floats.count > 100 {
                let speech = floats.withUnsafeBytes { floatPointer in
                    Data(bytes: floatPointer.baseAddress!, count: floatPointer.count)
                }
                await remote.send(.recordedSpeech, content: speech)
            }
            await remote.send(.recordedSpeechDone, content: emptyData)
        }
    }

    func toggleListeningMode() {
        Task {
            await remote.send(.toggleListeningMode, content: emptyData)
        }
    }

    private func setupMicObservation() {
        micObservation = Task {
            for await newState in mic.stateStream.stream {
                if case let .listening(micState) = remoteAppMode {
                    if newState != micState {
                        log("New mic state: \(newState)")
                        remoteAppMode = .listening(state: newState)
                        if case .quiet = newState, remoteActivationState == .voiceActivated {
                            Task {
                                await endMic(sendData: true)
                            }
                        }
                    }
                }
            }
        }
    }

    private func setupConnectionObservation() {
        connectionStateObservation = Task {
            for await newState in remote.stateStream.stream {
                log("Received a state update: \(newState)")
                if case .connected = newState {
                    // all good
                } else if case .connected = connectionState {
                    remoteAppMode = .booting
                }
                connectionState = newState
            }
        }
    }

    private func connectionLoop() async {
        log("Starting main connection loop")

        let stream = await remote.startClient()
        for await nibble in stream {
            log("From server: \(nibble.payload)")

            switch nibble.payload {
            case .appMode:
                if let data = nibble.data, let mode = AppMode(data: data) {
                    await speaker.waitForCompletion()
                    withAnimation {
                        remoteAppMode = mode
                    }
                }

            case .appActivationState:
                if let data = nibble.data, let state = ActivationState(data: data) {
                    withAnimation {
                        remoteActivationState = state
                    }
                }

            case .spokenSentence:
                if let data = nibble.data, let text = String(data: data, encoding: .utf8) {
                    await speaker.add(text: text)
                }

            case .textInitial:
                if let textData = nibble.data, textData.count > 1, let text = String(data: textData, encoding: .utf8) {
                    messageLog.setHistory(text)
                } else {
                    messageLog.setHistory("")
                }

            case .textDiff:
                if let textData = nibble.data, let text = String(data: textData, encoding: .utf8) {
                    messageLog.appendText(text)
                }

            case .responseDone:
                messageLog.commitNewText()

            case .buttonTap, .heartbeat, .hello, .recordedSpeech, .recordedSpeechDone, .requestReset, .textInput, .toggleListeningMode:
                break

            case .unknown:
                log("Warning: Unknown message from server")
            }
        }
        log("Incoming stream done")
    }
}
