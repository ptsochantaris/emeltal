import AVFoundation
import Combine
import Network
import SwiftUI

// Stub
final class AssetManager {}

@MainActor
@Observable
final class Emellink: ModeProvider {
    private let remote = EmeltalConnector()
    private let speaker = try? Speaker()
    private let mic = Mic()

    var connectionState = EmeltalConnector.State.boot
    private var connectionStateObservation: Cancellable!

    private var micObservation: Cancellable!

    var remoteActivationState = ActivationState.button {
        didSet {
            if oldValue != remoteActivationState {
                log("New remote activation state: \(remoteActivationState)")
            }
        }
    }

    var mode: AppMode { remoteAppMode }

    var remoteAppMode = AppMode.booting {
        didSet {
            if oldValue != remoteAppMode {
                log("New remote state: \(remoteAppMode)")
                if let speaker {
                    remoteAppMode.audioFeedback(using: speaker)
                }
                Task {
                    if case .listening = remoteAppMode {
                        await self.startMic()
                    } else if case .waiting = remoteAppMode, case .listening = oldValue {
                        await self.endMic(sendData: false)
                    } else if case .booting = remoteAppMode {
                        await endMic(sendData: false)
                    }
                }
            }
        }
    }

    func invalidateConnectionIfNotNeeded() async {
        if case .voiceActivated = remoteActivationState {
            return
        }
        await remote.invalidate()
    }

    func restoreConnectionIfNeeded() async {
        if case .connected = connectionState {
            return
        }
        Task { @NetworkActor in
            await connectionLoop()
        }
    }

    func buttonUp() {
        Task {
            if case .voiceActivated = remoteActivationState {
                return
            }
            await endMic(sendData: true)
        }
    }

    func buttonDown() {
        Task {
            await speaker?.cancelIfNeeded()
            await remote.send(.buttonDown, content: emptyData)
        }
    }

    private func startMic() async {
        await speaker?.cancelIfNeeded()
        try? await mic.start()
    }

    private func endMic(sendData: Bool) async {
        if sendData {
            if let floats = try? await mic.stop(temporary: remoteActivationState == .voiceActivated), floats.count > 100 {
                let speech = floats.withUnsafeBytes { floatPointer in
                    Data(bytes: floatPointer.baseAddress!, count: floatPointer.count)
                }
                await remote.send(.recordedSpeech, content: speech)
            }
            await remote.send(.recordedSpeechDone, content: emptyData)
        } else {
            _ = try? await mic.stop(temporary: remoteActivationState == .voiceActivated)
        }
    }

    func toggleListeningMode() {
        Task {
            await remote.send(.toggleListeningMode, content: emptyData)
        }
    }

    private func connectionLoop() async {
        log("Starting main connection loop")

        #if os(iOS)
            let av = AVAudioSession.sharedInstance()
            try? av.setCategory(.playAndRecord, options: [.duckOthers, .defaultToSpeaker])
            try? av.setPreferredInputNumberOfChannels(1)
            try? av.setActive(true)
        #endif

        micObservation = mic.statePublisher.receive(on: DispatchQueue.main).sink { [weak self] newState in
            guard let self else { return }
            if case let .listening(micState) = remoteAppMode {
                if newState != micState {
                    log("New mic state: \(newState)")
                    remoteAppMode = .listening(state: newState)
                    if case .quiet = newState, remoteActivationState == .voiceActivated {
                        Task {
                            await self.endMic(sendData: true)
                        }
                    }
                }
            }
        }

        connectionStateObservation = remote.statePublisher.receive(on: DispatchQueue.main).sink { [weak self] newState in
            guard let self else { return }
            if case .connected = newState {
                // all good
            } else if case .connected = connectionState {
                micObservation = nil
                connectionStateObservation = nil
                remoteAppMode = .booting
            }
            connectionState = newState
        }

        let stream = await remote.startClient()
        for await nibble in stream {
            log("From server: \(nibble.payload)")

            switch nibble.payload {
            case .appMode:
                if let data = nibble.data, let mode = AppMode(data: data) {
                    await speaker?.waitForCompletion()
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

            case .generatedSentence:
                if let data = nibble.data, let text = String(data: data, encoding: .utf8) {
                    await speaker?.add(text: text)
                }

            case .buttonDown, .heartbeat, .recordedSpeech, .recordedSpeechDone, .toggleListeningMode:
                break

            case .unknown:
                log("Warning: Unknown message from server")
            }
        }
        log("Incoming stream done")
    }
}

struct ContentView: View {
    let state: Emellink

    var body: some View {
        VStack {
            let connectionState = state.connectionState
            let mode = state.remoteAppMode
            let activation = state.remoteActivationState

            HStack {
                if !connectionState.isConnected {
                    Text(connectionState.label.uppercased())
                        .padding(8)
                        .padding([.leading, .trailing], 2)
                        .background {
                            Capsule(style: .continuous)
                                .foregroundStyle(connectionState.color)
                        }
                } else if mode.showAlwaysOn {
                    let on = activation == .voiceActivated
                    Text("Always On".uppercased())
                        .padding(8)
                        .padding([.leading, .trailing], 2)
                        .background {
                            Capsule(style: .continuous)
                                .stroke(style: StrokeStyle())
                        }
                        .onTapGesture {
                            state.toggleListeningMode()
                        }
                        .foregroundStyle(on ? .accent : .secondary)
                }
            }
            .font(.caption.bold())
            .foregroundStyle(.black)

            if mode.showGenie {
                Genie(show: mode.showGenie)
            } else {
                Spacer()
                    .frame(maxHeight: .infinity)
            }

            ZStack {
                ModeView(modeProvider: state)
                    .foregroundStyle(.primary)
                    .frame(maxWidth: assistantWidth)
                    .layoutPriority(2)

                if state.remoteAppMode.pushButtonActive {
                    PushButton { down in
                        if down {
                            state.buttonDown()
                        } else {
                            state.buttonUp()
                        }
                    }
                }
            }
        }
        .padding()
    }
}

@main
@MainActor
struct EmeltermApp: App {
    private let state = Emellink()

    @Environment(\.scenePhase) var scenePhase

    var body: some Scene {
        WindowGroup(id: "Emelterm") {
            ContentView(state: state)
                .frame(maxWidth: .infinity)
                .background(Image(.canvas).resizable().ignoresSafeArea())
                .preferredColorScheme(.dark)
                .onChange(of: scenePhase) { _, newValue in
                    Task {
                        if newValue == .background {
                            await state.invalidateConnectionIfNotNeeded()
                        } else if newValue == .active {
                            await state.restoreConnectionIfNeeded()
                        }
                    }
                }
        }
    }
}
