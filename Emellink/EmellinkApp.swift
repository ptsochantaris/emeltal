import AVFoundation
import Combine
import Network
import SwiftUI

// Stub
final class AssetManager {}

@MainActor
@Observable
final class EmelTerm {
    private let remote = EmeltalConnector()
    private let speaker = try? Speaker()
    private let mic = Mic()

    var connectionState = EmeltalConnector.State.boot
    private var connectionStateObservation: Cancellable!

    private var micObservation: Cancellable!

    var remoteActivationState = ActivationState.notListening

    var remoteAppMode = AppMode.booting {
        didSet {
            if oldValue != remoteAppMode {
                log("New remote state: \(remoteAppMode)")
                if let speaker {
                    remoteAppMode.audioFeedback(using: speaker)
                }
                Task {
                    if case .alwaysOn = remoteAppMode {
                        await self.startMic()
                    } else if case .waiting = remoteAppMode, case .alwaysOn = oldValue {
                        await self.endMic(processResult: false)
                    }
                }
            }
        }
    }

    init() {
        #if os(iOS)
            let av = AVAudioSession.sharedInstance()
            try? av.setCategory(.playAndRecord, options: [.duckOthers, .defaultToSpeaker])
            try? av.overrideOutputAudioPort(.speaker)
            try? av.setPreferredInputNumberOfChannels(1)
            try? av.setActive(true)
        #endif
    }

    func invalidateConnection() {
        Task { @NetworkActor in
            remote.invalidate()
        }
    }

    func restoreConnectionIfNeeded() {
        Task { @NetworkActor in
            await go()
        }
    }

    func buttonDown() {
        Task {
            await startMic()
            await remote.send(.buttonDown, content: emptyData)
        }
    }

    private func startMic() async {
        await speaker?.cancelIfNeeded()
        try? await mic.start()
    }

    private func endMic(processResult: Bool) async {
        if processResult {
            if let floats = try? await mic.stop(), floats.count > 100 {
                let speech = floats.withUnsafeBytes { floatPointer in
                    Data(bytes: floatPointer.baseAddress!, count: floatPointer.count)
                }
                await remote.send(.recordedSpeech, content: speech)
            }
        } else {
            _ = try? await mic.stop()
        }
    }

    func buttonUp() {
        Task {
            await endMic(processResult: true)
            await remote.send(.buttonUp, content: emptyData)
        }
    }

    func toggleListeningMode() {
        Task {
            await remote.send(.toggleListeningMode, content: emptyData)
        }
    }

    private func go() async {
        micObservation = mic.statePublisher.receive(on: DispatchQueue.main).sink { [weak self] newState in
            guard let self else { return }
            if remoteActivationState == .voiceActivated, case let .alwaysOn(myState) = remoteAppMode, newState != myState, case .quiet = newState {
                Task {
                    await self.endMic(processResult: true)
                }
            }
        }

        connectionStateObservation = remote.statePublisher.receive(on: DispatchQueue.main).sink { [weak self] state in
            self?.connectionState = state
        }

        let stream = await remote.startClient()
        for await nibble in stream {
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

            case .buttonDown, .buttonUp, .heartbeat, .recordedSpeech, .toggleListeningMode:
                break

            case .unknown:
                log("Warning: Unknown message from server")
            }
        }
        log("[Connector] Incoming stream done")
    }
}

struct ContentView: View {
    let state: EmelTerm

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
                ModeView(mode: state.remoteAppMode)
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
    private let state = EmelTerm()

    @Environment(\.scenePhase) var scenePhase

    var body: some Scene {
        WindowGroup(id: "Emelterm") {
            ContentView(state: state)
                .frame(maxWidth: .infinity)
                .background(Image(.canvas).resizable().ignoresSafeArea())
                .preferredColorScheme(.dark)
                .onChange(of: scenePhase) { _, newValue in
                    if newValue == .background {
                        state.invalidateConnection()
                    } else if newValue == .active {
                        state.restoreConnectionIfNeeded()
                    }
                }
        }
    }
}