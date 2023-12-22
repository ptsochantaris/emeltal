import Combine
import Network
import SwiftUI

@MainActor
@Observable
final class EmelTerm {
    private let remote = EmeltalConnector()
    private let speaker = try? Speaker()
    private let mic = Mic()

    var connectionState = EmeltalConnector.State.boot
    private var connectionStateObservation: Cancellable!

    var remoteActivationState = ActivationState.notListening

    var remoteAppMode = AppMode.booting {
        didSet {
            if let speaker, oldValue != remoteAppMode {
                remoteAppMode.audioFeedback(using: speaker)
            }
        }
    }

    init() {
        Task {
            await go()
        }
    }

    func buttonDown() {
        Task {
            await remote.send(.buttonDown, content: emptyData)
        }
    }

    func buttonUp() {
        Task {
            await remote.send(.buttonUp, content: emptyData)
        }
    }

    func toggleListeningMode() {
        Task {
            await remote.send(.toggleListeningMode, content: emptyData)
        }
    }

    private func go() async {
        connectionStateObservation = remote.statePublisher.receive(on: DispatchQueue.main).sink { [weak self] state in
            self?.connectionState = state
        }

        let stream = await remote.startClient()
        for await nibble in stream {
            switch nibble.payload {
            case .appMode:
                if let data = nibble.data, let mode = AppMode(data: data) {
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

            case .buttonDown, .buttonUp, .recordedSpeech, .recordedSpeechLast, .toggleListeningMode:
                break

            case .unknown:
                log("Warning: Unknown message from server")
            }
        }
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
                }

                if mode.showAlwaysOn {
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

    var body: some Scene {
        WindowGroup(id: "Emelterm") {
            ContentView(state: state)
                .frame(maxWidth: .infinity)
                .background(Image(.canvas).resizable().ignoresSafeArea())
                .preferredColorScheme(.dark)
        }
    }
}
