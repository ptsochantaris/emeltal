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

    func toggleListeningMode() {
        Task {
            await remote.send(.toggleListeningMode, content: Data([0]))
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
                    log("New app mode: \(mode)")
                    remoteAppMode = mode
                }

            case .generatedSentence:
                if let data = nibble.data, let text = String(data: data, encoding: .utf8) {
                    log("Generated sentence: \(text)")
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

            Group {
                Text(connectionState.label.uppercased())
                    .padding(8)
                    .background {
                        Capsule(style: .continuous)
                            .foregroundStyle(connectionState.color)
                    }

                if mode.showAlwaysOn {
                    let on = if case .listening = mode {
                        true
                    } else {
                        false
                    }

                    let title = on ? "Always On" : "Push To Speak"
                    Text(title.uppercased())
                        .padding(8)
                        .background {
                            Capsule(style: .continuous)
                                .foregroundStyle(on ? .accent : .secondary)
                        }
                        .onTapGesture {
                            state.toggleListeningMode()
                        }
                }
            }
            .font(.caption.bold())

            if mode.showGenie {
                Genie(show: mode.showGenie)
            } else {
                Spacer()
            }

            ModeView(mode: state.remoteAppMode)
                .foregroundStyle(.primary)
                .frame(maxWidth: assistantWidth)
        }
        .padding()
    }
}

@main
@MainActor
struct EmeltermApp: App {
    private let state = EmelTerm()

    var body: some Scene {
        Window("Emelterm", id: "Emelterm") {
            ContentView(state: state)
                .frame(maxWidth: .infinity)
                .background(Image(.canvas).resizable())
                .preferredColorScheme(.dark)
        }
    }
}
