import Combine
import Network
import SwiftUI

@MainActor
@Observable
final class EmelTerm {
    private let remote = EmeltalConnector()

    var connectionState = EmeltalConnector.State.boot
    private var connectionStateObservation: Cancellable!
    private let speaker = try? Speaker()

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

    private func go() async {
        connectionStateObservation = await remote.statePublisher.receive(on: DispatchQueue.main).sink { [weak self] state in
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

            case .recordedSpeech, .recordedSpeechLast:
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
            switch state.connectionState {
            case .boot, .searching, .unConnected:
                Text("Searching")
            case .connected:
                Text("Connected")
            case .connecting:
                Text("Connecting")
            case let .error(error):
                Text("Error: \(error.localizedDescription)")
            }

            let mode = state.remoteAppMode
            if mode.showAlwaysOn {
                Text("Always On")
            }

            if mode.showGenie {
                Text("Genie")
            }
        }
        .padding()
    }
}

@main
@MainActor
struct EmeltermApp: App {
    let state = EmelTerm()

    var body: some Scene {
        WindowGroup {
            ContentView(state: state)
        }
    }
}
