import Network
import SwiftUI

@MainActor
@Observable
final class EmelTerm {
    private let remote = EmeltalConnector()

    init() {
        Task {
            let stream = await remote.startClient()
            for await nibble in stream {
                switch nibble.payload {
                case .appMode:
                    if let data = nibble.data, let mode = AppMode(data: data) {
                        log("New app mode: \(mode)")
                    }

                case .generatedSentence:
                    if let data = nibble.data, let text = String(data: data, encoding: .utf8) {
                        log("Generated sentence: \(text)")
                    }

                case .recordedSpeech, .recordedSpeechLast:
                    break

                case .unknown:
                    log("Warning: Unknown message from server")
                }
            }
        }
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
