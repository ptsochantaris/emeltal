import Network
import SwiftUI

@MainActor
@Observable
final class EmelTerm {
    private let remote = EmeltalConnector()

    init() {
        Task {
            for await (payload, data) in remote.setupNetworkListener() {
                switch payload {
                case .appMode:
                    if let data, let mode = AppMode(data: data) {
                        log("New app mode: \(mode)")
                    }

                case .generatedSentence:
                    if let data, let text = String(data: data, encoding: .utf8) {
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
