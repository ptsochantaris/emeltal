import Network
import SwiftUI

@MainActor
@Observable
final class EmelTerm {
    private let remote = EmeltalConnector()

    init() {
        Task {
            for await transmission in remote.setupNetworkListener() {
                log("From server: \(transmission)")
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
