import SwiftUI

@main
struct EmeltermApp: App {
    private let state = LinkState()

    @Environment(\.scenePhase) var scenePhase

    var body: some Scene {
        WindowGroup(id: "Emelterm") {
            ContentView(state: state)
                .frame(maxWidth: .infinity)
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
