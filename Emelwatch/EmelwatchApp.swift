import SwiftUI

struct ContentView: View {
    let state: LinkState

    var body: some View {
        VStack {
            let connectionState = state.connectionState

            if connectionState.isConnected {
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
            } else {
                Text(connectionState.label.uppercased())
                    .padding(8)
                    .padding([.leading, .trailing], 2)
                    .background {
                        Capsule(style: .continuous)
                            .foregroundStyle(connectionState.color)
                    }
                    .font(.caption.bold())
                    .foregroundStyle(.black)
            }
        }
        .padding()
    }
}

@main
@MainActor
struct Emelwatch_Watch_AppApp: App {
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
