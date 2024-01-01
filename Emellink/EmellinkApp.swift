import SwiftUI

struct ContentView: View {
    @Bindable var state: LinkState

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

            } else if state.shouldPromptForIdealVoice {
                ZStack {
                    IdealVoicePrompt(shouldPromptForIdealVoice: $state.shouldPromptForIdealVoice)
                }
                .frame(maxHeight: .infinity)

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
    private let state = LinkState()

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
