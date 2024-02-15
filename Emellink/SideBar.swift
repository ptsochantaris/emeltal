import Foundation
import SwiftUI

struct SideBar: View {
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
                    PushButton { [weak state] down in
                        guard let state else { return }
                        if down {
                            state.buttonDown()
                        } else {
                            state.buttonUp()
                        }
                    }
                }
            }
        }
    }
}
