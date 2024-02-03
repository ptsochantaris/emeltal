import Foundation
import SwiftUI

struct SideBar: View {
    let state: AppState
    @FocusState.Binding var focusEntryField: Bool

    @State private var originalPos: CGPoint? = nil

    var body: some View {
        VStack(spacing: 0) {
            Button {
                state.floatingMode.toggle()
                if !state.floatingMode {
                    focusEntryField = true
                }
            } label: {
                Image(systemName: state.floatingMode ? "chevron.compact.up" : "chevron.compact.down")
                    .font(.largeTitle)
                    .padding()
            }
            .buttonStyle(.borderless)

            if let statusMessage = state.statusMessage {
                Text(statusMessage)
                    .foregroundStyle(.accent)
                    .multilineTextAlignment(.center)
                    .font(.caption2.bold())
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.vertical, 4)
                    .padding(.horizontal, 8)
                    .background {
                        Capsule(style: .continuous)
                            .stroke(.accent, lineWidth: 1)
                    }
            }

            if state.isRemoteConnected {
                Text("LINKED")
                    .padding(.vertical, 4)
                    .padding(.horizontal, 8)
                    .foregroundStyle(.black)
                    .font(.caption2.bold())
                    .background(Capsule(style: .continuous).foregroundStyle(.widgetForeground))
                    .padding(.bottom)
            }

            if state.mode.showGenie {
                Genie(show: state.mode.showGenie)
                    .padding(.top, -4)
                    .padding(.bottom, -1)
                    .gesture(DragGesture().onChanged { value in
                        dragged(by: value)
                    }.onEnded { _ in
                        dragEnd()
                    })

            } else {
                let va = state.activationState == .voiceActivated
                if state.mode.showAlwaysOn {
                    Button("ALWAYS ON") {
                        if va {
                            state.switchToPushButton()
                        } else {
                            state.switchToVoiceActivated()
                        }
                    }
                    .font(.caption2.bold())
                    .buttonStyle(PlainButtonStyle())
                    .foregroundStyle(va ? .accent : .widgetForeground)
                    .padding(.vertical, 4)
                    .padding(.horizontal, 8)
                    .background {
                        Capsule(style: .continuous)
                            .stroke(va ? .accent : .widgetForeground, lineWidth: 1)
                    }
                    .padding(1)
                }

                if state.mode.showAlwaysOn, state.isRemoteConnected {
                    Spacer()
                        .frame(height: 12)
                }

                Color(.clear)
                    .contentShape(Rectangle())
                    .gesture(DragGesture().onChanged { value in
                        dragged(by: value)
                    }.onEnded { _ in
                        dragEnd()
                    })
            }

            Assistant(state: state)
                .padding(.bottom, 4)
                .padding([.leading, .trailing])
                .foregroundColor(.widgetForeground.opacity(state.mode == .waiting ? 0.7 : 0.8))
                .frame(height: assistantWidth) // make it square
        }
        .foregroundColor(.widgetForeground)
        .background {
            Image(.canvas).resizable().aspectRatio(contentMode: .fill)
        }
        .contentShape(Rectangle())
        .frame(width: assistantWidth)
        .cornerRadius(44)
        .shadow(radius: state.floatingMode ? 0 : 6)
    }

    private func dragged(by value: DragGesture.Value) {
        guard let window = NSApplication.shared.keyWindow else {
            return
        }
        var frame = window.frame
        let p = window.frame.origin
        frame.origin = CGPoint(x: p.x + value.translation.width, y: p.y - value.translation.height)
        originalPos = frame.origin
        window.setFrame(frame, display: false)
    }

    private func dragEnd() {
        originalPos = nil
    }
}
