import MarkdownUI
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

            if state.mode.showGenie {
                Genie(state: state)
                    .padding(.top, -4)
                    .padding(.bottom, -1)
                    .gesture(DragGesture().onChanged { value in
                        dragged(by: value)
                    }.onEnded { _ in
                        dragEnd()
                    })

            } else {
                let va = state.listenState == .voiceActivated
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
        .background(Image(.canvas).resizable().aspectRatio(contentMode: .fill))
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

@MainActor
struct ContentView: View {
    @Bindable var state: AppState

    @FocusState private var focusEntryField

    private struct Identifier: Identifiable, Hashable {
        let id: String
    }

    private let bottomId = Identifier(id: "bottomId")

    var body: some View {
        VStack(spacing: 16) {
            HStack(alignment: .top, spacing: 14) {
                if !state.floatingMode {
                    VStack {
                        if case let .loading(managers) = state.mode {
                            let visibleFetchers = managers.filter(\.phase.shouldShowToUser)
                            if !visibleFetchers.isEmpty {
                                Text("Fetching ML models. This can take a while, but is only needed once.")
                                    .font(.headline)
                                ForEach(visibleFetchers) {
                                    ManagerRow(manager: $0)
                                }
                            }
                        } else if let message = state.statusMessage {
                            Text(message)
                                .foregroundStyle(.black)
                                .font(.headline)
                                .padding()
                                .background {
                                    Capsule(style: .continuous)
                                        .foregroundStyle(.accent)
                                }
                        }

                        ScrollViewReader { proxy in
                            ScrollView {
                                VStack(spacing: 0) {
                                    Markdown(MarkdownContent(state.messageLog))
                                        .textSelection(.enabled)
                                        .markdownTheme(.docC)
                                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                                    Spacer()
                                        .id(bottomId)
                                }
                            }
                            .onChange(of: state.messageLog) { _, _ in
                                proxy.scrollTo(bottomId)
                            }
                        }

                        TextField("Hold \"↓\" to speak, or enter your message here", text: $state.multiLineText)
                            .textFieldStyle(.roundedBorder)
                            .focused($focusEntryField)
                            .onSubmit {
                                if state.mode == .waiting {
                                    state.send()
                                }
                            }
                    }
                    .toolbar {
                        Button {
                            state.textOnly.toggle()
                        } label: {
                            HStack(spacing: 0) {
                                Image(systemName: state.textOnly ? "text.bubble" : "speaker.wave.2.bubble")
                                Text(state.textOnly ? "Text-Only Replies" : "Spoken Replies")
                                    .padding([.leading, .trailing], 4)
                                    .font(.caption)
                            }
                        }

                        Button {
                            Task {
                                if state.mode == .waiting {
                                    try? await state.reset()
                                }
                            }
                        } label: {
                            HStack(spacing: 0) {
                                Image(systemName: "clear")
                                Text("Reset")
                                    .padding([.leading, .trailing], 4)
                                    .font(.caption)
                            }
                        }
                        .keyboardShortcut(KeyEquivalent("k"), modifiers: .command)
                    }
                }

                SideBar(state: state, focusEntryField: $focusEntryField)
            }
        }
        .padding()
        .navigationTitle("Emeltal — \(state.displayName)")
        .task {
            do {
                try await state.boot()
            } catch {
                fatalError(error.localizedDescription)
            }
        }
    }
}
