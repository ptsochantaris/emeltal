import AVFoundation
import SwiftUI

@MainActor
struct ContentView: View {
    @Bindable var state: AppState

    @FocusState private var focusEntryField

    var body: some View {
        VStack(spacing: 16) {
            HStack(alignment: .top, spacing: 14) {
                if !state.floatingMode {
                    VStack {
                        if case let .loading(managers) = state.mode {
                            let visibleFetchers = managers.filter(\.phase.shouldShowToUser)
                            if !visibleFetchers.isEmpty {
                                Text("Fetching ML data. This is only needed once per type of model.")
                                    .font(.headline)
                                    .padding(.top)
                                ForEach(visibleFetchers) {
                                    ManagerRow(manager: $0)
                                }
                            }
                        }

                        if state.shouldPromptForIdealVoice {
                            IdealVoicePrompt(shouldPromptForIdealVoice: $state.shouldPromptForIdealVoice)
                        }

                        WebView(messageLog: state.messageLog)

                        TextField("Hold \"↓\" to speak, or enter your message here", text: $state.multiLineText)
                            .textFieldStyle(.roundedBorder)
                            .focused($focusEntryField)
                            .onSubmit {
                                if state.mode == .waiting {
                                    state.send()
                                }
                            }
                            .padding(.bottom)
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
                                if state.mode == .waiting || state.mode == .listening(state: .quiet(prefixBuffer: [])) || state.mode == .replying {
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
                    .padding([.top, .bottom])
            }
        }
        .padding([.leading, .trailing])
        .navigationTitle("Emeltal — \(state.displayName)")
        .task {
            do {
                try await state.boot()
            } catch {
                log("Error booting: \(error.localizedDescription)")
                fatalError(error.localizedDescription)
            }
        }
    }
}
