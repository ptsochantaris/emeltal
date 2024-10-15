import AVFoundation
import SwiftUI

@MainActor
struct ConversationView: View {
    @Binding var appPhase: AppStack.Phase

    @Bindable var state: ConversationState

    @FocusState private var focusEntryField

    var body: some View {
        #if os(visionOS)
            let spacing: CGFloat = 18
        #else
            let spacing: CGFloat = 8
        #endif

        if state.floatingMode {
            SideBar(state: state, focusEntryField: $focusEntryField)
                .padding(.vertical)
        } else {
            VStack(spacing: 0) {
                HStack(spacing: spacing) {
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
                                .padding(.top, spacing)
                        }

                        WebView(messageLog: state.messageLog)
                    }

                    SideBar(state: state, focusEntryField: $focusEntryField)
                    #if os(visionOS)
                        .padding(.top, 0)
                        .padding(.bottom, spacing - 1)
                    #elseif os(iOS)
                        .padding(.top, 0)
                        .padding(.bottom, spacing)
                    #else
                        .padding(.top, 8)
                        .padding(.bottom, spacing)
                    #endif
                }
                .padding(.horizontal, spacing)

                TextField("Hold \"↓\" to speak, or enter your message here", text: $state.multiLineText)
                    .textFieldStyle(.plain)
                #if os(visionOS)
                    .padding(22)
                    .background(.ultraThinMaterial)
                #else
                    .padding(7)
                    .padding(.horizontal, 5)
                    .background {
                        Capsule()
                            .foregroundStyle(.material)
                    }
                    .padding(.bottom, 8)
                    .padding(.horizontal, spacing)
                #endif
                    .focused($focusEntryField)
                    .onSubmit { [weak state] in
                        guard let state else { return }
                        if state.mode == .waiting {
                            state.send()
                        }
                    }
            }
            .toolbar {
                Button { [weak state] in
                    guard let state else { return }
                    state.textOnly.toggle()
                } label: {
                    HStack(spacing: 0) {
                        Image(systemName: state.textOnly ? "text.bubble" : "speaker.wave.2.bubble")
                        Text(state.textOnly ? "Text-Only" : "Spoken Replies")
                            .padding([.leading, .trailing], 4)
                            .font(.caption)
                    }
                }

                let ready = state.mode.nominal

                Button {
                    Task {
                        await state.shutdown()
                    }
                    appPhase = .selection
                } label: {
                    HStack(spacing: 0) {
                        Image(systemName: "square.grid.3x2")
                        Text("Models")
                            .padding([.leading, .trailing], 4)
                            .font(.caption)
                    }
                }
                .opacity(ready ? 1 : 0.3)
                .allowsHitTesting(ready)

                Button {
                    if state.mode == .waiting || state.mode == .listening(state: .quiet(prefixBuffer: [])) || state.mode == .replying {
                        Task {
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
                .opacity(ready ? 1 : 0.3)
                .allowsHitTesting(ready)
            }
            .navigationTitle("Emeltal — \(state.displayName)")
            .toolbarTitleDisplayMode(.inline)
            .background {
                Image(.background).resizable().aspectRatio(contentMode: .fill)
                    .ignoresSafeArea()
            }
        }
    }
}
