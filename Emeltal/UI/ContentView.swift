import AVFoundation
import MarkdownUI
import SwiftUI

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
                                Text("Fetching ML data. This is only needed once per type of model.")
                                    .font(.headline)
                                ForEach(visibleFetchers) {
                                    ManagerRow(manager: $0)
                                }
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

                if !state.isRemoteConnected {
                    SideBar(state: state, focusEntryField: $focusEntryField)
                }
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
