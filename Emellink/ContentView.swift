import Foundation
import SwiftUI

struct ContentView: View {
    @Bindable var state: LinkState

    @State private var visibility = NavigationSplitViewVisibility.doubleColumn
    @State private var preferredColumn = NavigationSplitViewColumn.sidebar

    @FocusState private var focusEntryField

    @Environment(\.horizontalSizeClass) private var horizontalSizeClass

    var body: some View {
        NavigationSplitView(columnVisibility: $visibility, preferredCompactColumn: $preferredColumn) {
            SideBar(state: state)
                .padding()
                .frame(maxWidth: .infinity)
                .background(Image(.canvas).resizable().ignoresSafeArea())
                .toolbar {
                    #if os(iOS) || os(visionOS)
                        if horizontalSizeClass == .compact {
                            ToolbarItem(placement: .topBarTrailing) {
                                Button {
                                    preferredColumn = .detail
                                } label: {
                                    Image(systemName: "text.justify.leading")
                                }
                            }
                        }
                    #endif
                }
                .navigationTitle("Emeltal")
                .toolbarTitleDisplayMode(.inline)

        } detail: {
            VStack(spacing: 0) {
                WebView(messageLog: state.messageLog)
                    .navigationBarBackButtonHidden()
                    .toolbar {
                        #if os(iOS) || os(visionOS)
                            if horizontalSizeClass == .compact {
                                ToolbarItem(placement: .topBarLeading) {
                                    Button {
                                        preferredColumn = .sidebar
                                    } label: {
                                        Image(systemName: state.mode.iconImageName)
                                    }
                                }
                            }
                        #endif

                        ToolbarItem {
                            Button {
                                state.textOnly.toggle()
                            } label: {
                                Image(systemName: state.textOnly ? "text.bubble" : "speaker.wave.2.bubble")
                            }
                        }

                        ToolbarItem {
                            Button {
                                if state.mode == .waiting || state.mode == .listening(state: .quiet(prefixBuffer: [])) || state.mode == .replying {
                                    state.requestReset()
                                }
                            } label: {
                                Image(systemName: "clear")
                            }
                            .keyboardShortcut(KeyEquivalent("k"), modifiers: .command)
                        }
                    }

                TextField("Enter your message here", text: $state.multiLineText)
                #if os(macOS)
                    .textFieldStyle(.roundedBorder)
                #else
                    .submitLabel(.send)
                    .padding()
                    .background(.black.opacity(0.3))
                    .onTapGesture {
                        focusEntryField = true
                    }
                #endif
                    .onSubmit {
                        if state.mode == .waiting {
                            Task {
                                await state.send()
                            }
                        }
                    }
                    .focused($focusEntryField)
            }
            .background(Image(.canvas).resizable().ignoresSafeArea().opacity(0.7))
            .navigationTitle("Conversation")
            .toolbarTitleDisplayMode(.inline)
        }
        .navigationSplitViewStyle(.balanced)
        .overlay {
            if state.shouldPromptForIdealVoice {
                IdealVoicePrompt(shouldPromptForIdealVoice: $state.shouldPromptForIdealVoice)
                    .frame(maxWidth: 512, maxHeight: .infinity)
            }
        }
    }
}
