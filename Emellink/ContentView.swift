import Foundation
import SwiftUI

struct ContentView: View {
    @Bindable var state: LinkState
    @State private var visibility = NavigationSplitViewVisibility.doubleColumn
    @State private var preferredColumn = NavigationSplitViewColumn.sidebar

    @Environment(\.horizontalSizeClass) private var horizontalSizeClass

    var body: some View {
        NavigationSplitView(columnVisibility: $visibility, preferredCompactColumn: $preferredColumn) {
            SideBar(state: state)
                .padding()
                .background(Image(.canvas).resizable().aspectRatio(contentMode: .fill).ignoresSafeArea())
                .toolbar {
                    if horizontalSizeClass == .compact {
                        ToolbarItem(placement: .topBarTrailing) {
                            Button {
                                preferredColumn = .detail
                            } label: {
                                Image(systemName: "text.justify.leading")
                            }
                        }
                    }
                }
                .navigationTitle("Emeltal")
                .toolbarTitleDisplayMode(.inline)

        } detail: {
            MessageLog(messageLog: $state.messageLog, padding: true)
                .background(Image(.canvas).resizable().aspectRatio(contentMode: .fill).ignoresSafeArea().opacity(0.7))
                .navigationBarBackButtonHidden()
                .toolbar {
                    if horizontalSizeClass == .compact {
                        ToolbarItem(placement: .topBarLeading) {
                            Button {
                                preferredColumn = .sidebar
                            } label: {
                                Image(systemName: state.mode.iconImageName)
                            }
                        }
                    }

                    ToolbarItem {
                        Button {
                            if state.mode == .waiting || state.mode == .listening(state: .quiet(prefixBuffer: [])) || state.mode == .replying {
                                state.requestReset()
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
