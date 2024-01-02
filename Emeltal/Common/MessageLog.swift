import Foundation
import MarkdownUI
import SwiftUI

private struct Identifier: Identifiable, Hashable {
    let id: String
}

struct MessageLog: View {
    @Binding var messageLog: String
    let padding: Bool

    private let bottomId = Identifier(id: "bottomId")

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(spacing: 0) {
                    Markdown(MarkdownContent(messageLog))
                        .textSelection(.enabled)
                        .markdownTheme(.docC)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                        .padding(padding ? .all : [])
                    Spacer()
                        .id(bottomId)
                }
            }
            .onChange(of: messageLog) { _, _ in
                proxy.scrollTo(bottomId)
            }
        }
    }
}
