import Foundation
import SwiftUI

struct Assistant: View {
    let state: ConversationState

    var body: some View {
        ZStack {
            ModeView(modeProvider: state)

            if state.mode.pushButtonActive {
                PushButton { [weak state] down in
                    guard let state else { return }
                    if down {
                        state.pushButtonDown()
                    } else {
                        state.pushButtonUp()
                    }
                }
            }
        }
    }
}
