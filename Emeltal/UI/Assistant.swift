import Foundation
import SwiftUI

struct Assistant: View {
    let state: AppState

    var body: some View {
        ZStack {
            ModeView(modeProvider: state)

            if state.mode.pushButtonActive {
                PushButton { down in
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
