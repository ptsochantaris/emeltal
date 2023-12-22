import Foundation
import SwiftUI

struct Assistant: View {
    let state: AppState

    var body: some View {
        ZStack {
            ModeView(mode: state.mode)

            if state.mode != .loading(managers: []) {
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
