import Foundation
import SwiftUI

@main
@MainActor
struct EmeltalApp: App {
    enum Phase {
        case selection, conversation(ConversationState)
    }

    @State private var appPhase = Phase.selection

    var body: some Scene {
        WindowGroup {
            NavigationStack {
                switch appPhase {
                case .selection:
                    ModelPicker(appPhase: $appPhase)
                case let .conversation(state):
                    ConversationView(appPhase: $appPhase, state: state)
                }
            }
            .preferredColorScheme(.dark)
        }
        .defaultSize(width: 1000, height: 1000)
    }
}
