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
            .frame(idealWidth: 1000, idealHeight: 950)
        }
    }
}
