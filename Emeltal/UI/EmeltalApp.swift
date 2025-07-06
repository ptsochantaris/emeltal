import Foundation
import SwiftUI

struct AppStack: View {
    enum Phase {
        case selection, conversation(ConversationState)
    }

    @State private var appPhase = Phase.selection

    var body: some View {
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
}

@main
struct EmeltalApp: App {
    var body: some Scene {
        #if canImport(AppKit)
            Window("Emeltal", id: "Emeltal") {
                AppStack()
            }
            .defaultSize(width: 1000, height: 1000)
        #else
            WindowGroup("Emeltal") {
                AppStack()
            }
            .defaultSize(width: 1000, height: 1000)
        #endif
    }
}
