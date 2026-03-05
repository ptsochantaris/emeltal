import Foundation
import SwiftUI

extension Notification.Name {
    static let AppWillQuit = Self("AppWillQuit")
}

#if canImport(AppKit)
    final class AppDelegate: NSObject, NSApplicationDelegate {
        func applicationShouldTerminate(_: NSApplication) -> NSApplication.TerminateReply {
            NotificationCenter.default.post(name: .AppWillQuit, object: nil)
            return .terminateLater
        }
    }
#endif

struct AppStack: View {
    enum Phase {
        case selection, conversation(ConversationState), shutdown
    }

    #if canImport(AppKit)
        @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    #endif

    @State private var appPhase = Phase.selection

    var body: some View {
        NavigationStack {
            switch appPhase {
            case .selection:
                ModelPicker(appPhase: $appPhase)

            case let .conversation(state):
                ConversationView(appPhase: $appPhase, state: state)

            case .shutdown:
                Text("Bye")
                    .task {
                        // create a delay so things can de-init
                        try? await Task.sleep(for: .seconds(0.4))
                        NSApplication.shared.reply(toApplicationShouldTerminate: true)
                    }
            }
        }
        .preferredColorScheme(.dark)
        .onReceive(NotificationCenter.default.publisher(for: .AppWillQuit)) { _ in
            if case .shutdown = appPhase {
                // All good
            } else {
                appPhase = .shutdown
            }
        }
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
