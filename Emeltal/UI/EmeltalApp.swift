import Foundation
import SwiftUI

extension Notification.Name {
    static let AppWillQuit = Self("AppWillQuit")
}

#if canImport(AppKit)
    final class AppDelegate: NSObject, NSApplicationDelegate {
        func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
            NotificationCenter.default.post(name: .AppWillQuit, object: nil)

            Task {
                // create a delay so things can de-init
                try? await Task.sleep(for: .seconds(0.3))
                sender.reply(toApplicationShouldTerminate: true)
            }
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
            }
        }
        .preferredColorScheme(.dark)
        .onReceive(NotificationCenter.default.publisher(for: .AppWillQuit)) { _ in
            switch appPhase {
            case .selection, .shutdown:
                return
            case .conversation:
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
