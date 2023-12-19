import Carbon.HIToolbox.Events
import Foundation
import SwiftUI

struct PushButton: NSViewRepresentable {
    private let view: MouseTracker

    init(handler: @escaping (Bool) -> Void) {
        view = MouseTracker(handler: handler)
    }

    func makeNSView(context _: Context) -> MouseTracker {
        view
    }

    func updateNSView(_: MouseTracker, context _: Context) {}

    final class MouseTracker: NSView {
        var handler: (Bool) -> Void

        private var pushed = false

        init(handler: @escaping (Bool) -> Void, pressed _: Bool = false) {
            self.handler = handler
            super.init(frame: .zero)
            NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .keyUp]) { [weak self] event in
                guard event.keyCode == kVK_DownArrow, let self else { return event }
                if event.isARepeat { return nil }
                switch event.type {
                case .keyUp:
                    self.handler(false)
                    return nil
                case .keyDown:
                    self.handler(true)
                    return nil
                default:
                    return event
                }
            }
        }

        @available(*, unavailable)
        required init?(coder _: NSCoder) {
            fatalError()
        }

        override func mouseDown(with event: NSEvent) {
            super.mouseDown(with: event)
            handler(true)

            while true {
                guard let theEvent = window?.nextEvent(matching: [.leftMouseUp]) else {
                    continue
                }
                switch theEvent.type {
                case .leftMouseUp:
                    handler(false)
                    return
                default:
                    break
                }
            }
        }
    }
}
