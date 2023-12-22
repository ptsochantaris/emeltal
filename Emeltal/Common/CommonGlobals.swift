import Foundation
import OSLog
#if canImport(AppKit)
    import AppKit
#endif

let assistantWidth: CGFloat = 140
let assistantHeight: CGFloat = 380

extension String: Error {}

#if canImport(AppKit)
    extension NSSound: @unchecked Sendable {}
#endif

func log(_ message: @autoclosure () -> String) {
    #if DEBUG
        os_log("%{public}@", message())
    #endif
}
