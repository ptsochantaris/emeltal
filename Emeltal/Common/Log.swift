import Foundation
import OSLog
#if canImport(AppKit)
    import AppKit
#endif

extension String: Error {}

#if canImport(AppKit)
    extension NSSound: @unchecked Sendable {}
#endif

func log(_ message: @autoclosure () -> String) {
    #if DEBUG
        os_log("%{public}@", message())
    #endif
}
