import Combine
import Foundation
#if DEBUG
    import OSLog
#endif

let assistantWidth: CGFloat = 140
let assistantHeight: CGFloat = 380
let emptyData = Data([0])

extension String: Error {}
extension CurrentValueSubject: @unchecked Sendable {}

func log(_ message: @autoclosure () -> String) {
    #if DEBUG
        os_log("%{public}@", message())
    #endif
}
