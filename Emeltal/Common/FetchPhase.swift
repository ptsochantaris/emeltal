import Foundation

enum FetchPhase {
    case boot, fetching(downloaded: Int64, expected: Int64), error(error: Error), done, cancelled

    var isOngoing: Bool {
        switch self {
        case .boot, .fetching:
            true
        case .cancelled, .done, .error:
            false
        }
    }

    var shouldShowToUser: Bool {
        switch self {
        case .boot, .cancelled, .error, .fetching:
            true
        case .done:
            false
        }
    }
}
