import Foundation

extension Model {
    @MainActor
    enum Status: Equatable {
        case checking, available(size: String?), recommended(size: String?), installed(AssetFetcher, size: String?), notReady, installing(AssetFetcher, size: String?)

        var sizeDescription: String? {
            switch self {
            case let .available(size), let .installed(_, size), let .installing(_, size), let .recommended(size):
                size
            case .notReady:
                nil
            case .checking:
                "…"
            }
        }

        var badgeInfo: (label: String, progress: CGFloat)? {
            switch self {
            case .available, .checking:
                return nil
            case .recommended:
                return ("START HERE", 0)
            case .installed:
                return ("INSTALLED", 0)
            case .notReady:
                return ("NOT AVAILABLE", 0)
            case let .installing(fetcher, _):
                let percent = fetcher.progressPercentage
                return ("INSTALLING: \(Int(percent * 100))%", percent)
            }
        }
    }
}
