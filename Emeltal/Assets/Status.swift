import Foundation

extension Model {
    @MainActor
    enum Status: Sendable, Equatable {
        case checking, available, recommended, installed(AssetFetcher), notReady, installing(AssetFetcher)

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
            case let .installing(fetcher):
                let percent = fetcher.progressPercentage
                return ("INSTALLING: \(Int(percent * 100))%", percent)
            }
        }
    }
}
