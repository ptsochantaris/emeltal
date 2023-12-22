import Foundation

enum MicState: Equatable {
    static func == (lhs: Self, rhs: Self) -> Bool {
        switch lhs {
        case .quiet:
            if case .quiet = rhs {
                return true
            }
        case .listening:
            if case .listening = rhs {
                return true
            }
        }
        return false
    }

    case quiet(prefixBuffer: [Float]), listening(quietPeriods: Int)

    var isQuiet: Bool {
        if case .quiet = self {
            return true
        }
        return false
    }
}
