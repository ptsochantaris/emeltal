import Foundation

enum ActivationState {
    case button, voiceActivated

    init?(data: Data) {
        let primary: UInt8 = data[0]
        switch primary {
        case 1:
            self = .button
        case 2:
            self = .voiceActivated
        default:
            return nil
        }
    }

    var data: Data {
        var data = Data(repeating: 0, count: 1)

        switch self {
        case .button:
            data[0] = 1
        case .voiceActivated:
            data[0] = 2
        }
        return data
    }

    var isManual: Bool {
        switch self {
        case .button:
            true
        case .voiceActivated:
            false
        }
    }
}

enum AppMode: Equatable {
    static func == (lhs: Self, rhs: Self) -> Bool {
        switch lhs {
        case .startup:
            if case .startup = rhs {
                return true
            }
        case .booting:
            if case .booting = rhs {
                return true
            }
        case .warmup:
            if case .warmup = rhs {
                return true
            }
        case .listening:
            if case .listening = rhs {
                return true
            }
        case .loading:
            if case .loading = rhs {
                return true
            }
        case .noting:
            if case .noting = rhs {
                return true
            }
        case .replying:
            if case .replying = rhs {
                return true
            }
        case .thinking:
            if case .thinking = rhs {
                return true
            }
        case .waiting:
            if case .waiting = rhs {
                return true
            }
        }
        return false
    }

    case startup, booting, warmup, loading(managers: [AssetManager]), waiting, listening(state: Mic.State), noting, thinking, replying

    init?(data: Data) {
        let primary: UInt8 = data[0]
        let secondary: UInt8 = data[1]
        switch primary {
        case 1:
            self = .booting
        case 2:
            switch secondary {
            case 1:
                self = .listening(state: .talking(quietPeriods: 0))
            case 2:
                self = .listening(state: .quiet(prefixBuffer: []))
            default:
                return nil
            }
        case 3:
            self = .loading(managers: [])
        case 4:
            self = .noting
        case 5:
            self = .replying
        case 6:
            self = .startup
        case 7:
            self = .thinking
        case 8:
            self = .waiting
        case 9:
            self = .warmup
        default:
            return nil
        }
    }

    var data: Data {
        var data = Data(repeating: 0, count: 2)

        switch self {
        case .booting:
            data[0] = 1
        case let .listening(state):
            data[0] = 2
            switch state {
            case .talking:
                data[1] = 1
            case .quiet:
                data[1] = 2
            }
        case .loading:
            data[0] = 3
        case .noting:
            data[0] = 4
        case .replying:
            data[0] = 5
        case .startup:
            data[0] = 6
        case .thinking:
            data[0] = 7
        case .waiting:
            data[0] = 8
        case .warmup:
            data[0] = 9
        }
        return data
    }

    func audioFeedback(using speaker: Speaker) {
        switch self {
        case .listening:
            Task {
                await speaker.playEffect(.startListening)
            }
        case .noting:
            Task {
                await speaker.playEffect(.endListening)
            }
        case .booting, .loading, .replying, .startup, .thinking, .waiting, .warmup:
            break
        }
    }

    var showGenie: Bool {
        switch self {
        case .noting, .replying, .thinking:
            true
        case .booting, .listening, .loading, .startup, .waiting, .warmup:
            false
        }
    }

    var showAlwaysOn: Bool {
        switch self {
        case .booting, .loading, .noting, .replying, .startup, .thinking, .warmup:
            false
        case .listening, .waiting:
            canUseMic
        }
    }

    var pushButtonActive: Bool {
        switch self {
        case .listening, .replying, .waiting:
            canUseMic
        case .booting, .loading, .noting, .startup, .thinking, .warmup:
            false
        }
    }
}
