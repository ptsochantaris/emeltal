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
    case startup, booting, warmup, loading(managers: [AssetFetcher]), waiting, listening(state: Mic.State), transcribing, processingPrompt, transcribingDone, replying, shutdown

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
        case .transcribing:
            if case .transcribing = rhs {
                return true
            }
        case .replying:
            if case .replying = rhs {
                return true
            }
        case .processingPrompt:
            if case .processingPrompt = rhs {
                return true
            }
        case .transcribingDone:
            if case .transcribingDone = rhs {
                return true
            }
        case .waiting:
            if case .waiting = rhs {
                return true
            }
        case .shutdown:
            if case .shutdown = rhs {
                return true
            }
        }
        return false
    }

    var nominal: Bool {
        switch self {
        case .listening, .replying, .waiting:
            true
        case .booting, .loading, .processingPrompt, .shutdown, .startup, .transcribing, .transcribingDone, .warmup:
            false
        }
    }

    init?(data: Data) {
        let primary: UInt8 = data[0]
        let secondary: UInt8 = data[1]
        switch primary {
        case 1:
            self = .booting
        case 2:
            switch secondary {
            case 0:
                self = .listening(state: .talking(voiceDetected: false, quietCount: 0))
            case 1:
                self = .listening(state: .talking(voiceDetected: true, quietCount: 0))
            case 2:
                self = .listening(state: .quiet(prefixBuffer: []))
            default:
                return nil
            }
        case 3:
            self = .loading(managers: [])
        case 4:
            self = .transcribing
        case 5:
            self = .replying
        case 6:
            self = .startup
        case 7:
            self = .processingPrompt
        case 8:
            self = .waiting
        case 9:
            self = .warmup
        case 10:
            self = .shutdown
        case 11:
            self = .transcribingDone
        default:
            return nil
        }
    }

    var iconImageName: String {
        switch self {
        case .booting, .loading, .shutdown, .startup, .warmup: "hourglass.circle"
        case .transcribing, .waiting: "circle"
        case .listening: "waveform.circle"
        case .processingPrompt, .transcribingDone: "ellipsis.circle"
        case .replying: "waveform.circle.fill"
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
            case let .talking(voiceDetected, _):
                data[1] = voiceDetected ? 1 : 0
            case .quiet:
                data[1] = 2
            }
        case .loading:
            data[0] = 3
        case .transcribing:
            data[0] = 4
        case .replying:
            data[0] = 5
        case .startup:
            data[0] = 6
        case .processingPrompt:
            data[0] = 7
        case .waiting:
            data[0] = 8
        case .warmup:
            data[0] = 9
        case .shutdown:
            data[0] = 10
        case .transcribingDone:
            data[0] = 11
        }
        return data
    }

    func audioFeedback(using speaker: Speaker) {
        switch self {
        case .listening:
            Task {
                await speaker.play(effect: .startListening)
            }
        case .transcribing:
            Task {
                await speaker.play(effect: .endListening)
            }
        case .booting, .loading, .processingPrompt, .replying, .shutdown, .startup, .transcribingDone, .waiting, .warmup:
            break
        }
    }

    var showGenie: Bool {
        switch self {
        case .processingPrompt, .replying, .transcribing, .transcribingDone:
            true
        case .booting, .listening, .loading, .shutdown, .startup, .waiting, .warmup:
            false
        }
    }

    var showAlwaysOn: Bool {
        switch self {
        case .booting, .loading, .processingPrompt, .replying, .shutdown, .startup, .transcribing, .transcribingDone, .warmup:
            false
        case .listening, .waiting:
            Mic.havePermission
        }
    }

    var pushButtonActive: Bool {
        switch self {
        case .listening, .replying, .waiting:
            Mic.havePermission
        case .booting, .loading, .processingPrompt, .shutdown, .startup, .transcribing, .transcribingDone, .warmup:
            false
        }
    }
}
