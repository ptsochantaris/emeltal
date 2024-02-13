import Foundation

struct Template {
    enum Step {
        case initial, turn(text: String, index: Int), cancel
    }

    enum Format {
        case alpaca, chatml, userAssistant, openChat, llamaLarge, mistral, miniCpm, vicuna

        var acceptsSystemPrompt: Bool {
            switch self {
            case .miniCpm, .mistral, .openChat, .userAssistant: false
            case .alpaca, .chatml, .llamaLarge, .vicuna: true
            }
        }

        var blockBosToken: Bool {
            false
        }
    }

    init(format: Format, system: String, bosToken: String) {
        self.format = format
        self.system = system
        self.bosToken = format.blockBosToken ? "" : bosToken
    }

    func text(for step: Step) -> String {
        let prefix = prefix(for: step)
        let suffix = suffix(for: step)
        return switch step {
        case .initial:
            if let systemText {
                "\(bosToken)\(prefix)\(systemText)\(suffix)"
            } else {
                "\(bosToken)"
            }
        case let .turn(text, _):
            "\(prefix)\(text)\(suffix)"
        case .cancel:
            suffix
        }
    }

    var systemText: String? {
        if format.acceptsSystemPrompt, !system.isEmpty {
            system
        } else {
            nil
        }
    }

    private let format: Format
    private let system: String
    private let bosToken: String

    private func prefix(for step: Step) -> String {
        switch format {
        case .vicuna:
            switch step {
            case .cancel, .initial: ""
            case .turn: "USER: "
            }

        case .miniCpm:
            switch step {
            case .cancel, .initial: ""
            case .turn: "<用户>"
            }

        case .openChat:
            switch step {
            case .cancel, .initial: ""
            case .turn: "GPT4 Correct User: "
            }

        case .chatml:
            switch step {
            case .initial: "<|im_start|>system\n"
            case .turn: "\n<|im_start|>user\n"
            case .cancel: ""
            }

        case .alpaca:
            switch step {
            case .cancel, .initial: ""
            case let .turn(_, index):
                if index == 0 {
                    "\n\n### Instruction:\n\n"
                } else {
                    "<s>\n\n### Instruction:\n\n"
                }
            }

        case .userAssistant:
            switch step {
            case .initial: "### User:\n"
            case .turn: "\n\n### User:\n"
            case .cancel: ""
            }

        case .llamaLarge:
            switch step {
            case .initial: "Source: system\n\n "
            case .turn: "Source: user\n\n "
            case .cancel: ""
            }

        case .mistral:
            switch step {
            case .initial, .turn: " [INST] "
            case .cancel: ""
            }
        }
    }

    private func suffix(for step: Step) -> String {
        switch format {
        case .vicuna:
            switch step {
            case .cancel: "\n"
            case .initial: "\n\n"
            case .turn: "\nASSISTANT: "
            }

        case .miniCpm:
            switch step {
            case .cancel, .initial: ""
            case .turn: "<AI>"
            }

        case .mistral:
            switch step {
            case .initial, .turn: " [/INST] "
            case .cancel: ""
            }

        case .llamaLarge:
            switch step {
            case .cancel, .initial: " <step> "
            case .turn: " <step> Source: assistant\nDestination: user\n\n "
            }

        case .openChat:
            switch step {
            case .initial: ""
            case .turn: "<|end_of_turn|>GPT4 Correct Assistant: "
            case .cancel: "<|end_of_turn|>"
            }

        case .chatml:
            switch step {
            case .initial: "<|im_end|>"
            case .turn: "<|im_end|>\n<|im_start|>assistant\n"
            case .cancel: "<|im_end|>"
            }

        case .alpaca:
            switch step {
            case .initial: ""
            case .turn: "\n\n### Response:\n\n"
            case .cancel: "\n\n"
            }

        case .userAssistant:
            switch step {
            case .cancel, .initial: ""
            case .turn: "\n\n### Assistant:\n"
            }
        }
    }
}
