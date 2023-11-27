import Foundation

struct Template {
    enum Step {
        case initial, turn(text: String)
    }

    enum Format {
        case instruct, chatml, userAssistant
    }

    init(format: Format, system: String, bosToken: String) {
        self.format = format
        self.system = system
        self.bosToken = bosToken
    }

    func text(for step: Step) -> String {
        let prefix = prefix(for: step)
        let suffix = suffix(for: step)
        return switch step {
        case .initial:
            if bosToken.isEmpty {
                if system.isEmpty {
                    ""
                } else {
                    "\(prefix)\(system)\(suffix)"
                }
            } else if system.isEmpty {
                "\(bosToken)"
            } else {
                "\(bosToken)\(prefix)\(system)\(suffix)"
            }
        case let .turn(text):
            "\(prefix)\(text)\(suffix)"
        }
    }

    private let format: Format
    private let system: String
    private let bosToken: String

    private func prefix(for step: Step) -> String {
        switch format {
        case .chatml:
            switch step {
            case .initial:
                "<|im_start|>system\n"
            case .turn:
                "\n<|im_start|>user\n"
            }
        case .instruct:
            switch step {
            case .initial:
                ""
            case .turn:
                "\n\n### Instruction:\n\n"
            }
        case .userAssistant:
            switch step {
            case .initial:
                "### System:\n"
            case .turn:
                "### User:\n"
            }
        }
    }

    private func suffix(for step: Step) -> String {
        switch format {
        case .userAssistant:
            switch step {
            case .initial:
                "\n\n"
            case .turn:
                "\n\n### Assistant:\n"
            }
        case .chatml:
            switch step {
            case .initial:
                ""
            case .turn:
                "<|im_end|>\n<|im_start|>assistant\n"
            }
        case .instruct:
            switch step {
            case .initial:
                ""
            case .turn:
                "\n\n### Response:\n\n"
            }
        }
    }
}
