import Foundation

struct Template {
    enum Step {
        case initial, turn(text: String, index: Int), cancel
    }

    enum Format {
        case alpaca, chatml, userAssistant

        var acceptsSystemPrompt: Bool {
            switch self {
            case .userAssistant: false
            case .alpaca, .chatml: true
            }
        }
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
            if format.acceptsSystemPrompt, !system.isEmpty {
                "\(bosToken)\(prefix)\(system)\(suffix)"
            } else {
                "\(bosToken)"
            }
        case let .turn(text, _):
            "\(prefix)\(text)\(suffix)"
        case .cancel:
            suffix
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
            case .cancel:
                ""
            }
        case .alpaca:
            switch step {
            case .initial:
                ""
            case let .turn(_, index):
                if index == 0 {
                    "\n\n### Instruction:\n\n"
                } else {
                    "<s>\n\n### Instruction:\n\n"
                }
            case .cancel:
                ""
            }
        case .userAssistant:
            switch step {
            case .initial:
                "### User:\n"
            case .turn:
                "\n\n### User:\n"
            case .cancel:
                ""
            }
        }
    }

    private func suffix(for step: Step) -> String {
        switch format {
        case .chatml:
            switch step {
            case .initial:
                "<|im_end|>"
            case .turn:
                "<|im_end|>\n<|im_start|>assistant\n"
            case .cancel:
                "<|im_end|>"
            }
        case .alpaca:
            switch step {
            case .initial:
                ""
            case .turn:
                "\n\n### Response:\n\n"
            case .cancel:
                "\n\n"
            }
        case .userAssistant:
            switch step {
            case .initial:
                ""
            case .turn:
                "\n\n### Assistant:\n"
            case .cancel:
                ""
            }
        }
    }
}
