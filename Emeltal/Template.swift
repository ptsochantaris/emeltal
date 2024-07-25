import Foundation

struct Template {
    enum Step {
        case initial, turn(text: String, index: Int), cancel
    }

    enum Format {
        case alpaca, chatml, userAssistant, openChat, llamaLarge, mistral, miniCpm, vicuna, gemma, llama3

        var acceptsSystemPrompt: Bool {
            switch self {
            case .miniCpm, .mistral, .openChat, .userAssistant: false
            case .alpaca, .chatml, .gemma, .llama3, .llamaLarge, .vicuna: true
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

        case .gemma:
            switch step {
            case .initial: "<start_of_turn>system\n"
            case .cancel: ""
            case .turn: "<start_of_turn>user\n"
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

        case .llama3:
            switch step {
            case .initial: "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            case .turn: "<|start_header_id|>user<|end_header_id|>\n\n"
            case .cancel: ""
            }

        case .chatml:
            switch step {
            case .initial: "<|im_start|>system\n"
            case .turn: "<|im_start|>user\n"
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

        case .gemma:
            switch step {
            case .cancel, .initial: "<end_of_turn>\n"
            case .turn: "<end_of_turn>\n<start_of_turn>model"
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

        case .llama3:
            switch step {
            case .cancel, .initial:
                "<|eot_id|>\n"
            case .turn:
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
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
            case .initial: "<|im_end|>\n"
            case .turn: "<|im_end|>\n<|im_start|>assistant\n"
            case .cancel: "<|im_end|>\n"
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

    var failsafeStop: String? {
        switch format {
        case .gemma, .llama3, .llamaLarge, .miniCpm, .mistral, .openChat, .userAssistant, .vicuna:
            nil

        case .chatml:
            "<|im_start|>user\n"

        case .alpaca:
            "### Instruction:\n\n"
        }
    }
}
