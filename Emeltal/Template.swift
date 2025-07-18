import Foundation

struct Template {
    enum Step {
        case initial, turn(text: String, index: Int), cancel
    }

    enum Format {
        case alpaca, chatml, chatmlNoThink, llamaLarge, mistral, mistralNew, vicuna, gemma, llama2, llama3, llama4, glm

        var acceptsSystemPrompt: Bool {
            switch self {
            case .llama2, .llama4, .mistral: false
            case .alpaca, .chatml, .chatmlNoThink, .gemma, .glm, .llama3, .llamaLarge, .mistralNew, .vicuna: true
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
            if let systemText {
                "\(prefix)\(systemText)\(suffix)"
            } else {
                ""
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
            case .initial: "\(bosToken)<start_of_turn>system\n"
            case .cancel: ""
            case .turn: "\(bosToken)<start_of_turn>user\n"
            }

        case .llama3:
            switch step {
            case .initial: "\(bosToken)<|start_header_id|>system<|end_header_id|>\n\n"
            case .turn: "<|start_header_id|>user<|end_header_id|>\n\n"
            case .cancel: ""
            }

        case .glm:
            switch step {
            case .initial: "[gMASK]<sop><|system|>\n"
            case .turn: "<|user|>\n"
            case .cancel: ""
            }

        case .llama4:
            switch step {
            case .initial: "\(bosToken)<|header_start|>user<|header_end|>\n\n"
            case .turn: "<|header_start|>user<|header_end|>\n\n"
            case .cancel: ""
            }

        case .chatml, .chatmlNoThink:
            switch step {
            case .initial: "\(bosToken)<|im_start|>system\n"
            case .turn: "<|im_start|>user\n"
            case .cancel: ""
            }

        case .alpaca:
            switch step {
            case .cancel, .initial: ""
            case .turn:
                "### Instruction:\n"
            }

        case .llamaLarge:
            switch step {
            case .initial: "\(bosToken)Source: system\n\n "
            case .turn: "Source: user\n\n "
            case .cancel: ""
            }

        case .llama2:
            switch step {
            case .initial: bosToken
            case let .turn(_, index): index == 0 ? "\(bosToken)[INST] " : "\n\(bosToken)[INST] "
            case .cancel: "</s>\n"
            }

        case .mistral:
            switch step {
            case .initial: "\(bosToken)[INST] "
            case .turn: " [INST] "
            case .cancel: ""
            }

        case .mistralNew:
            switch step {
            case .initial: "\(bosToken)[SYSTEM_PROMPT]"
            case .turn: " [INST] "
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

        case .mistral:
            switch step {
            case .initial, .turn: " [/INST] "
            case .cancel: ""
            }

        case .mistralNew:
            switch step {
            case .initial: "[/SYSTEM_PROMPT]"
            case .turn: " [/INST] "
            case .cancel: ""
            }

        case .llama2:
            switch step {
            case .cancel: ""
            case .initial: "\n<</SYS>>\n\n"
            case .turn: " [/INST] "
            }

        case .llama3:
            switch step {
            case .cancel, .initial:
                "<|eot_id|>\n"
            case .turn:
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            }

        case .llama4:
            switch step {
            case .cancel, .initial:
                "<|eot|>\n"
            case .turn:
                "<|eot|><|header_start|>assistant<|header_end|>\n\n"
            }

        case .llamaLarge:
            switch step {
            case .cancel, .initial: " <step> "
            case .turn: " <step> Source: assistant\nDestination: user\n\n "
            }

        case .glm:
            switch step {
            case .initial: ""
            case .turn: "<|assistant|>\n"
            case .cancel: "\n"
            }

        case .chatml:
            switch step {
            case .initial: "<|im_end|>\n"
            case .turn: "<|im_end|>\n<|im_start|>assistant\n"
            case .cancel: "<|im_end|>\n"
            }

        case .chatmlNoThink:
            switch step {
            case .initial: "<|im_end|>\n"
            case .turn: "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            case .cancel: "<|im_end|>\n"
            }

        case .alpaca:
            switch step {
            case .initial: ""
            case .turn: "\n\n### Response:\n"
            case .cancel: "\n\n"
            }
        }
    }

    var failsafeStop: String? {
        switch format {
        case .gemma, .glm, .llama2, .llama3, .llama4, .llamaLarge, .mistral, .mistralNew, .vicuna:
            nil

        case .chatml, .chatmlNoThink:
            "<|im_start|>user\n"

        case .alpaca:
            "### Instruction:\n\n"
        }
    }
}
