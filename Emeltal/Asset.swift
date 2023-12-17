import Foundation

enum Asset: String, Identifiable, CaseIterable {
    case dolphinMixtral, deepSeekCoder, mythoMax, whisper, none

    func mlTemplate(in context: LlamaContext) -> Template? {
        switch self {
        case .dolphinMixtral:
            Template(format: .chatml,
                     system: "You are an intelligent and cheerful chatbot.",
                     bosToken: context.bosToken)
        case .deepSeekCoder:
            Template(format: .instruct,
                     system: "You are an intelligent and helpful coding assistant.",
                     bosToken: context.bosToken)
        case .mythoMax:
            Template(format: .instruct,
                     system: "You are an intelligent and helpful writing assistant.",
                     bosToken: context.bosToken)
        case .none, .whisper:
            nil
        }
    }

    var sizeDescription: String {
        switch self {
        case .dolphinMixtral: "32.2 GB"
        case .deepSeekCoder: "35.4 GB"
        case .mythoMax: "10.6 GB"
        case .whisper: "1.1 GB"
        case .none: ""
        }
    }

    var aboutText: String {
        switch self {
        case .deepSeekCoder:
            "This no-nonsense model focuses specifically on code-related generation and questions"
        case .dolphinMixtral:
            "The current state of the art, with multifaceted expertise and good conversational ability."
        case .mythoMax:
            "MythoMax is a model designed to be both imaginative and useful for creativity and writing."
        case .whisper:
            "OpenAI's industry leading speech recognition. Lets you talk directly to the model if you prefer. Ensure you have a good mic and 'voice isolation' is selected from the menubar for best results."
        case .none:
            ""
        }
    }

    var maxBatch: UInt32 {
        switch self {
        case .deepSeekCoder: 1024
        case .dolphinMixtral: 1024
        case .mythoMax: 1024
        case .none, .whisper: 0
        }
    }

    var topK: Int32 {
        switch self {
        case .deepSeekCoder: 49
        case .dolphinMixtral: 49
        case .mythoMax: 49
        case .none, .whisper: 0
        }
    }

    var topP: Float {
        switch self {
        case .deepSeekCoder: 0.14
        case .dolphinMixtral: 0.14
        case .mythoMax: 0.14
        case .none, .whisper: 0
        }
    }

    var temperature: Float {
        switch self {
        case .deepSeekCoder: 1.31
        case .dolphinMixtral: 1.31
        case .mythoMax: 1.31
        case .none, .whisper: 0
        }
    }

    var repeatPenatly: Float {
        switch self {
        case .deepSeekCoder: 1
        case .dolphinMixtral: 1.17
        case .mythoMax: 1.17
        case .none, .whisper: 0
        }
    }

    var frequencyPenalty: Float {
        switch self {
        case .deepSeekCoder: 0
        case .dolphinMixtral: 0.1
        case .mythoMax: 0.1
        case .none, .whisper: 0
        }
    }

    var presentPenalty: Float {
        switch self {
        case .deepSeekCoder: 1
        case .dolphinMixtral: 1.1
        case .mythoMax: 1.1
        case .none, .whisper: 0
        }
    }

    var fetchUrl: URL {
        let uri = switch self {
        case .dolphinMixtral:
            "https://huggingface.co/TheBloke/dolphin-2.5-mixtral-8x7b-GGUF/resolve/main/dolphin-2.5-mixtral-8x7b.Q5_K_M.gguf"
        case .deepSeekCoder:
            "https://huggingface.co/TheBloke/deepseek-coder-33B-instruct-GGUF/resolve/main/deepseek-coder-33b-instruct.Q8_0.gguf"
        case .mythoMax:
            "https://huggingface.co/TheBloke/MythoMax-L2-13B-GGUF/resolve/main/mythomax-l2-13b.Q8_0.gguf"
        case .whisper:
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-q5_0.bin"
        case .none:
            ""
        }
        return URL(string: uri)!
    }

    var localPath: URL {
        let modelDir = appDocumentsUrl.appendingPathComponent("models", conformingTo: .directory)
        let fm = FileManager.default
        if !fm.fileExists(atPath: modelDir.path) {
            try! fm.createDirectory(at: modelDir, withIntermediateDirectories: true)
        }
        return modelDir.appendingPathComponent(fetchUrl.lastPathComponent)
    }

    var displayName: String {
        switch self {
        case .dolphinMixtral: "Dolphin 2.5 (on Mixtral 7b)"
        case .deepSeekCoder: "DeepSeek Coder"
        case .mythoMax: "MythoMax Writing Assistant"
        case .whisper: "Whisper Large v3"
        case .none: ""
        }
    }

    var id: String {
        rawValue
    }
}
