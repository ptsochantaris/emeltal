import Foundation
import Metal

enum Asset: String, Identifiable, CaseIterable {
    case dolphinMixtral, deepSeekCoder, mythoMax, solar, whisper, none

    func mlTemplate(in context: LlamaContext) -> Template? {
        switch self {
        case .solar:
            Template(format: .instruct,
                     system: "You are an intelligent and cheerful chatbot.",
                     bosToken: context.bosToken)

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

    var useGpuOnThisSystem: Bool {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError( "Failed to get the system's default Metal device." )
        }
        let vramSize = device.recommendedMaxWorkingSetSize / 1_000_000_000
        log("Checking if current model selection can run on GPU (\(vramSize) GB)")
        return switch vramSize {
        case 0 ..< 22:
            false
        case 22 ..< 51:
            switch self {
            case .deepSeekCoder: false
            case .dolphinMixtral: false
            case .mythoMax, .solar: true
            case .whisper, .none: false
            }
        default:
            true
        }
    }

    var sizeDescription: String {
        switch self {
        case .dolphinMixtral: "32.2 GB"
        case .deepSeekCoder: "35.4 GB"
        case .solar: "7.6 GB"
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
        case .solar:
            "One of the highest performing models for chat. A great starting point."
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
        case .solar: 1024
        case .none, .whisper: 0
        }
    }

    var topK: Int32 {
        switch self {
        case .deepSeekCoder: 49
        case .dolphinMixtral: 49
        case .mythoMax: 49
        case .solar: 49
        case .none, .whisper: 0
        }
    }

    var topP: Float {
        switch self {
        case .deepSeekCoder: 0.14
        case .dolphinMixtral: 0.14
        case .mythoMax: 0.14
        case .solar: 0.14
        case .none, .whisper: 0
        }
    }

    var temperature: Float {
        switch self {
        case .deepSeekCoder: 0.4
        case .dolphinMixtral: 1.31
        case .mythoMax: 1.31
        case .solar: 1.31
        case .none, .whisper: 0
        }
    }

    var repeatPenatly: Float {
        switch self {
        case .deepSeekCoder: 1
        case .dolphinMixtral: 1.17
        case .mythoMax: 1.17
        case .solar: 1.17
        case .none, .whisper: 0
        }
    }

    var frequencyPenalty: Float {
        switch self {
        case .deepSeekCoder: 0
        case .dolphinMixtral: 0.1
        case .mythoMax: 0.1
        case .solar: 0.1
        case .none, .whisper: 0
        }
    }

    var presentPenalty: Float {
        switch self {
        case .deepSeekCoder: 1
        case .dolphinMixtral: 1.1
        case .mythoMax: 1.1
        case .solar: 1.1
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
        case .solar:
            "https://huggingface.co/jan-hq/Solar-10.7B-SLERP-GGUF/resolve/main/solar-10.7b-slerp.Q5_K_M.gguf"
        case .none:
            ""
        }
        return URL(string: uri)!
    }

    var localModelPath: URL {
        let modelDir = appDocumentsUrl.appendingPathComponent("models", conformingTo: .directory)
        let fm = FileManager.default
        if !fm.fileExists(atPath: modelDir.path) {
            try! fm.createDirectory(at: modelDir, withIntermediateDirectories: true)
        }
        return modelDir.appendingPathComponent(fetchUrl.lastPathComponent)
    }

    var localStatePath: URL {
        let fm = FileManager.default
        let statePath = appDocumentsUrl.appendingPathComponent("states-\(id)", conformingTo: .directory)
        if !fm.fileExists(atPath: statePath.path) {
            try? fm.createDirectory(at: statePath, withIntermediateDirectories: true)
        }
        return statePath
    }

    var isInstalled: Bool {
        FileManager.default.fileExists(atPath: localModelPath.path)
    }

    func unInstall() {
        try? FileManager.default.removeItem(at: localModelPath)
        try? FileManager.default.removeItem(at: localStatePath)
    }

    var displayName: String {
        switch self {
        case .dolphinMixtral: "Dolphin 2.5 (on Mixtral 7b)"
        case .deepSeekCoder: "DeepSeek Coder"
        case .mythoMax: "MythoMax Writing Assistant"
        case .whisper: "Whisper Large v3"
        case .solar: "Pandora Solar"
        case .none: ""
        }
    }

    var id: String {
        rawValue
    }
}
