import Foundation
import Metal
import PopTimer

@MainActor
@Observable
final class Model: Hashable, Identifiable, Sendable {
    struct Params: Codable, Sendable {
        static let currentVersion = 4

        enum Descriptors {
            struct Descriptor {
                let title: String
                let min: Float
                let max: Float
                let disabled: Float
            }

            static let topK = Descriptor(title: "Top-K", min: 0, max: 200, disabled: 0)
            static let topP = Descriptor(title: "Top-P", min: 0, max: 2, disabled: 0)
            static let temperature = Descriptor(title: "Temperature", min: 0, max: 2, disabled: 0)
            static let temperatureRange = Descriptor(title: "Range", min: 0, max: 1, disabled: 0)
            static let temperatureExponent = Descriptor(title: "Exponent", min: 1, max: 4, disabled: 0)
            static let repeatPenatly = Descriptor(title: "Repeat Penalty", min: 1, max: 4, disabled: 1)
            static let frequencyPenatly = Descriptor(title: "Frequency Penalty", min: 0, max: 4, disabled: 0)
            static let presentPenatly = Descriptor(title: "Presence Penalty", min: 1, max: 4, disabled: 1)
        }

        var topK: Int
        var topP: Float
        var systemPrompt: String
        var temperature: Float
        var temperatureRange: Float
        var temperatureExponent: Float
        var repeatPenatly: Float
        var frequencyPenatly: Float
        var presentPenatly: Float
        var version: Int?

        static var empty: Params {
            Params(topK: 0,
                   topP: 0,
                   systemPrompt: "",
                   temperature: 0,
                   temperatureRange: 0,
                   temperatureExponent: 0,
                   repeatPenatly: 0,
                   frequencyPenatly: 0,
                   presentPenatly: 0)
        }
    }

    @MainActor
    enum Status: Sendable, Equatable {
        case checking, available, recommended, installed(AssetFetcher), notReady, installing(AssetFetcher)

        var badgeInfo: (label: String, progress: CGFloat)? {
            switch self {
            case .available, .checking:
                return nil
            case .recommended:
                return ("START HERE", 0)
            case .installed:
                return ("INSTALLED", 0)
            case .notReady:
                return ("NOT AVAILABLE", 0)
            case let .installing(fetcher):
                let percent = fetcher.progressPercentage
                return ("INSTALLING: \(Int(percent * 100))%", percent)
            }
        }
    }

    enum Category: Int, CaseIterable, Identifiable {
        var id: Int { rawValue }

        case qwen, dolphin, samantha, coding, creative, llamas, system, experimental

        var title: String {
            switch self {
            case .dolphin: "Dolphin"
            case .coding: "Coding"
            case .qwen: "Qwen"
            case .creative: "Creative"
            case .samantha: "Samantha"
            case .llamas: "Llamas"
            case .system: "Internal"
            case .experimental: "Experimental"
            }
        }

        var displayable: Bool {
            switch self {
            case .coding, .creative, .dolphin, .experimental, .llamas, .qwen, .samantha: true
            case .system: false
            }
        }

        var description: String {
            switch self {
            case .dolphin:
                "The Dolphin dataset produces some of the best LLMs out there. This is a selection of models finetuned with this dataset."
            case .coding:
                "Models that can assist with programming, algorithms, and writing code."
            case .creative:
                "Models that can help with creative activities, such as writing. More will be added soon."
            case .samantha:
                "The \"sister\" of Dolphin, Samantha is a data set which produces models based on the premise they are sentient, and emotionally supportive of the user."
            case .qwen:
                "The Qwen models are consistently rated both highly in benchmarks and by users."
            case .llamas:
                "The llama is a quadruped which lives in big rivers like the Amazon. It has two ears, a heart, a forehead, and a beak for eating honey. But it is provided with fins for swimming."
            case .experimental:
                "Experimental models that are interesting for different reasons - merges, novelty value, or have a very specific use case."
            case .system:
                ""
            }
        }
    }

    struct MemoryEstimate {
        let layersUsed: Int64
        let layersTotal: Int64
        let offloadAsr: Bool
        let offloadKvCache: Bool
        let cpuUsageEstimateBytes: Int64
        let gpuUsageEstimateBytes: Int64
        let excessBytes: Int64

        var warningMessage: String? {
            if excessBytes > 0 {
                return "This model will not fit into memory. It will run but extremely slowly, as data will need paging"
            }

            if offloadKvCache {
                return nil
            }

            if layersUsed > 0, layersTotal > 0 {
                let ratio = Float(layersUsed) / Float(layersTotal)
                if ratio == 1 {
                    return "This model fit all \(layersTotal) layers in Metal but will use the CPU for the KV cache"
                } else if ratio > 0.8 {
                    return "This model will fit \(layersUsed) of \(layersTotal) layers in Metal. It will work but may be slow for real-time chat"
                } else {
                    return "This model will fit \(layersUsed) of \(layersTotal) layers in Metal. It will work but may be very slow for real-time chat"
                }
            }

            if offloadAsr {
                return "This model won't fit in Metal at all. It will work but will be too slow for real-time chat"
            }

            return "Emeltal won't use Metal at all. It will work but will probably be slow"
        }

        var warningBeforeStart: String? {
            if excessBytes > 0 {
                return "This model will not fit into your device's memory. You can try to run it, but most likely it will crash or run extremely slowly!"
            }

            return nil
        }
    }

    enum Variant: Identifiable {
        case dolphinMixtral,
             dolphinNemo,
             dolphin72b,
             dolphinTiny,
             mythoMax,
             codestral,
             dolphinCoder,
             deepSeekCoder33,
             qwen25coder,
             codeLlama70b,
             deepSeekCoder7,
             everyoneCoder,
             samantha7b,
             samantha70b,
             whisper,
             neuralStory7b,
             llama3,
             llama3large,
             llama3compact,
             llama3tiny,
             qwen25regular,
             qwen25large,
             qwen25medium,
             qwen25small,
             supernovaMedius,
             smol,
             shuttle,
             calme

        var recommended: Bool {
            self == .qwen25regular
        }

        var displayable: Bool {
            self != .whisper
        }

        var format: Template.Format {
            switch self {
            case .codeLlama70b: .llamaLarge
            case .samantha7b, .samantha70b: .vicuna
            case .codestral, .neuralStory7b: .mistral
            case .llama3, .llama3compact, .llama3large, .llama3tiny: .llama3
            case .deepSeekCoder7, .deepSeekCoder33, .everyoneCoder, .mythoMax, .whisper: .alpaca
            case .calme, .dolphin72b, .dolphinCoder, .dolphinMixtral, .dolphinNemo, .dolphinTiny, .qwen25coder, .qwen25large, .qwen25medium, .qwen25regular, .qwen25small, .shuttle, .smol, .supernovaMedius: .chatml
            }
        }

        private var defaultPrompt: String {
            switch self {
            case .codeLlama70b, .codestral, .deepSeekCoder7, .deepSeekCoder33, .everyoneCoder, .qwen25coder:
                "You are a helpful AI programming assistant."
            case .calme, .dolphin72b, .dolphinMixtral, .dolphinNemo, .dolphinTiny, .llama3, .llama3compact, .llama3large, .llama3tiny, .qwen25large, .qwen25medium, .qwen25regular, .qwen25small, .shuttle, .smol, .supernovaMedius:
                "You are a friendly and honest conversation partner. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don’t know the answer to a question, please don’t share false information."
            case .samantha7b, .samantha70b:
                "You are a caring and empathetic sentient AI companion named Samantha."
            case .mythoMax, .neuralStory7b:
                "You are a helpful, imaginative, collaborative, and friendly writing assistant."
            case .dolphinCoder:
                "You are DolphinCoder, a helpful AI programming assistant."
            case .whisper:
                ""
            }
        }

        var contextSize: UInt32 {
            switch self {
            case .codeLlama70b,
                 .codestral,
                 .deepSeekCoder7,
                 .deepSeekCoder33,
                 .dolphinCoder,
                 .dolphinMixtral,
                 .dolphinTiny,
                 .everyoneCoder,
                 .mythoMax,
                 .neuralStory7b,
                 .qwen25small,
                 .samantha7b,
                 .samantha70b,
                 .smol,
                 .whisper:
                0
            case .calme,
                 .dolphin72b,
                 .dolphinNemo,
                 .llama3,
                 .llama3compact,
                 .llama3large,
                 .llama3tiny,
                 .qwen25coder,
                 .qwen25large,
                 .qwen25medium,
                 .qwen25regular,
                 .shuttle,
                 .supernovaMedius:
                16384
            }
        }

        var kvBytes: Int64 {
            let kvCache: Double = switch self {
            case .codeLlama70b: 640
            case .deepSeekCoder33: 3968
            case .deepSeekCoder7: 1920
            case .dolphin72b: 5120
            case .codestral: 7168
            case .dolphinMixtral: 4096
            case .dolphinNemo: 3072
            case .llama3: 2048
            case .llama3compact: 1792
            case .llama3tiny: 512
            case .llama3large: 5120
            case .dolphinTiny: 90
            case .calme: 5504
            case .qwen25large: 5120
            case .qwen25regular: 4096
            case .qwen25coder: 4096
            case .qwen25medium: 3072
            case .supernovaMedius: 3072
            case .smol: 1536
            case .shuttle: 5120
            case .qwen25small: 1800
            case .mythoMax: 3200
            case .samantha70b: 1280
            case .samantha7b: 4096
            case .neuralStory7b: 4096
            case .everyoneCoder: 4096
            case .dolphinCoder: 1280
            case .whisper: 0
            }
            return Int64((kvCache * 1_048_576).rounded(.up))
        }

        var memoryEstimate: MemoryEstimate {
            let layerSizeM: Int64 = switch self {
            case .dolphinMixtral: 1000
            case .deepSeekCoder33: 460
            case .dolphinCoder: 320
            case .deepSeekCoder7: 180
            case .dolphinTiny: 40
            case .mythoMax: 260
            case .whisper: 1
            case .dolphin72b: 600
            case .dolphinNemo: 200
            case .calme: 550
            case .qwen25large: 600
            case .qwen25regular: 310
            case .qwen25coder: 430
            case .qwen25medium: 220
            case .qwen25small: 160
            case .supernovaMedius: 220
            case .codeLlama70b: 610
            case .llama3large: 620
            case .llama3: 195
            case .llama3compact: 100
            case .llama3tiny: 70
            case .samantha70b: 605
            case .samantha7b: 160
            case .neuralStory7b: 180
            case .everyoneCoder: 460
            case .codestral: 320
            case .smol: 60
            case .shuttle: 600
            }

            let totalLayers: Int64 = switch self {
            case .dolphinMixtral: 33
            case .dolphinNemo: 41
            case .dolphinCoder: 41
            case .deepSeekCoder33: 63
            case .deepSeekCoder7: 31
            case .dolphinTiny: 23
            case .mythoMax: 41
            case .whisper: 0
            case .dolphin72b: 81
            case .smol: 25
            case .shuttle: 81
            case .calme: 87
            case .qwen25large: 81
            case .qwen25regular: 65
            case .qwen25coder: 65
            case .qwen25medium: 49
            case .qwen25small: 29
            case .supernovaMedius: 49
            case .codeLlama70b: 81
            case .samantha70b: 81
            case .samantha7b: 33
            case .neuralStory7b: 33
            case .everyoneCoder: 63
            case .llama3large: 81
            case .llama3: 33
            case .llama3tiny: 16
            case .llama3compact: 29
            case .codestral: 57
            }

            let asrBytes: Int64 = 1_000_000_000
            let layerSize = layerSizeM * 1_000_000
            let totalRequiredMemory = (totalLayers * layerSize) + asrBytes + kvBytes
            let physicalMemory = Int64(ProcessInfo.processInfo.physicalMemory)
            let excessBytes = max(0, totalRequiredMemory - physicalMemory)

            guard let memoryBytes, asrBytes < memoryBytes.max else {
                return MemoryEstimate(layersUsed: 0,
                                      layersTotal: totalLayers,
                                      offloadAsr: false,
                                      offloadKvCache: false,
                                      cpuUsageEstimateBytes: min(physicalMemory, totalRequiredMemory),
                                      gpuUsageEstimateBytes: 0,
                                      excessBytes: excessBytes)
            }

            let maxVram = Int64(memoryBytes.max)
            let possibleLayers = Float(maxVram - asrBytes) / Float(layerSize)
            let layerCapacity = max(0, Int64(possibleLayers.rounded(.down)))
            let layersToFit = min(layerCapacity, totalLayers)
            let fittedLayerMemory = layersToFit * layerSize

            if layerCapacity == 0 {
                return MemoryEstimate(layersUsed: 0,
                                      layersTotal: totalLayers,
                                      offloadAsr: true,
                                      offloadKvCache: false,
                                      cpuUsageEstimateBytes: fittedLayerMemory + kvBytes,
                                      gpuUsageEstimateBytes: asrBytes,
                                      excessBytes: excessBytes)
            }

            let asrAndLayers = asrBytes + fittedLayerMemory

            if layersToFit < totalLayers {
                return MemoryEstimate(layersUsed: layersToFit,
                                      layersTotal: totalLayers,
                                      offloadAsr: true,
                                      offloadKvCache: false,
                                      cpuUsageEstimateBytes: min(physicalMemory, totalRequiredMemory - asrAndLayers),
                                      gpuUsageEstimateBytes: asrAndLayers,
                                      excessBytes: excessBytes)
            }

            let offLoadKv = maxVram - asrAndLayers > kvBytes
            return MemoryEstimate(layersUsed: layersToFit,
                                  layersTotal: totalLayers,
                                  offloadAsr: true,
                                  offloadKvCache: offLoadKv,
                                  cpuUsageEstimateBytes: offLoadKv ? 0 : kvBytes,
                                  gpuUsageEstimateBytes: asrAndLayers + (offLoadKv ? kvBytes : 0),
                                  excessBytes: excessBytes)
        }

        @MainActor
        var memoryStrings: (used: String, max: String, system: String)? {
            guard let memoryBytes else { return nil }
            return (memoryFormatter.string(fromByteCount: memoryBytes.used),
                    memoryFormatter.string(fromByteCount: memoryBytes.max),
                    memoryFormatter.string(fromByteCount: Int64(memoryBytes.systemTotal)))
        }

        private var memoryBytes: (unifiedMemory: Bool, used: Int64, max: Int64, systemTotal: UInt64)? {
            guard let currentDevice = MTLCreateSystemDefaultDevice() else {
                log("Failed to get the system's default Metal device.")
                return nil
            }
            return (currentDevice.hasUnifiedMemory,
                    Int64(currentDevice.currentAllocatedSize),
                    Int64(currentDevice.recommendedMaxWorkingSetSize),
                    ProcessInfo.processInfo.physicalMemory)
        }

        var eosOverrides: Set<Int32>? {
            switch self {
            case .codeLlama70b: [32015]
            default: nil
            }
        }

        var sizeDescription: String {
            switch self {
            case .calme: "47.0 GB"
            case .dolphin72b: "47.5 GB"
            case .dolphinNemo: "8.8 GB"
            case .dolphinMixtral: "32.2 GB"
            case .deepSeekCoder33: "27.4 GB"
            case .deepSeekCoder7: "5.67 GB"
            case .mythoMax: "10.7 GB"
            case .whisper: "0.6 GB"
            case .dolphinTiny: "0.9 GB"
            case .qwen25large: "47.5 GB"
            case .qwen25regular: "20.0 GB"
            case .qwen25coder: "27.3 GB"
            case .qwen25medium: "11.0 GB"
            case .supernovaMedius: "10.5 GB"
            case .qwen25small: "4.5 GB"
            case .codeLlama70b: "48.8"
            case .samantha70b: "48.8 GB"
            case .samantha7b: "5.2 GB"
            case .everyoneCoder: "27.4 GB"
            case .neuralStory7b: "6.0 GB"
            case .dolphinCoder: "13.1 GB"
            case .llama3large: "48.7 GB"
            case .llama3: "6.6 GB"
            case .llama3tiny: "1.1 GB"
            case .llama3compact: "2.8 GB"
            case .codestral: "18.3 GB"
            case .smol: "1.5 GB"
            case .shuttle: "47.5 GB"
            }
        }

        var aboutText: String {
            switch self {
            case .deepSeekCoder33: "This no-nonsense model focuses specifically on code-related generation and questions."
            case .deepSeekCoder7: "A more compact version of the Deepseek Coder model, focusing on code-related generation and questions."
            case .dolphinMixtral: "A well rounded model, with multifaceted expertise and good conversational ability."
            case .mythoMax: "MythoMax is a model designed to be both imaginative, and useful for creativity and writing."
            case .dolphin72b: "An extra large size version of Dolphin for those with a lot of memory, curiosity and/or patience."
            case .whisper: "OpenAI's industry leading speech recognition. Lets you talk directly to the model if you prefer. Ensure you have a good mic and 'voice isolation' is selected from the menubar for best results."
            case .dolphinNemo: "The Dolhpin personality running on the Mistral Nemo base model."
            case .dolphinTiny: "The Doplhin chatbot running on the Tinyllama model, great for systems with constrained storage or processing requirements."
            case .qwen25large, .qwen25medium, .qwen25regular, .qwen25small: "A consistently well regarded all-round model by users and benchmarks."
            case .qwen25coder: "A consistently well regarded all-round model by users and benchmarks."
            case .codeLlama70b: "The latest large coding assistant model from Meta, for more intricate but obviously slower coding problems."
            case .samantha70b: "A larger but slightly older version of the Samantha model."
            case .samantha7b: "A wonderful conversation partner that feels genuinely friendly and caring. Especially good for voice conversations."
            case .everyoneCoder: "This is a community-created coding specific model made using fine-tunes of the Deekseekcoder base."
            case .neuralStory7b: "This fine-tune has been tailored to provide detailed and creative responses in the context of narrative, and optimised for short story telling."
            case .dolphinCoder: "The Dolphin personality applied to the very powerful StarCoder2 model."
            case .llama3large: "The large version of the Llama-3 model, with finetuning by NVIDIA."
            case .llama3: "The regular version of the latest Llama-3 model from Meta."
            case .llama3compact: "A compact, edge-optimised version of the Llama-3 model from Meta."
            case .llama3tiny: "The smallest, edge-optimised version of the Llama-3 model from Meta."
            case .codestral: "The state of the art code assistant from Mistral.AI"
            case .supernovaMedius: "By leveraging these two models, SuperNova-Medius achieves high-quality results in a mid-sized, efficient form."
            case .shuttle: "Shuttle-3 is a fine-tuned version of Qwen, emulating the writing style of Claude 3 models and thoroughly trained on role-playing data."
            case .smol: "A very capable mini-model by HuggingFace, currently with the top performance in the compact model range."
            case .calme: "Derived from Qwen using a method that allegedly improves performance, and finetuned for chat. Currently top of the open-source benchmarks."
            }
        }

        var maxBatch: UInt32 {
            switch self {
            case .codeLlama70b, .codestral, .deepSeekCoder7, .deepSeekCoder33, .dolphinCoder, .everyoneCoder, .qwen25coder: 4096
            case .calme, .dolphin72b, .dolphinMixtral, .dolphinNemo, .llama3, .llama3compact, .llama3large, .llama3tiny, .mythoMax, .neuralStory7b, .qwen25large, .qwen25medium, .qwen25regular, .qwen25small, .samantha7b, .samantha70b, .shuttle, .smol, .supernovaMedius: 1024
            case .dolphinTiny: 256
            case .whisper: 0
            }
        }

        var isCodingLLm: Bool {
            switch self {
            case .codeLlama70b, .codestral, .deepSeekCoder7, .deepSeekCoder33, .dolphinCoder, .everyoneCoder, .qwen25coder:
                true
            default:
                false
            }
        }

        private var defaultTopK: Int {
            90
        }

        private var defaultTopP: Float {
            0.9
        }

        private var defaultTemperature: Float {
            if isCodingLLm {
                0.1
            } else {
                0.7
            }
        }

        private var defaultTemperatureRange: Float {
            if isCodingLLm {
                0
            } else {
                0.2
            }
        }

        private var defaultTemperatureExponent: Float {
            1.0
        }

        private var defaultRepeatPenatly: Float {
            if isCodingLLm {
                1.0
            } else {
                1.17
            }
        }

        private var defaultFrequencyPenalty: Float {
            if isCodingLLm {
                0
            } else {
                0.1
            }
        }

        private var defaultPresentPenalty: Float {
            1
        }

        var quoteTag: String? {
            switch self {
            default: nil
            }
        }

        var emeltalRepo: URL {
            URL(string: "https://huggingface.co/acertainbru/emeltal-collection")!
        }

        var originalRepoUrl: URL {
            let uri = switch self {
            case .mythoMax: "https://huggingface.co/Gryphe/MythoMax-L2-13b"
            case .whisper: "https://huggingface.co/ggerganov/whisper.cpp"
            case .deepSeekCoder33: "https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct"
            case .deepSeekCoder7: "https://huggingface.co/deepseek-ai/deepseek-coder-7b-instruct-v1.5"
            case .codeLlama70b: "https://huggingface.co/codellama/CodeLlama-70b-Instruct-hf"
            case .everyoneCoder: "https://huggingface.co/rombodawg/Everyone-Coder-33b-v2-Base"
            case .codestral: "https://huggingface.co/mistralai/Codestral-22B-v0.1"
            case .dolphinCoder: "https://huggingface.co/cognitivecomputations/dolphincoder-starcoder2-15b"
            case .qwen25regular: "https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF"
            case .qwen25coder: "https://huggingface.co/bartowski/Qwen2.5-Coder-32B-Instruct-GGUF"
            case .qwen25large: "https://huggingface.co/bartowski/Qwen2.5-72B-Instruct-GGUF"
            case .qwen25medium: "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF"
            case .qwen25small: "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF"
            case .dolphinMixtral: "https://huggingface.co/cognitivecomputations/dolphin-2.7-mixtral-8x7b"
            case .dolphin72b: "https://huggingface.co/mradermacher/dolphin-2.9.2-qwen2-72b-i1-GGUF"
            case .dolphinNemo: "https://huggingface.co/cognitivecomputations/dolphin-2.9.3-mistral-nemo-12b-gguf"
            case .dolphinTiny: "https://huggingface.co/cognitivecomputations/TinyDolphin-2.8-1.1b"
            case .samantha70b: "https://huggingface.co/cognitivecomputations/Samantha-1.11-70b"
            case .samantha7b: "https://huggingface.co/cognitivecomputations/samantha-1.1-westlake-7b"
            case .neuralStory7b: "https://huggingface.co/NeuralNovel/Mistral-7B-Instruct-v0.2-Neural-Story"
            case .llama3large: "https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
            case .llama3: "https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct"
            case .llama3tiny: "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct"
            case .llama3compact: "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct"
            case .supernovaMedius: "https://huggingface.co/arcee-ai/SuperNova-Medius"
            case .smol: "https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct"
            case .shuttle: "https://huggingface.co/shuttleai/shuttle-3"
            case .calme: "https://huggingface.co/MaziyarPanahi/calme-2.4-rys-78b"
            }
            return URL(string: uri)!
        }

        var fileName: String {
            switch self {
            case .dolphinMixtral: "dolphin-2.7-mixtral-8x7b.Q5_K_M.gguf"
            case .deepSeekCoder33: "deepseek-coder-33b-instruct.Q6_K.gguf"
            case .deepSeekCoder7: "deepseek-coder-7b-instruct-v1.5-Q6_K.gguf"
            case .mythoMax: "mythomax-l2-13b.Q6_K.gguf"
            case .whisper: "ggml-large-v3-turbo-q5_0.bin"
            case .dolphin72b: "dolphin-2.9.2-qwen2-72b.i1-Q4_K_M.gguf"
            case .dolphinTiny: "tinydolphin-2.8-1.1b.Q6_K.gguf"
            case .dolphinNemo: "dolphin-2.9.3-mistral-nemo-12b.Q5_K_M.gguf"
            case .qwen25regular: "Qwen2.5-32B-Instruct-Q4_K_M.gguf"
            case .qwen25large: "Qwen2.5-72B-Instruct-Q4_K_M.gguf"
            case .qwen25coder: "Qwen2.5-Coder-32B-Instruct-Q6_K_L.gguf"
            case .qwen25medium: "Qwen2.5-14B-Instruct-Q5_K_L.gguf"
            case .qwen25small: "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
            case .codeLlama70b: "CodeLlama-70b-Instruct-hf-Q5_K_M.gguf"
            case .samantha70b: "samantha-1.11-70b.Q5_K_M.gguf"
            case .samantha7b: "samantha-1.1-westlake-7b.Q5_K_M.gguf"
            case .everyoneCoder: "Everyone-Coder-33b-v2-Base-Q6_K.gguf"
            case .neuralStory7b: "Mistral-7B-Instruct-v0.2-Neural-Story_Q6_K.gguf"
            case .dolphinCoder: "dolphincoder-starcoder2-15b.Q6_K.gguf"
            case .llama3large: "Llama-3.1-Nemotron-70B-Instruct-HF-Q5_K_S.gguf"
            case .llama3: "Meta-Llama-3.1-8B-Instruct-Q6_K.gguf"
            case .llama3tiny: "Llama-3.2-1B-Instruct-Q6_K_L.gguf"
            case .llama3compact: "Llama-3.2-3B-Instruct-Q6_K_L.gguf"
            case .codestral: "Codestral-22B-v0.1-Q6_K.gguf"
            case .supernovaMedius: "SuperNova-Medius-Q5_K_M.gguf"
            case .smol: "SmolLM2-1.7B-Instruct-Q6_K_L.gguf"
            case .shuttle: "shuttle-3-Q4_K_M.gguf"
            case .calme: "calme-2.4-rys-78b.i1-Q4_K_S.gguf"
            }
        }

        var fetchUrl: URL {
            emeltalRepo
                .appendingPathComponent("resolve", conformingTo: .directory)
                .appendingPathComponent("main", conformingTo: .directory)
                .appendingPathComponent(fileName)
        }

        var displayName: String {
            switch self {
            case .dolphinMixtral: "Dolphin"
            case .dolphinNemo: "Dolphin Nemo"
            case .deepSeekCoder33: "DeepSeek Coder"
            case .deepSeekCoder7: "DeepSeek Coder (Compact)"
            case .qwen25coder: "Qwen 2.5 Coder"
            case .mythoMax: "MythoMax Writing Assistant"
            case .whisper: "Whisper Voice Recognition"
            case .dolphin72b: "Dolphin (Large)"
            case .dolphinTiny: "Dolphin (Compact)"
            case .qwen25large: "Qwen 2.5 (Large)"
            case .qwen25regular: "Qwen 2.5"
            case .qwen25medium: "Qwen 2.5 (Medium)"
            case .qwen25small: "Qwen 2.5 (Compact)"
            case .codeLlama70b: "CodeLlama (Large)"
            case .samantha70b: "Samantha (Large)"
            case .samantha7b: "Samantha"
            case .everyoneCoder: "EveryoneCoder"
            case .neuralStory7b: "Neural Story"
            case .dolphinCoder: "Dolphin Coder"
            case .llama3large: "Llama 3.1 (Nemotron)"
            case .llama3: "Llama 3.1 (Regular)"
            case .llama3compact: "Llama 3.2 (Small)"
            case .llama3tiny: "Llama 3.2 (Compact)"
            case .codestral: "Codestral"
            case .supernovaMedius: "Supernova Medius"
            case .smol: "SmolLM 2"
            case .shuttle: "Shuttle 3"
            case .calme: "Calme 2.4"
            }
        }

        var detail: String {
            switch self {
            case .dolphinMixtral: "v2.7, on Mixtral 8x7b"
            case .deepSeekCoder33: "33b variant, on Llama2"
            case .deepSeekCoder7: "v1.5, on Llama2"
            case .mythoMax: "vL2 13b variant"
            case .whisper: "Large v3 Turbo"
            case .dolphin72b: "v2.9.2 on Qwen 2.5 72b"
            case .dolphinTiny: "v2.8, on TinyLlama"
            case .dolphinNemo: "v2.9.3 on Mistral Nemo 12b"
            case .qwen25large: "v2.5, 72b variant"
            case .qwen25regular: "v2.5, 32b variant"
            case .qwen25coder: "v2.5, 32b variant"
            case .qwen25medium: "v2.5, 14b variant"
            case .qwen25small: "v2.5, 7b variant"
            case .codeLlama70b: "70b HF variant, on Llama2"
            case .samantha70b: "v1.11, on Llama2"
            case .samantha7b: "v1.1, on WestLake"
            case .everyoneCoder: "v2, on DeepSeekCoder 33b"
            case .neuralStory7b: "on Mistral-Instruct 0.2"
            case .dolphinCoder: "on StarCoder2 15b"
            case .llama3large: "v3.1, finetuned, 70b params"
            case .llama3: "v3.1, 8b params"
            case .llama3compact: "v3.2, 3b params"
            case .llama3tiny: "v3.2, 1b params"
            case .supernovaMedius: "on LLama 3.1 405b & Qwen 2.5 14b"
            case .codestral: "22b params"
            case .smol: "v2, 1.7b variant"
            case .shuttle: "v2, on Qwen-2.5-72b-Instruct"
            case .calme: "v2.4, on Qwen 2 78b"
            }
        }

        var id: String {
            switch self {
            case .dolphinMixtral: "43678C6F-FB70-4EDB-9C15-3B75E7C483FA"
            case .deepSeekCoder33: "73FD5E35-94F3-4923-9E28-070564DF5B6E"
            case .mythoMax: "AA4B3287-CA79-466F-8F84-87486D701256"
            case .whisper: "0FCCC65B-BD2B-470C-AFE2-637FABDA95EE"
            case .dolphin72b: "26FD3A09-48C6-412C-A9C0-51F17A3E5C9A"
            case .dolphinTiny: "5CDE7417-9281-4186-9C53-921674E8DCC0"
            case .dolphinNemo: "006EFFA0-CFCA-4EB9-87C0-2F07BB0EB4CE"
            case .qwen25regular: "7D602BDC-DB4E-4DD1-8FC3-0A6A173B38DE"
            case .qwen25medium: "AC9644D5-E5FA-4804-B247-BB4DF21C88C7"
            case .qwen25coder: "ED72B569-DD85-40B0-B41A-6E1FA33B18B0"
            case .qwen25large: "56D9FEFF-788B-4319-9C01-564110C5275A"
            case .qwen25small: "50911087-9A0E-467B-AF5C-18132840C33D"
            case .deepSeekCoder7: "57A70BFB-4005-4B53-9404-3A2B107A6677"
            case .codeLlama70b: "C1B93F86-721B-4560-A398-A6E69BFCA99B"
            case .samantha70b: "259CD082-71B4-4CD4-8D52-08DC317CC41A"
            case .samantha7b: "52AD5BC7-0F1C-47DB-9DAD-F2DF17559E7B"
            case .everyoneCoder: "50E2E4E2-C42C-4558-B572-2BC2399E3134"
            case .neuralStory7b: "5506DA6E-5403-4BEC-BBA8-5D8F1046DCDD"
            case .dolphinCoder: "80D5B47C-E4ED-47B1-B1D3-F0EE0258670A"
            case .llama3large: "CAD7F881-CD06-46C5-B0FC-0AD54BC3CBAA"
            case .llama3: "2F547D7D-612B-4BA0-A42E-B17392346FA0"
            case .llama3compact: "8EBC25F2-8F1D-492E-8A55-9B67AFB3AA89"
            case .llama3tiny: "611A636C-59C0-451C-A435-FD6A9041DB37"
            case .codestral: "303D7134-7861-4167-B465-402DA071C685"
            case .supernovaMedius: "CDCA7E8F-7411-4AEC-A76B-2DB17A62BE3F"
            case .smol: "0767CF26-7090-4B85-A584-2ECAE5499C22"
            case .shuttle: "9044B741-783F-471B-8447-FB773AAEF051"
            case .calme: "5F0BEDAB-59B3-43B8-B4D7-65F9B16A8735"
            }
        }

        var defaultParams: Params {
            Params(topK: defaultTopK,
                   topP: defaultTopP,
                   systemPrompt: defaultPrompt,
                   temperature: defaultTemperature,
                   temperatureRange: defaultTemperatureRange,
                   temperatureExponent: defaultTemperatureExponent,
                   repeatPenatly: defaultRepeatPenatly,
                   frequencyPenatly: defaultFrequencyPenalty,
                   presentPenatly: defaultPresentPenalty,
                   version: Params.currentVersion)
        }
    }

    let id: String
    let category: Category
    let variant: Variant

    private var saveTimer: PopTimer?

    var params: Params {
        didSet {
            saveTimer?.push()
        }
    }

    private(set) var status: Status

    static var modelsDir = appDocumentsUrl.appendingPathComponent("models", conformingTo: .directory)

    nonisolated static func == (lhs: Model, rhs: Model) -> Bool {
        lhs.id == rhs.id
    }

    nonisolated func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }

    init(category: Category, variant: Variant) {
        let myId = "\(category.id)-\(variant.id)"
        id = myId
        self.category = category
        self.variant = variant
        status = .checking

        if let modelParams = Persisted.modelParams, let list = try? JSONDecoder().decode([ParamsHolder].self, from: modelParams), let mine = list.first(where: { $0.modelId == myId }) {
            params = mine.params
        } else {
            params = variant.defaultParams
        }

        saveTimer = PopTimer(timeInterval: 0.1) { [weak self] in
            self?.save()
        }

        updateInstalledStatus()
    }

    func unInstall() {
        let fm = FileManager.default
        try? fm.removeItem(at: localModelPath)
        try? fm.removeItem(at: localStatePath)
        status = .checking
        updateInstalledStatus()
    }

    func updateInstalledStatus() {
        if FileManager.default.fileExists(atPath: localModelPath.path) {
            let fetcher = AssetFetcher(fetching: self)
            status = .installed(fetcher)
            return
        }

        let task = Task.detached { [weak self] in
            guard let self else { return Model.Status.notReady }
            log("Checking availability of \(variant.displayName)")

            var request = URLRequest(url: variant.fetchUrl)
            request.httpMethod = "head"
            let response = try? await URLSession.shared.data(for: request).1 as? HTTPURLResponse
            let newStatus: Model.Status = if let code = response?.statusCode, code >= 200, code < 300 {
                variant.recommended ? .recommended : .available
            } else {
                .notReady
            }
            return newStatus
        }

        Task {
            let newStatus = await task.value
            if !Task.isCancelled, status != newStatus {
                status = newStatus
                log("Status for \(variant.displayName) determined to be \(newStatus)")
            }
        }
    }

    var localModelPath: URL {
        let modelDir = Self.modelsDir
        let fm = FileManager.default
        if !fm.fileExists(atPath: modelDir.path) {
            try! fm.createDirectory(at: modelDir, withIntermediateDirectories: true)
        }
        return modelDir.appendingPathComponent(variant.fileName)
    }

    var localStatePath: URL {
        let fm = FileManager.default
        let statePath = appDocumentsUrl.appendingPathComponent("states-\(id)", conformingTo: .directory)
        if !fm.fileExists(atPath: statePath.path) {
            try? fm.createDirectory(at: statePath, withIntermediateDirectories: true)
        }
        return statePath
    }

    func mlTemplate(in context: LlamaContext) -> Template? {
        Template(format: variant.format,
                 system: params.systemPrompt,
                 bosToken: context.bosToken)
    }

    func resetToDefaults() {
        params = variant.defaultParams
    }

    func save() {
        var list = if let modelParams = Persisted.modelParams, let list = try? JSONDecoder().decode([ParamsHolder].self, from: modelParams) {
            list
        } else {
            [ParamsHolder]()
        }

        let myParams = ParamsHolder(modelId: id, params: params)
        if let index = list.firstIndex(where: { $0.modelId == myParams.modelId }) {
            list[index] = myParams
        } else {
            list.append(myParams)
        }
        Persisted.modelParams = try? JSONEncoder().encode(list)
        log("Saved params for model \(id)")
    }

    func cancelInstall() {
        guard case let .installing(fetcher) = status else { return }
        fetcher.cancel()
        status = .available
    }

    func install() {
        let fetcher = AssetFetcher(fetching: self)
        status = .installing(fetcher)
    }

    private var cachedMemoryEstimate: MemoryEstimate?
    var memoryEstimate: MemoryEstimate {
        if let cachedMemoryEstimate { return cachedMemoryEstimate }
        let calculated = variant.memoryEstimate
        cachedMemoryEstimate = calculated
        return calculated
    }
}
