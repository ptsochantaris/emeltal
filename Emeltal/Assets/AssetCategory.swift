import Foundation
import Metal

extension Asset {
    enum Section: Int, CaseIterable, Identifiable {
        var id: Int { rawValue }

        case qwen, dolphin, samantha, coding, creative, llamas, deprecated

        var presentedModels: [Variant] {
            switch self {
            case .llamas:
                [.llama3, .llama3large, .llama3reflection]
            case .dolphin:
                [.dolphinMixtral, .dolphinTiny, .dolphin70b]
            case .qwen:
                [.qwen2regular, .qwen2large, .qwen2small]
            case .coding:
                [.codestral, .dolphinCoder, .deepSeekCoder33, .deepSeekCoder7, .everyoneCoder, .codeLlama70b]
            case .creative:
                [.mythoMax, .neuralStory7b]
            case .samantha:
                [.samantha7b, .samantha70b]
            case .deprecated:
                []
            }
        }

        var title: String {
            switch self {
            case .dolphin: "Dolphin"
            case .coding: "Coding"
            case .qwen: "Qwen"
            case .creative: "Creative"
            case .deprecated: "Deprecated"
            case .samantha: "Samantha"
            case .llamas: "Llamas"
            }
        }

        var description: String {
            switch self {
            case .dolphin: "The Dolphin dataset produces some of the best LLMs out there. This is a selection of models finetuned with this dataset."
            case .coding: "Models that can assist with programming, algorithms, and writing code."
            case .creative: "Models that can help with creative activities, such as writing. More will be added soon."
            case .deprecated: "Models from previous versions of Emeltal that are installed but no longer offered."
            case .samantha: "The \"sister\" of Dolphin, Samantha is a data set which produces models based on the premise they are sentient, and emotionally supportive of the user."
            case .qwen: "The Qwen models are consistently rated both highly in benchmarks and by users."
            case .llamas: "The llama is a quadruped which lives in big rivers like the Amazon. It has two ears, a heart, a forehead, and a beak for eating honey. But it is provided with fins for swimming."
            }
        }
    }

    enum Variant: Int, Identifiable, Codable, CaseIterable, Sendable {
        case dolphinMixtral = 100,
             dolphin70b = 200,
             dolphinTiny = 300,
             mythoMax = 800,
             codestral = 820,
             dolphinCoder = 850,
             deepSeekCoder33 = 900,
             codeLlama70b = 1000,
             deepSeekCoder7 = 1100,
             everyoneCoder = 1200,
             samantha7b = 1600,
             samantha70b = 1700,
             whisper = 1800,
             neuralStory7b = 2000,
             llama3 = 2300,
             llama3large = 2400,
             llama3reflection = 2410,
             qwen2regular = 2500,
             qwen2large = 2600,
             qwen2small = 2700

        var selectable: Bool {
            switch self {
            case .whisper: false
            default: true
            }
        }

        var recommended: Bool {
            self == .qwen2regular
        }

        var section: Section? {
            Section.allCases.first { $0.presentedModels.contains(self) }
        }

        var format: Template.Format {
            switch self {
            case .deepSeekCoder7, .deepSeekCoder33: .alpaca
            case .dolphin70b, .dolphinCoder, .dolphinMixtral, .dolphinTiny: .chatml
            case .mythoMax: .alpaca
            case .whisper: .alpaca
            case .qwen2large, .qwen2regular, .qwen2small: .chatml
            case .codeLlama70b: .llamaLarge
            case .samantha7b, .samantha70b: .vicuna
            case .everyoneCoder: .alpaca
            case .codestral, .neuralStory7b: .mistral
            case .llama3, .llama3large, .llama3reflection: .llama3
            }
        }

        private var defaultPrompt: String {
            switch self {
            case .codeLlama70b, .codestral, .deepSeekCoder7, .deepSeekCoder33, .everyoneCoder:
                "You are a helpful AI programming assistant."
            case .dolphin70b, .dolphinMixtral, .dolphinTiny, .llama3, .llama3large, .llama3reflection, .qwen2large, .qwen2regular, .qwen2small:
                "You are a helpful, respectful, friendly and honest conversation partner. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don’t know the answer to a question, please don’t share false information."
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
                 .dolphin70b,
                 .dolphinCoder,
                 .dolphinMixtral,
                 .dolphinTiny,
                 .everyoneCoder,
                 .mythoMax,
                 .neuralStory7b,
                 .qwen2small,
                 .samantha7b,
                 .samantha70b,
                 .whisper:
                0
            case .llama3,
                 .llama3large,
                 .llama3reflection,
                 .qwen2large,
                 .qwen2regular:
                16384
            }
        }

        struct GpuUsage {
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
        }

        var kvBytes: Int64 {
            let kvCache: Double = switch self {
            case .codeLlama70b: 640
            case .deepSeekCoder33: 3968
            case .deepSeekCoder7: 1920
            case .dolphin70b: 1280
            case .codestral: 1000
            case .dolphinMixtral: 4096
            case .llama3: 2048
            case .llama3large, .llama3reflection: 5120
            case .dolphinTiny: 89
            case .qwen2large: 5120
            case .qwen2regular: 4096
            case .qwen2small: 1800
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

        // TODO: Installing a new model does not update the selector if going back from the "grid" icon

        var usage: GpuUsage {
            let layerSizeM: Int64 = switch self {
            case .dolphinMixtral: 1070
            case .deepSeekCoder33: 460
            case .dolphinCoder: 350
            case .deepSeekCoder7: 160
            case .dolphinTiny: 48
            case .mythoMax: 260
            case .whisper: 1
            case .dolphin70b: 610
            case .qwen2large: 580
            case .qwen2regular: 290
            case .qwen2small: 150
            case .codeLlama70b: 610
            case .llama3large, .llama3reflection: 600
            case .llama3: 155
            case .samantha70b: 600
            case .samantha7b: 210
            case .neuralStory7b: 240
            case .everyoneCoder: 460
            case .codestral: 480
            }

            let totalLayers: Int64 = switch self {
            case .dolphinMixtral: 33
            case .dolphinCoder: 41
            case .deepSeekCoder33: 63
            case .deepSeekCoder7: 31
            case .dolphinTiny: 23
            case .mythoMax: 41
            case .whisper: 0
            case .dolphin70b: 81
            case .qwen2large: 81
            case .qwen2regular: 65
            case .qwen2small: 29
            case .codeLlama70b: 81
            case .samantha70b: 81
            case .samantha7b: 33
            case .neuralStory7b: 33
            case .everyoneCoder: 63
            case .llama3large, .llama3reflection: 81
            case .llama3: 33
            case .codestral: 57
            }

            let asrBytes: Int64 = 3_500_000_000
            let layerSize = layerSizeM * 1_000_000
            let totalRequiredMemory = (totalLayers * layerSize) + asrBytes + kvBytes
            let physicalMemory = Int64(ProcessInfo.processInfo.physicalMemory)
            let excessBytes = max(0, totalRequiredMemory - physicalMemory)

            guard let memoryBytes, asrBytes < memoryBytes.max else {
                return GpuUsage(layersUsed: 0,
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
                return GpuUsage(layersUsed: 0,
                                layersTotal: totalLayers,
                                offloadAsr: true,
                                offloadKvCache: false,
                                cpuUsageEstimateBytes: fittedLayerMemory + kvBytes,
                                gpuUsageEstimateBytes: asrBytes,
                                excessBytes: excessBytes)
            }

            let asrAndLayers = asrBytes + fittedLayerMemory

            if layersToFit < totalLayers {
                return GpuUsage(layersUsed: layersToFit,
                                layersTotal: totalLayers,
                                offloadAsr: true,
                                offloadKvCache: false,
                                cpuUsageEstimateBytes: min(physicalMemory, totalRequiredMemory - asrAndLayers),
                                gpuUsageEstimateBytes: asrAndLayers,
                                excessBytes: excessBytes)
            }

            let offLoadKv = maxVram - asrAndLayers > kvBytes
            return GpuUsage(layersUsed: layersToFit,
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
            case .dolphin70b: "48.8 GB"
            case .dolphinMixtral: "32.2 GB"
            case .deepSeekCoder33: "27.4 GB"
            case .deepSeekCoder7: "5.67 GB"
            case .mythoMax: "10.7 GB"
            case .whisper: "1.1 GB"
            case .dolphinTiny: "0.9 GB"
            case .qwen2large: "47.4 GB"
            case .qwen2regular: "20.0 GB"
            case .qwen2small: "4.5 GB"
            case .codeLlama70b: "48.8"
            case .samantha70b: "48.8 GB"
            case .samantha7b: "5.2 GB"
            case .everyoneCoder: "27.4 GB"
            case .neuralStory7b: "6.0 GB"
            case .dolphinCoder: "13.1 GB"
            case .llama3large: "50.0 GB"
            case .llama3reflection: "48.7 GB"
            case .llama3: "6.6 GB"
            case .codestral: "18.3 GB"
            }
        }

        var aboutText: String {
            switch self {
            case .deepSeekCoder33: "This no-nonsense model focuses specifically on code-related generation and questions"
            case .deepSeekCoder7: "A more compact version of the Deepseek Coder model, focusing on code-related generation and questions"
            case .dolphinMixtral: "The current state of the art, with multifaceted expertise and good conversational ability."
            case .mythoMax: "MythoMax is a model designed to be both imaginative, and useful for creativity and writing."
            case .dolphin70b: "An extra large size version of Dolphin for those with a lot of memory, curiosity and/or patience."
            case .whisper: "OpenAI's industry leading speech recognition. Lets you talk directly to the model if you prefer. Ensure you have a good mic and 'voice isolation' is selected from the menubar for best results."
            case .dolphinTiny: "The Doplhin chatbot running on the Tinyllama model, great for systems with constrained storage or processing requirements."
            case .qwen2large, .qwen2regular, .qwen2small: "A consistently well regarded all-round model by users and benchmarks."
            case .codeLlama70b: "The latest large coding assistant model from Meta, for more intricate but obviously slower coding problems."
            case .samantha70b: "A larger but slightly older version of the Samantha model."
            case .samantha7b: "A wonderful conversation partner that feels genuinely friendly and caring. Especially good for voice conversations."
            case .everyoneCoder: "This is a community-created coding specific model made using fine-tunes of the Deekseekcoder base."
            case .neuralStory7b: "This fine-tune has been tailored to provide detailed and creative responses in the context of narrative, and optimised for short story telling."
            case .dolphinCoder: "The Dolphin personality applied to the very powerful StarCoder2 model."
            case .llama3large: "The large version of the latest Llama-3 model from Meta"
            case .llama3reflection: "The smarter, retuned, Reflection variant of the Llama-3 model from Meta. Will display its chain-of-thought in the response."
            case .llama3: "The compact version of the latest Llama-3 model from Meta"
            case .codestral: "The state of the art code assistant from Mistral.AI"
            }
        }

        var maxBatch: UInt32 {
            switch self {
            case .codeLlama70b, .codestral, .deepSeekCoder7, .deepSeekCoder33, .dolphinCoder, .everyoneCoder: 4096
            case .dolphin70b, .dolphinMixtral, .llama3, .llama3large, .llama3reflection, .mythoMax, .neuralStory7b, .qwen2large, .qwen2regular, .qwen2small, .samantha7b, .samantha70b: 1024
            case .dolphinTiny: 256
            case .whisper: 0
            }
        }

        var isCodingLLm: Bool {
            switch self {
            case .codeLlama70b, .codestral, .deepSeekCoder7, .deepSeekCoder33, .dolphinCoder, .everyoneCoder:
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
            case .llama3reflection: "thinking"
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
            case .qwen2regular: "https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF"
            case .qwen2large: "https://huggingface.co/bartowski/Qwen2.5-Math-72B-Instruct-GGUF"
            case .qwen2small: "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF"
            case .dolphinMixtral: "https://huggingface.co/cognitivecomputations/dolphin-2.7-mixtral-8x7b"
            case .dolphin70b: "https://huggingface.co/cognitivecomputations/dolphin-2.2-70b"
            case .dolphinTiny: "https://huggingface.co/cognitivecomputations/TinyDolphin-2.8-1.1b"
            case .samantha70b: "https://huggingface.co/cognitivecomputations/Samantha-1.11-70b"
            case .samantha7b: "https://huggingface.co/cognitivecomputations/samantha-1.1-westlake-7b"
            case .neuralStory7b: "https://huggingface.co/NeuralNovel/Mistral-7B-Instruct-v0.2-Neural-Story"
            case .llama3large: "https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct"
            case .llama3reflection: "https://huggingface.co/mattshumer/Reflection-Llama-3.1-70B"
            case .llama3: "https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct"
            }
            return URL(string: uri)!
        }

        var fetchUrl: URL {
            let fileName = switch self {
            case .dolphinMixtral: "dolphin-2.7-mixtral-8x7b.Q5_K_M.gguf"
            case .deepSeekCoder33: "deepseek-coder-33b-instruct.Q6_K.gguf"
            case .deepSeekCoder7: "deepseek-coder-7b-instruct-v1.5-Q6_K.gguf"
            case .mythoMax: "mythomax-l2-13b.Q6_K.gguf"
            case .whisper: "ggml-large-v3-q5_k.bin"
            case .dolphin70b: "dolphin-2.2-70b.Q5_K_M.gguf"
            case .dolphinTiny: "tinydolphin-2.8-1.1b.Q6_K.gguf"
            case .qwen2regular: "Qwen2.5-32B-Instruct-Q4_K_M.gguf"
            case .qwen2large: "Qwen2.5-72B-Instruct-Q4_K_M.gguf"
            case .qwen2small: "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
            case .codeLlama70b: "CodeLlama-70b-Instruct-hf-Q5_K_M.gguf"
            case .samantha70b: "samantha-1.11-70b.Q5_K_M.gguf"
            case .samantha7b: "samantha-1.1-westlake-7b.Q5_K_M.gguf"
            case .everyoneCoder: "Everyone-Coder-33b-v2-Base-Q6_K.gguf"
            case .neuralStory7b: "Mistral-7B-Instruct-v0.2-Neural-Story_Q6_K.gguf"
            case .dolphinCoder: "dolphincoder-starcoder2-15b.Q6_K.gguf"
            case .llama3large: "Meta-Llama-3.1-70B-Instruct.Q5_K_M.gguf"
            case .llama3: "Meta-Llama-3.1-8B-Instruct-Q6_K.gguf"
            case .llama3reflection: "Reflection-Llama-3.1-70B.Q5_K_S.gguf"
            case .codestral: "Codestral-22B-v0.1-Q6_K.gguf"
            }

            return emeltalRepo
                .appendingPathComponent("resolve", conformingTo: .directory)
                .appendingPathComponent("main", conformingTo: .directory)
                .appendingPathComponent(fileName)
        }

        var displayName: String {
            switch self {
            case .dolphinMixtral: "Dolphin"
            case .deepSeekCoder33: "DeepSeek Coder"
            case .deepSeekCoder7: "DeepSeek Coder (Compact)"
            case .mythoMax: "MythoMax Writing Assistant"
            case .whisper: "Whisper Voice Recognition"
            case .dolphin70b: "Dolphin (Large)"
            case .dolphinTiny: "Dolphin (Compact)"
            case .qwen2large: "Qwen2 (Large)"
            case .qwen2regular: "Qwen2"
            case .qwen2small: "Qwen2 (Compact)"
            case .codeLlama70b: "CodeLlama (Large)"
            case .samantha70b: "Samantha (Large)"
            case .samantha7b: "Samantha"
            case .everyoneCoder: "EveryoneCoder"
            case .neuralStory7b: "Neural Story"
            case .dolphinCoder: "Dolphin Coder"
            case .llama3large: "Llama 3 Large"
            case .llama3: "Llama 3"
            case .llama3reflection: "Llama 3 Reflection"
            case .codestral: "Codestral"
            }
        }

        var detail: String {
            switch self {
            case .dolphinMixtral: "v2.7, on Mixtral 8x7b"
            case .deepSeekCoder33: "33b variant, on Llama2"
            case .deepSeekCoder7: "v1.5, on Llama2"
            case .mythoMax: "vL2 13b variant"
            case .whisper: "Large v3"
            case .dolphin70b: "on Llama 70b (x2)"
            case .dolphinTiny: "v2.8, on TinyLlama"
            case .qwen2large: "v2.5, 72b variant"
            case .qwen2regular: "v2.5, 32b variant"
            case .qwen2small: "v2.5, 7b variant"
            case .codeLlama70b: "70b HF variant, on Llama2"
            case .samantha70b: "v1.11, on Llama2"
            case .samantha7b: "v1.1, on WestLake"
            case .everyoneCoder: "v2, on DeepSeekCoder 33b"
            case .neuralStory7b: "on Mistral-Instruct 0.2"
            case .dolphinCoder: "on StarCoder2 15b"
            case .llama3large: "v3.1, 70b params"
            case .llama3: "v3.1, 8b params"
            case .llama3reflection: "Retuned variant, on Llama v3.1 70b"
            case .codestral: "22b params"
            }
        }

        var id: String {
            switch self {
            case .dolphinMixtral: "43678C6F-FB70-4EDB-9C15-3B75E7C483FA"
            case .deepSeekCoder33: "73FD5E35-94F3-4923-9E28-070564DF5B6E"
            case .mythoMax: "AA4B3287-CA79-466F-8F84-87486D701256"
            case .whisper: "0FCCC65B-BD2B-470C-AFE2-637FABDA95EE"
            case .dolphin70b: "0D70BC73-9559-4778-90A6-E5F2E4B71213"
            case .dolphinTiny: "5CDE7417-9281-4186-9C53-921674E8DCC0"
            case .qwen2regular: "7D602BDC-DB4E-4DD1-8FC3-0A6A173B38DE"
            case .qwen2large: "56D9FEFF-788B-4319-9C01-564110C5275A"
            case .qwen2small: "50911087-9A0E-467B-AF5C-18132840C33D"
            case .deepSeekCoder7: "57A70BFB-4005-4B53-9404-3A2B107A6677"
            case .codeLlama70b: "C1B93F86-721B-4560-A398-A6E69BFCA99B"
            case .samantha70b: "259CD082-71B4-4CD4-8D52-08DC317CC41A"
            case .samantha7b: "52AD5BC7-0F1C-47DB-9DAD-F2DF17559E7B"
            case .everyoneCoder: "50E2E4E2-C42C-4558-B572-2BC2399E3134"
            case .neuralStory7b: "5506DA6E-5403-4BEC-BBA8-5D8F1046DCDD"
            case .dolphinCoder: "80D5B47C-E4ED-47B1-B1D3-F0EE0258670A"
            case .llama3large: "15D7333B-ECC3-4499-BB41-457F2007F207"
            case .llama3: "2F547D7D-612B-4BA0-A42E-B17392346FA0"
            case .codestral: "303D7134-7861-4167-B465-402DA071C685"
            case .llama3reflection: "246EC625-7F21-4797-9988-3F733D8CEBD6"
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
}
