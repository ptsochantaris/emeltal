import Foundation
import Metal

extension Model {
    enum Variant: Identifiable {
        case dolphinThreeR1,
             dolphinThree8b,
             dolphinThree3b,
             dolphinThreeTiny,
             dolphinNemo,
             dolphin72b,
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
             qwenQwQ32,
             supernovaMedius,
             smol,
             shuttle,
             athene,
             calme,
             gemma327,
             gemma312,
             gemma34,
             gemma31,
             dsro70,
             olympicCoder

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
            case .dsro70, .llama3, .llama3compact, .llama3large, .llama3tiny: .llama3
            case .deepSeekCoder7, .deepSeekCoder33, .everyoneCoder, .mythoMax, .whisper: .alpaca
            case .athene, .calme, .dolphin72b, .dolphinCoder, .dolphinNemo, .dolphinThree3b, .dolphinThree8b, .dolphinThreeR1, .dolphinThreeTiny, .qwen25coder, .qwen25large, .qwen25medium, .qwen25regular, .qwen25small, .qwenQwQ32, .shuttle, .smol, .supernovaMedius, .olympicCoder: .chatml
            case .gemma31, .gemma34, .gemma312, .gemma327: .gemma
            }
        }

        private var defaultPrompt: String {
            switch self {
            case .codeLlama70b, .codestral, .deepSeekCoder7, .deepSeekCoder33, .everyoneCoder, .olympicCoder, .qwen25coder:
                "You are a helpful AI programming assistant."
            case .athene, .calme, .dolphin72b, .dolphinNemo, .dolphinThree3b, .dolphinThree8b, .dolphinThreeR1, .dolphinThreeTiny, .gemma31, .gemma34, .gemma312, .gemma327, .llama3, .llama3compact, .llama3large, .llama3tiny, .qwen25large, .qwen25medium, .qwen25regular, .qwen25small, .qwenQwQ32, .shuttle, .smol, .supernovaMedius:
                "You are a friendly and honest conversation partner. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don’t know the answer to a question, please don’t share false information."
            case .samantha7b, .samantha70b:
                "You are a caring and empathetic sentient AI companion named Samantha."
            case .mythoMax, .neuralStory7b:
                "You are a helpful, imaginative, collaborative, and friendly writing assistant."
            case .dolphinCoder:
                "You are DolphinCoder, a helpful AI programming assistant."
            case .dsro70, .whisper:
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
                 .everyoneCoder,
                 .mythoMax,
                 .neuralStory7b,
                 .olympicCoder,
                 .qwen25small,
                 .samantha7b,
                 .samantha70b,
                 .smol,
                 .whisper:
                0
            case .athene,
                 .calme,
                 .dolphin72b,
                 .dolphinNemo,
                 .dolphinThree3b,
                 .dolphinThree8b,
                 .dolphinThreeR1,
                 .dolphinThreeTiny,
                 .dsro70,
                 .gemma31,
                 .gemma34,
                 .gemma312,
                 .gemma327,
                 .llama3,
                 .llama3compact,
                 .llama3large,
                 .llama3tiny,
                 .qwen25coder,
                 .qwen25large,
                 .qwen25medium,
                 .qwen25regular,
                 .qwenQwQ32,
                 .shuttle,
                 .supernovaMedius:
                16384
            }
        }

        var kvBytes: Int64 {
            let kvCache: Double = switch self {
            case .gemma31: 416
            case .gemma34: 2176
            case .gemma312: 6144
            case .gemma327: 7936
            case .codeLlama70b: 640
            case .deepSeekCoder33: 3968
            case .deepSeekCoder7: 1920
            case .dolphin72b: 5120
            case .codestral: 7168
            case .dolphinNemo: 3072
            case .llama3: 2048
            case .llama3compact: 1792
            case .llama3tiny: 512
            case .llama3large: 5120
            case .calme: 5504
            case .qwen25large: 5120
            case .qwen25regular: 4096
            case .qwen25coder: 4096
            case .qwen25medium: 3072
            case .qwenQwQ32: 4096
            case .athene: 5120
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
            case .dsro70: 5120
            case .dolphinThreeTiny: 448
            case .dolphinThree3b: 576
            case .dolphinThree8b: 2048
            case .dolphinThreeR1: 2560
            case .olympicCoder: 8192
            case .whisper: 0
            }
            return Int64((kvCache * 1_048_576).rounded(.up))
        }

        var memoryEstimate: MemoryEstimate {
            let layerSizeM: Int64 = switch self {
            case .deepSeekCoder33: 460
            case .dolphinCoder: 320
            case .deepSeekCoder7: 180
            case .mythoMax: 260
            case .whisper: 1
            case .dolphin72b: 600
            case .dolphinNemo: 200
            case .calme: 550
            case .qwen25large: 600
            case .qwenQwQ32: 380
            case .qwen25regular: 310
            case .qwen25coder: 430
            case .qwen25medium: 220
            case .qwen25small: 160
            case .athene: 580
            case .supernovaMedius: 220
            case .codeLlama70b: 610
            case .llama3large: 600
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
            case .dsro70: 535
            case .dolphinThreeR1: 430
            case .dolphinThreeTiny: 54
            case .dolphinThree3b: 76
            case .dolphinThree8b: 174
            case .gemma31: 45
            case .gemma34: 83
            case .gemma312: 160
            case .gemma327: 270
            case .olympicCoder: 380
            }

            let totalLayers: Int64 = switch self {
            case .dolphinNemo: 41
            case .dolphinCoder: 41
            case .deepSeekCoder33: 63
            case .deepSeekCoder7: 31
            case .mythoMax: 41
            case .whisper: 1
            case .dolphin72b: 81
            case .smol: 25
            case .shuttle: 81
            case .calme: 87
            case .qwen25large: 81
            case .qwen25regular: 65
            case .qwen25coder: 65
            case .qwen25medium: 49
            case .qwen25small: 29
            case .qwenQwQ32: 65
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
            case .athene: 81
            case .dsro70: 81
            case .dolphinThreeR1: 41
            case .dolphinThreeTiny: 29
            case .dolphinThree3b: 37
            case .dolphinThree8b: 33
            case .gemma31: 27
            case .gemma34: 35
            case .gemma312: 49
            case .gemma327: 63
            case .olympicCoder: 65
            }

            let layerSize = layerSizeM * 1_000_000

            let outputLayerSize: Int64 = switch self {
            case .athene: 5_000_000_000
            case .dsro70, .llama3large: 5_000_000_000
            case .whisper: 0
            default: layerSize * 2 // catch-all
            }

            guard let memoryBytes else {
                return MemoryEstimate(layersOffloaded: 0,
                                      layersTotal: totalLayers,
                                      offloadAsr: false,
                                      offloadKvCache: false,
                                      nonOffloadedEstimateBytes: 0,
                                      gpuUsageEstimateBytes: 0,
                                      totalSystemBytes: 0,
                                      unifiedMemory: false)
            }

            let asrBytes: Int64 = 1_000_000_000
            var components: [Int64] = [asrBytes] + (0 ..< totalLayers - 1).map { _ in layerSize } + [outputLayerSize, kvBytes]
            var cpuBound = [Int64]()

            let everythingInGpu = components.reduce(0, +)
            if everythingInGpu < memoryBytes.max {
                return MemoryEstimate(layersOffloaded: totalLayers,
                                      layersTotal: totalLayers,
                                      offloadAsr: true,
                                      offloadKvCache: true,
                                      nonOffloadedEstimateBytes: cpuBound.reduce(0, +),
                                      gpuUsageEstimateBytes: everythingInGpu,
                                      totalSystemBytes: memoryBytes.systemTotal,
                                      unifiedMemory: memoryBytes.unifiedMemory)
            }

            if let last = components.last { cpuBound.append(last) }
            components = components.dropLast()

            let minusKv = components.reduce(0, +)
            if minusKv < memoryBytes.max {
                return MemoryEstimate(layersOffloaded: totalLayers,
                                      layersTotal: totalLayers,
                                      offloadAsr: true,
                                      offloadKvCache: false,
                                      nonOffloadedEstimateBytes: cpuBound.reduce(0, +),
                                      gpuUsageEstimateBytes: minusKv,
                                      totalSystemBytes: memoryBytes.systemTotal,
                                      unifiedMemory: memoryBytes.unifiedMemory)
            }

            if let last = components.last { cpuBound.append(last) }
            components = components.dropLast()

            let minusOutputLayer = components.reduce(0, +)
            if minusOutputLayer < memoryBytes.max {
                return MemoryEstimate(layersOffloaded: totalLayers - 1,
                                      layersTotal: totalLayers,
                                      offloadAsr: true,
                                      offloadKvCache: false,
                                      nonOffloadedEstimateBytes: cpuBound.reduce(0, +),
                                      gpuUsageEstimateBytes: minusOutputLayer,
                                      totalSystemBytes: memoryBytes.systemTotal,
                                      unifiedMemory: memoryBytes.unifiedMemory)
            }

            for layer in 0 ..< (totalLayers - 1) {
                if let last = components.last { cpuBound.append(last) }
                components = components.dropLast()

                let minusLayer = components.reduce(0, +)
                if minusLayer < memoryBytes.max {
                    return MemoryEstimate(layersOffloaded: totalLayers - 1 - layer,
                                          layersTotal: totalLayers,
                                          offloadAsr: true,
                                          offloadKvCache: false,
                                          nonOffloadedEstimateBytes: cpuBound.reduce(0, +),
                                          gpuUsageEstimateBytes: minusLayer,
                                          totalSystemBytes: memoryBytes.systemTotal,
                                          unifiedMemory: memoryBytes.unifiedMemory)
                }
            }

            if let last = components.last { cpuBound.append(last) }
            components = components.dropLast()

            let totalCpuUse = cpuBound.reduce(0, +)
            if totalCpuUse < memoryBytes.systemTotal {
                return MemoryEstimate(layersOffloaded: 0,
                                      layersTotal: totalLayers,
                                      offloadAsr: false,
                                      offloadKvCache: false,
                                      nonOffloadedEstimateBytes: totalCpuUse,
                                      gpuUsageEstimateBytes: 0,
                                      totalSystemBytes: memoryBytes.systemTotal,
                                      unifiedMemory: memoryBytes.unifiedMemory)
            }

            return MemoryEstimate(layersOffloaded: 0,
                                  layersTotal: totalLayers,
                                  offloadAsr: false,
                                  offloadKvCache: false,
                                  nonOffloadedEstimateBytes: totalCpuUse,
                                  gpuUsageEstimateBytes: 0,
                                  totalSystemBytes: memoryBytes.systemTotal,
                                  unifiedMemory: memoryBytes.unifiedMemory)
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
            case .deepSeekCoder33: "27.4 GB"
            case .deepSeekCoder7: "5.67 GB"
            case .mythoMax: "10.7 GB"
            case .whisper: "0.6 GB"
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
            case .athene: "48.3 GB"
            case .dsro70: "42.5 GB"
            case .dolphinThreeR1: "17.2 GB"
            case .dolphinThreeTiny: "1.4 GB"
            case .dolphinThree3b: "2.7 GB"
            case .dolphinThree8b: "5.8 GB"
            case .qwenQwQ32: "23.8 GB"
            case .gemma31: "0.8 GB"
            case .gemma34: "2.5 GB"
            case .gemma312: "7.3 GB"
            case .gemma327: "16.6 GB"
            case .olympicCoder: "23.8 GB"
            }
        }

        var aboutText: String {
            switch self {
            case .deepSeekCoder33: "This no-nonsense model focuses specifically on code-related generation and questions."
            case .deepSeekCoder7: "A more compact version of the Deepseek Coder model, focusing on code-related generation and questions."
            case .mythoMax: "MythoMax is a model designed to be both imaginative, and useful for creativity and writing."
            case .dolphin72b: "An extra large size version of Dolphin for those with a lot of memory, curiosity and/or patience."
            case .whisper: "OpenAI's industry leading speech recognition. Lets you talk directly to the model if you prefer. Ensure you have a good mic and 'voice isolation' is selected from the menubar for best results."
            case .dolphinNemo: "The Dolhpin personality running on the Mistral Nemo base model."
            case .qwenQwQ32: "An evolution of the Qwen model that, at least on benchmarks, offers a very high quality of reasoning. It will display its thinking process by default."
            case .qwen25large, .qwen25medium, .qwen25regular, .qwen25small: "A consistently well regarded all-round model by users and benchmarks."
            case .qwen25coder: "A consistently well regarded all-round model by users and benchmarks."
            case .codeLlama70b: "The latest large coding assistant model from Meta, for more intricate but obviously slower coding problems."
            case .samantha70b: "A larger but slightly older version of the Samantha model."
            case .samantha7b: "A wonderful conversation partner that feels genuinely friendly and caring. Especially good for voice conversations."
            case .everyoneCoder: "This is a community-created coding specific model made using fine-tunes of the Deekseekcoder base."
            case .neuralStory7b: "This fine-tune has been tailored to provide detailed and creative responses in the context of narrative, and optimised for short story telling."
            case .dolphinCoder: "The Dolphin personality applied to the very powerful StarCoder2 model."
            case .llama3large: "The largest and most recent version of the Llama-3 model."
            case .llama3: "The regular version of the latest Llama-3 model from Meta."
            case .llama3compact: "A compact, edge-optimised version of the Llama-3 model from Meta."
            case .llama3tiny: "The smallest, edge-optimised version of the Llama-3 model from Meta."
            case .codestral: "The state of the art code assistant from Mistral.AI"
            case .supernovaMedius: "By leveraging these two models, SuperNova-Medius achieves high-quality results in a mid-sized, efficient form."
            case .shuttle: "Shuttle-3 is a fine-tuned version of Qwen, emulating the writing style of Claude 3 models and thoroughly trained on role-playing data."
            case .smol: "A very capable mini-model by HuggingFace, currently with the top performance in the compact model range."
            case .calme: "Derived from Qwen using a method that allegedly improves performance, and finetuned for chat. Currently top of the open-source benchmarks."
            case .athene: "Athene is based on Qwen and seems to have promise, with very high benchmark results, even above Nemotron Llama."
            case .dsro70: "Distill of DeepSeek R1 on the Llama 70b model"
            case .dolphinThreeR1: "Dolphin 3.0 R1 combines the latest release with training data from the R1 model."
            case .dolphinThreeTiny: "The smallest and lightest version of Dolphin."
            case .dolphinThree3b: "A compact simplified version of Dolphin for low memory environments."
            case .dolphinThree8b: "The \"regular\" Dolphin model, a great default starting point for lower memory systems."
            case .gemma31, .gemma34, .gemma312, .gemma327: "A quantised variant of the Gemma 3 model."
            case .olympicCoder: "Achieves strong performance on competitive coding benchmarks such as LiveCodeBench and 2024 IOI"
            }
        }

        var isCodingLLm: Bool {
            switch self {
            case .codeLlama70b, .codestral, .deepSeekCoder7, .deepSeekCoder33, .dolphinCoder, .everyoneCoder, .olympicCoder, .qwen25coder:
                true
            default:
                false
            }
        }

        private var defaultTopK: Int {
            if case .qwenQwQ32 = self {
                40
            } else {
                90
            }
        }

        private var defaultTopP: Float {
            if case .qwenQwQ32 = self {
                0.95
            } else {
                0.9
            }
        }

        private var defaultTemperature: Float {
            if case .qwenQwQ32 = self {
                0.6
            } else if isCodingLLm {
                0.1
            } else {
                0.7
            }
        }

        private var defaultTemperatureRange: Float {
            if case .qwenQwQ32 = self {
                0
            } else if isCodingLLm {
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

        private var defaultRepeatCheckPenalty: Int {
            if isCodingLLm {
                0
            } else {
                64
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
            case .qwenQwQ32: "https://huggingface.co/bartowski/Qwen_QwQ-32B-GGUF"
            case .dolphin72b: "https://huggingface.co/mradermacher/dolphin-2.9.2-qwen2-72b-i1-GGUF"
            case .dolphinNemo: "https://huggingface.co/cognitivecomputations/dolphin-2.9.3-mistral-nemo-12b-gguf"
            case .samantha70b: "https://huggingface.co/cognitivecomputations/Samantha-1.11-70b"
            case .samantha7b: "https://huggingface.co/cognitivecomputations/samantha-1.1-westlake-7b"
            case .neuralStory7b: "https://huggingface.co/NeuralNovel/Mistral-7B-Instruct-v0.2-Neural-Story"
            case .llama3large: "https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF"
            case .llama3: "https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct"
            case .llama3tiny: "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct"
            case .llama3compact: "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct"
            case .supernovaMedius: "https://huggingface.co/arcee-ai/SuperNova-Medius"
            case .smol: "https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct"
            case .shuttle: "https://huggingface.co/shuttleai/shuttle-3"
            case .calme: "https://huggingface.co/MaziyarPanahi/calme-2.4-rys-78b"
            case .athene: "https://huggingface.co/bartowski/Athene-V2-Chat-GGUF"
            case .dsro70: "https://huggingface.co/collections/unsloth/deepseek-r1-all-versions-678e1c48f5d2fce87892ace5"
            case .dolphinThreeTiny: "https://huggingface.co/bartowski/Dolphin3.0-Qwen2.5-1.5B-GGUF"
            case .dolphinThree3b: "https://huggingface.co/bartowski/Dolphin3.0-Qwen2.5-3b-GGUF"
            case .dolphinThree8b: "https://huggingface.co/cognitivecomputations/Dolphin3.0-Llama3.1-8B-GGUF"
            case .dolphinThreeR1: "https://huggingface.co/bartowski/cognitivecomputations_Dolphin3.0-R1-Mistral-24B-GGUF"
            case .gemma31: "https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF"
            case .gemma34: "https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF"
            case .gemma312: "https://huggingface.co/ggml-org/gemma-3-12b-it-GGUF"
            case .gemma327: "https://huggingface.co/ggml-org/gemma-3-27b-it-GGUF"
            case .olympicCoder: "https://huggingface.co/bartowski/open-r1_OlympicCoder-32B-GGUF"
            }
            return URL(string: uri)!
        }

        var fileName: String {
            switch self {
            case .deepSeekCoder33: "deepseek-coder-33b-instruct.Q6_K.gguf"
            case .deepSeekCoder7: "deepseek-coder-7b-instruct-v1.5-Q6_K.gguf"
            case .mythoMax: "mythomax-l2-13b.Q6_K.gguf"
            case .whisper: "ggml-large-v3-turbo-q5_0.bin"
            case .dolphin72b: "dolphin-2.9.2-qwen2-72b.i1-Q4_K_M.gguf"
            case .dolphinNemo: "dolphin-2.9.3-mistral-nemo-12b.Q5_K_M.gguf"
            case .qwen25regular: "Qwen2.5-32B-Instruct-Q4_K_M.gguf"
            case .qwen25large: "Qwen2.5-72B-Instruct-Q4_K_M.gguf"
            case .qwen25coder: "Qwen2.5-Coder-32B-Instruct-Q6_K_L.gguf"
            case .qwen25medium: "Qwen2.5-14B-Instruct-Q5_K_L.gguf"
            case .qwen25small: "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
            case .qwenQwQ32: "Qwen_QwQ-32B-Q5_K_L.gguf"
            case .codeLlama70b: "CodeLlama-70b-Instruct-hf-Q5_K_M.gguf"
            case .samantha70b: "samantha-1.11-70b.Q5_K_M.gguf"
            case .samantha7b: "samantha-1.1-westlake-7b.Q5_K_M.gguf"
            case .everyoneCoder: "Everyone-Coder-33b-v2-Base-Q6_K.gguf"
            case .neuralStory7b: "Mistral-7B-Instruct-v0.2-Neural-Story_Q6_K.gguf"
            case .dolphinCoder: "dolphincoder-starcoder2-15b.Q6_K.gguf"
            case .llama3large: "Llama-3.3-70B-Instruct-Q5_K_S.gguf"
            case .llama3: "Meta-Llama-3.1-8B-Instruct-Q6_K.gguf"
            case .llama3tiny: "Llama-3.2-1B-Instruct-Q6_K_L.gguf"
            case .llama3compact: "Llama-3.2-3B-Instruct-Q6_K_L.gguf"
            case .codestral: "Codestral-22B-v0.1-Q6_K.gguf"
            case .supernovaMedius: "SuperNova-Medius-Q5_K_M.gguf"
            case .smol: "SmolLM2-1.7B-Instruct-Q6_K_L.gguf"
            case .shuttle: "shuttle-3-Q4_K_M.gguf"
            case .calme: "calme-2.4-rys-78b.i1-Q4_K_S.gguf"
            case .athene: "Athene-V2-Chat-Q4_K_L.gguf"
            case .dsro70: "DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf"
            case .dolphinThreeR1: "cognitivecomputations_Dolphin3.0-R1-Mistral-24B-Q5_K_L.gguf"
            case .dolphinThreeTiny: "Dolphin3.0-Qwen2.5-1.5B-Q6_K_L.gguf"
            case .dolphinThree3b: "Dolphin3.0-Qwen2.5-3b-Q6_K_L.gguf"
            case .dolphinThree8b: "Dolphin3.0-Llama3.1-8B-Q5_K_M.gguf"
            case .gemma31: "gemma-3-1b-it-Q4_K_M.gguf"
            case .gemma34: "gemma-3-4b-it-Q4_K_M.gguf"
            case .gemma312: "gemma-3-12b-it-Q4_K_M.gguf"
            case .gemma327: "gemma-3-27b-it-Q4_K_M.gguf"
            case .olympicCoder: "open-r1_OlympicCoder-32B-Q5_K_L.gguf"
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
            case .dolphinNemo: "Dolphin Nemo"
            case .deepSeekCoder33: "DeepSeek Coder"
            case .deepSeekCoder7: "DeepSeek Coder (Compact)"
            case .qwenQwQ32: "Qwen QwQ"
            case .qwen25coder: "Qwen 2.5 Coder"
            case .mythoMax: "MythoMax Writing Assistant"
            case .whisper: "Whisper Voice Recognition"
            case .dolphin72b: "Dolphin (Large)"
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
            case .llama3large: "Llama 3.3"
            case .llama3: "Llama 3.1 (Regular)"
            case .llama3compact: "Llama 3.2 (Small)"
            case .llama3tiny: "Llama 3.2 (Compact)"
            case .codestral: "Codestral"
            case .supernovaMedius: "Supernova Medius"
            case .smol: "SmolLM 2"
            case .shuttle: "Shuttle 3"
            case .calme: "Calme 2.4"
            case .athene: "Athene V2"
            case .dsro70: "Deepseek R1 Distill"
            case .dolphinThreeR1: "Dolphin 3 R1"
            case .dolphinThreeTiny: "Dolphin 3 Tiny"
            case .dolphinThree3b: "Dolphin 3 Compact"
            case .dolphinThree8b: "Dolphin 3"
            case .gemma31: "Gemma 3 Tiny"
            case .gemma34: "Gemma 3 Compact"
            case .gemma312: "Gemma 3 Small"
            case .gemma327: "Gemma 3 Regular"
            case .olympicCoder: "Olympic Coder"
            }
        }

        var detail: String {
            switch self {
            case .deepSeekCoder33: "33b variant, on Llama2"
            case .deepSeekCoder7: "v1.5, on Llama2"
            case .mythoMax: "vL2 13b variant"
            case .whisper: "Large v3 Turbo"
            case .qwenQwQ32: "32b params"
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
            case .llama3large: "v3.3, finetuned, 70b params"
            case .llama3: "v3.1, 8b params"
            case .llama3compact: "v3.2, 3b params"
            case .llama3tiny: "v3.2, 1b params"
            case .supernovaMedius: "on LLama 3.1 405b & Qwen 2.5 14b"
            case .codestral: "22b params"
            case .smol: "v2, 1.7b variant"
            case .shuttle: "v2, on Qwen-2.5-72b-Instruct"
            case .calme: "v2.4, on Qwen 2 78b"
            case .athene: "v2, on Qwen 2.5 72b"
            case .dsro70: "R1, on Llama 70b"
            case .dolphin72b: "v2.9.2 on Qwen 2.5 72b"
            case .dolphinNemo: "v2.9.3 on Mistral Nemo 12b"
            case .dolphinCoder: "on StarCoder2 15b"
            case .dolphinThreeTiny: "v3, on Qwen 2.5 1.5b"
            case .dolphinThree3b: "v3, on Qwen 2.5 3b"
            case .dolphinThree8b: "v3, on Llama 3.1 8b"
            case .dolphinThreeR1: "v3, on Mistral 24b & R1 dataset"
            case .gemma31: "v3, 1b variant"
            case .gemma34: "v3, 4b variant"
            case .gemma312: "v3, 12b variant"
            case .gemma327: "v3, 27b params"
            case .olympicCoder: "32b params"
            }
        }

        var id: String {
            switch self {
            case .deepSeekCoder33: "73FD5E35-94F3-4923-9E28-070564DF5B6E"
            case .mythoMax: "AA4B3287-CA79-466F-8F84-87486D701256"
            case .whisper: "0FCCC65B-BD2B-470C-AFE2-637FABDA95EE"
            case .dolphin72b: "26FD3A09-48C6-412C-A9C0-51F17A3E5C9A"
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
            case .llama3large: "23B52C8F-F7BC-4597-95EE-B60B0AB3263E"
            case .llama3: "2F547D7D-612B-4BA0-A42E-B17392346FA0"
            case .llama3compact: "8EBC25F2-8F1D-492E-8A55-9B67AFB3AA89"
            case .llama3tiny: "611A636C-59C0-451C-A435-FD6A9041DB37"
            case .codestral: "303D7134-7861-4167-B465-402DA071C685"
            case .supernovaMedius: "CDCA7E8F-7411-4AEC-A76B-2DB17A62BE3F"
            case .smol: "0767CF26-7090-4B85-A584-2ECAE5499C22"
            case .shuttle: "9044B741-783F-471B-8447-FB773AAEF051"
            case .calme: "5F0BEDAB-59B3-43B8-B4D7-65F9B16A8735"
            case .athene: "87D0BDE7-55EE-4349-9428-18FAACB524EC"
            case .dsro70: "0E1DA288-951A-45AE-841D-4F8F2F451801"
            case .dolphinThreeR1: "E08D44B7-28F5-4270-BFD8-3CEB212AD658"
            case .dolphinThreeTiny: "9B7C2BEB-18AE-4AE7-8979-869EB93E7538"
            case .dolphinThree3b: "FB9A0C29-596F-455E-A7D1-FCD96DB4E10D"
            case .dolphinThree8b: "9DD1221B-3574-4A21-9BD9-F3D4DE54BCE3"
            case .qwenQwQ32: "EFB122A6-E3FE-4EDF-82FF-12AB4236DA95"
            case .gemma31: "B858AFEE-A5AD-4101-9063-C6A559119F1D"
            case .gemma34: "61F39A7E-BF26-4A01-9A9F-0220E77A4BF5"
            case .gemma312: "484668C7-1E34-4E39-927C-29D9BD20690A"
            case .gemma327: "3927111B-0409-4082-91C5-0ABE347285B4"
            case .olympicCoder: "3E4CB40A-1E49-440B-B91C-11B23E6BDCBE"
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
                   repeatCheckPenalty: defaultRepeatCheckPenalty)
        }
    }
}
