import Foundation
import Metal

extension Asset {
    enum Section: Int, CaseIterable, Identifiable {
        var id: Int { rawValue }

        case general, dolphin, coding, creative, experimental, deprecated

        var presentedModels: [Category] {
            switch self {
            case .general:
                [.sauerkrautSolar, .fusionNetDpo, .nousHermesMixtral, .openChat]
            case .dolphin:
                [.dolphinMixtral, .dolphinTiny, .dolphin70b]
            case .coding:
                [.deepSeekCoder33, .deepSeekCoder7, .codeLlama70b]
            case .creative:
                [.mythoMax]
            case .experimental:
                [.smaug, .miqu, .internlm2]
            case .deprecated:
                []
            }
        }

        var title: String {
            switch self {
            case .general: "General Chat"
            case .dolphin: "Dolphin"
            case .coding: "Coding"
            case .creative: "Creative"
            case .experimental: "Experimental"
            case .deprecated: "Deprecated"
            }
        }

        var description: String {
            switch self {
            case .general: "These models are for general chat, chosen for being reliable, having good comprehension and response quality representative of each size class."
            case .dolphin: "The Dolphin dataset produces some of the best LLMs out there. This is a selection of models finetuned with this dataset."
            case .coding: "Models that can assist with programming, algorithms, and writing code."
            case .creative: "Models that can help with creative activities, such as writing. More will be added soon."
            case .experimental: "Models that are less about being useful and more about being noteworthy for some reason."
            case .deprecated: "Models from previous versions of Emeltal that are installed but no longer offered."
            }
        }
    }

    enum Category: Identifiable, Codable, CaseIterable {
        case dolphinMixtral, deepSeekCoder33, deepSeekCoder7, mythoMax, sauerkrautSolar, dolphin70b, dolphinTiny, openChat, whisper, nousHermesMixtral, fusionNetDpo, smaug, codeLlama70b, miqu, internlm2

        var selectable: Bool {
            switch self {
            case .whisper: false
            default: true
            }
        }

        var section: Section? {
            Section.allCases.first { $0.presentedModels.contains(self) }
        }

        var format: Template.Format {
            switch self {
            case .deepSeekCoder7, .deepSeekCoder33: .alpaca
            case .dolphin70b, .dolphinMixtral, .dolphinTiny: .chatml
            case .mythoMax: .alpaca
            case .sauerkrautSolar: .userAssistant
            case .whisper: .alpaca
            case .openChat: .openChat
            case .nousHermesMixtral: .chatml
            case .fusionNetDpo: .alpaca
            case .smaug: .chatml
            case .codeLlama70b: .llamaLarge
            case .miqu: .mistral
            case .internlm2: .chatml
            }
        }

        private var defaultPrompt: String {
            switch self {
            case .codeLlama70b, .deepSeekCoder7, .deepSeekCoder33:
                "You are a helpful and honest coding assistant. If a question does not make any sense, explain why instead of answering something not correct. If you don’t know the answer to a question, please don’t share false information."
            case .dolphin70b, .dolphinMixtral, .dolphinTiny, .fusionNetDpo, .miqu, .nousHermesMixtral, .openChat, .sauerkrautSolar, .smaug:
                "You are a helpful, respectful, friendly and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don’t know the answer to a question, please don’t share false information."
            case .internlm2:
                "You are an AI assistant called InternLM"
            case .mythoMax:
                "You are a helpful, imaginative, collaborative, and friendly writing assistant."
            case .whisper:
                ""
            }
        }

        var contextSize: UInt32 {
            switch self {
            case .smaug: 4096
            default: 0
            }
        }

        enum GpuUsage {
            case none, low(Int, Int), partial(Int, Int), full(Int, Bool)

            var involvesGpu: Bool {
                if case .none = self {
                    return false
                }
                return true
            }

            var isFull: Bool {
                if case .full = self {
                    return true
                }
                return false
            }

            var offloadKvCache: Bool {
                switch self {
                case .low, .none, .partial: false
                case let .full(_, offload): offload
                }
            }

            var usedLayers: Int {
                switch self {
                case .none: 0
                case let .full(usedLayers, _), let .low(usedLayers, _), let .partial(usedLayers, _): usedLayers
                }
            }
        }

        var kvBytes: Int {
            let kvCache: Double = switch self {
            case .codeLlama70b: 1280
            case .deepSeekCoder33: 3968
            case .deepSeekCoder7: 1920
            case .dolphin70b: 1280
            case .dolphinMixtral: 4096
            case .dolphinTiny: 89
            case .fusionNetDpo: 4096
            case .miqu: 10238.75
            case .smaug: 10240
            case .mythoMax: 3200
            case .nousHermesMixtral: 4096
            case .openChat: 1024
            case .sauerkrautSolar: 1536
            case .internlm2: 6144
            case .whisper: 0
            }
            return Int((kvCache * 1_048_576).rounded(.up))
        }

        var usage: GpuUsage {
            guard let vramBytes else { return .none }

            let layerSize = switch self {
            case .dolphinMixtral: 1_350_000_000
            case .deepSeekCoder33: 600_000_000
            case .deepSeekCoder7: 330_000_000
            case .sauerkrautSolar: 260_000_000
            case .dolphinTiny: 48_000_000
            case .mythoMax: 410_000_000
            case .whisper: 1
            case .dolphin70b: 608_000_000
            case .openChat: 290_000_000
            case .nousHermesMixtral: 1_350_000_000
            case .fusionNetDpo: 610_000_000
            case .smaug: 640_000_000
            case .codeLlama70b: 640_000_000
            case .miqu: 760_000_000
            case .internlm2: 440_000_000
            }

            let totalLayers = switch self {
            case .dolphinMixtral: 33
            case .deepSeekCoder33: 63
            case .deepSeekCoder7: 31
            case .sauerkrautSolar: 49
            case .dolphinTiny: 23
            case .mythoMax: 41
            case .whisper: 0
            case .dolphin70b: 81
            case .openChat: 33
            case .nousHermesMixtral: 33
            case .fusionNetDpo: 33
            case .smaug: 81
            case .codeLlama70b: 81
            case .miqu: 81
            case .internlm2: 49
            }

            let asrBytes = 2_000_000_000

            if asrBytes > vramBytes.max {
                log("Will not use GPU at all, as space is not enough to hold the base overhead")
                return .none
            }

            let maxRecommendedVram = Int(vramBytes.max)
            let availableGpuForLayers = maxRecommendedVram - asrBytes

            let layers = Float(availableGpuForLayers) / Float(layerSize)
            let layersThatCanFit = Int(layers.rounded(.down))
            let layersToFit = min(layersThatCanFit, totalLayers)
            let usedGpuForLayers = layersToFit * Int(layerSize)
            let expected = usedGpuForLayers + asrBytes
            let totalMemoryUsed = expected + kvBytes

            if vramBytes.unifiedMemory, totalMemoryUsed > ProcessInfo.processInfo.physicalMemory {
                log("Will not use GPU at all, as total memory use is larger than the whole system's memory")
                return .none
            }

            if layersToFit >= totalLayers {
                let offLoad = maxRecommendedVram - expected > kvBytes
                if offLoad {
                    log("Estimating GPU use to be \(sizeFormatter.string(fromByteCount: Int64(totalMemoryUsed))) / \(sizeFormatter.string(fromByteCount: Int64(vramBytes.max))) using \(layersToFit) / \(totalLayers) layers, KV cache: \(sizeFormatter.string(fromByteCount: Int64(kvBytes)))")
                } else {
                    log("Estimating GPU use to be \(sizeFormatter.string(fromByteCount: Int64(expected))) / \(sizeFormatter.string(fromByteCount: Int64(vramBytes.max))) using \(layersToFit) / \(totalLayers) layers")
                }
                return .full(totalLayers, offLoad)
            }

            if layersToFit == 0 {
                log("Will not use GPU at all, as space is not enough to hold any layers")
                return .none
            }

            let ratio = Float(layersToFit) / Float(totalLayers)
            if ratio < 0.5 {
                log("Estimating GPU use to be \(sizeFormatter.string(fromByteCount: Int64(expected))) / \(sizeFormatter.string(fromByteCount: Int64(vramBytes.max))) using \(layersToFit) / \(totalLayers) layers")
                return .low(layersToFit, totalLayers)
            } else {
                log("Estimating GPU use to be \(sizeFormatter.string(fromByteCount: Int64(expected))) / \(sizeFormatter.string(fromByteCount: Int64(vramBytes.max))) using \(layersToFit) / \(totalLayers) layers")
                return .partial(layersToFit, totalLayers)
            }
        }

        var vram: (used: String, max: String)? {
            guard let vramBytes else { return nil }
            return (sizeFormatter.string(fromByteCount: vramBytes.used),
                    sizeFormatter.string(fromByteCount: vramBytes.max))
        }

        var vramBytes: (unifiedMemory: Bool, used: Int64, max: Int64)? {
            guard let currentDevice = MTLCreateSystemDefaultDevice() else {
                log("Failed to get the system's default Metal device.")
                return nil
            }
            return (currentDevice.hasUnifiedMemory,
                    Int64(currentDevice.currentAllocatedSize),
                    Int64(currentDevice.recommendedMaxWorkingSetSize))
        }

        var eosOverrides: Set<Int32>? {
            switch self {
            case .codeLlama70b: [32015]
            case .internlm2: [243, 92542]
            default: nil
            }
        }

        var sizeDescription: String {
            switch self {
            case .dolphin70b: "48.8 GB"
            case .dolphinMixtral: "32.2 GB"
            case .deepSeekCoder33: "27.4 GB"
            case .deepSeekCoder7: "5.67 GB"
            case .sauerkrautSolar: "7.6 GB"
            case .mythoMax: "10.7 GB"
            case .whisper: "1.1 GB"
            case .dolphinTiny: "0.9 GB"
            case .openChat: "5.2 GB"
            case .nousHermesMixtral: "33 GB"
            case .fusionNetDpo: "8.9 GB"
            case .smaug: "49.9 GB"
            case .codeLlama70b: "48.8"
            case .miqu: "48.8 GB"
            case .internlm2: "14.1 GB"
            }
        }

        var aboutText: String {
            switch self {
            case .deepSeekCoder33: "This no-nonsense model focuses specifically on code-related generation and questions"
            case .deepSeekCoder7: "A more compact version of the Deepseek Coder model, focusing on code-related generation and questions"
            case .dolphinMixtral: "The current state of the art, with multifaceted expertise and good conversational ability."
            case .mythoMax: "MythoMax is a model designed to be both imaginative, and useful for creativity and writing."
            case .sauerkrautSolar: "One of the highest performing models for chat. A great starting point."
            case .dolphin70b: "An extra large size version of Dolphin for those with a lot of memory, curiosity and/or patience."
            case .whisper: "OpenAI's industry leading speech recognition. Lets you talk directly to the model if you prefer. Ensure you have a good mic and 'voice isolation' is selected from the menubar for best results."
            case .dolphinTiny: "The Doplhin chatbot running on the Tinyllama model, great for systems with constrained storage or processing requirements."
            case .openChat: "One of the highest performing models at the medium-small size range."
            case .nousHermesMixtral: "The Nous Hermes chatbot running on the Mixtral state of the art model."
            case .fusionNetDpo: "Excellent experimental model with the current top sentence completion performance."
            case .smaug: "A further finetune of Moreh's finetune of Qwen 72B. Currently top on the HuggingFace leaderboard. Capped at a context of 4096, but still slow & bulky."
            case .codeLlama70b: "The latest large coding assistant model from Meta, for more intricate but obviously slower coding problems."
            case .miqu: "A work-in-progress version of the Mistral Medium model. Very high quality but most probably not suitable for any commercial use."
            case .internlm2: "An experimental mix of the v2 InternLM models. High performance and very good at chat, but may be buggy."
            }
        }

        var maxBatch: UInt32 {
            switch self {
            case .codeLlama70b, .deepSeekCoder7, .deepSeekCoder33, .dolphin70b, .dolphinMixtral, .fusionNetDpo, .internlm2, .miqu, .mythoMax, .nousHermesMixtral, .openChat, .sauerkrautSolar, .smaug: 1024
            case .dolphinTiny: 256
            case .whisper: 0
            }
        }

        private var defaultTopK: Int {
            switch self {
            case .codeLlama70b:
                10
            default:
                50
            }
        }

        private var defaultTopP: Float {
            switch self {
            case .codeLlama70b:
                0.97
            default:
                0.5
            }
        }

        private var defaultTemperature: Float {
            switch self {
            case .deepSeekCoder7, .deepSeekCoder33:
                0
            case .codeLlama70b:
                0.8
            case .internlm2:
                0.4
            default:
                0.7
            }
        }

        private var defaultTemperatureRange: Float {
            switch self {
            case .codeLlama70b, .deepSeekCoder7, .deepSeekCoder33:
                0
            default:
                0.2
            }
        }

        private var defaultTemperatureExponent: Float {
            1.0
        }

        private var defaultRepeatPenatly: Float {
            switch self {
            case .deepSeekCoder7, .deepSeekCoder33:
                1.0
            case .codeLlama70b:
                1.1
            default:
                1.17
            }
        }

        private var defaultFrequencyPenalty: Float {
            switch self {
            case .deepSeekCoder7, .deepSeekCoder33:
                0
            case .codeLlama70b:
                0
            default:
                0.1
            }
        }

        private var defaultPresentPenalty: Float {
            switch self {
            case .codeLlama70b:
                0
            default:
                1
            }
        }

        var emeltalRepo: URL {
            URL(string: "https://huggingface.co/acertainbru/emeltal-collection")!
        }

        var originalRepoUrl: URL {
            let uri = switch self {
            case .dolphinMixtral: "https://huggingface.co/cognitivecomputations/dolphin-2.7-mixtral-8x7b"
            case .deepSeekCoder33: "https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct"
            case .deepSeekCoder7: "https://huggingface.co/deepseek-ai/deepseek-coder-7b-instruct-v1.5"
            case .mythoMax: "https://huggingface.co/Gryphe/MythoMax-L2-13b"
            case .whisper: "https://huggingface.co/ggerganov/whisper.cpp"
            case .sauerkrautSolar: "https://huggingface.co/VAGOsolutions/SauerkrautLM-SOLAR-Instruct"
            case .dolphin70b: "https://huggingface.co/cognitivecomputations/dolphin-2.2-70b"
            case .dolphinTiny: "https://huggingface.co/cognitivecomputations/TinyDolphin-2.8-1.1b"
            case .openChat: "https://huggingface.co/openchat/openchat-3.5-0106"
            case .nousHermesMixtral: "https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
            case .fusionNetDpo: "https://huggingface.co/yunconglong/Truthful_DPO_TomGrc_FusionNet_7Bx2_MoE_13B"
            case .smaug: "https://huggingface.co/abacusai/Smaug-72B-v0.1"
            case .codeLlama70b: "https://huggingface.co/codellama/CodeLlama-70b-Instruct-hf"
            case .miqu: "https://huggingface.co/miqudev/miqu-1-70b"
            case .internlm2: "https://huggingface.co/intervitens/internlm2-limarp-chat-20b-GGUF"
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
            case .sauerkrautSolar: "sauerkrautlm-solar-instruct.Q5_K_M.gguf"
            case .dolphin70b: "dolphin-2.2-70b.Q5_K_M.gguf"
            case .dolphinTiny: "tinydolphin-2.8-1.1b.Q6_K.gguf"
            case .openChat: "openchat-3.5-0106.Q5_K_M.gguf"
            case .nousHermesMixtral: "nous-hermes-2-mixtral-8x7b-dpo.Q5_K_M.gguf"
            case .fusionNetDpo: "Truthful_DPO_TomGrc_FusionNet_7Bx2_MoE_13B-q5_k_m.gguf"
            case .smaug: "Smaug-72B-v0.1-q5_k_s.gguf"
            case .codeLlama70b: "codellama-70b-instruct.Q5_K_M.gguf"
            case .miqu: "miqu-1-70b.q5_K_M.gguf"
            case .internlm2: "internlm2-limarp-chat-20b.Q5_K_M_imx.gguf"
            }

            if case .miqu = self {
                // Not storing this in the Emeltal repo currently, as the distribution rights of the model are not clear, although
                // Mistral are aware of the miqudev repo and only requested attribution, so it's at least legal to use non-commercially
                return URL(string: "https://huggingface.co/miqudev/miqu-1-70b/blob/main/miqu-1-70b.q5_K_M.gguf")!
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
            case .sauerkrautSolar: "Sauerkraut"
            case .dolphin70b: "Dolphin (Large)"
            case .dolphinTiny: "Dolphin (Compact)"
            case .openChat: "OpenChat"
            case .nousHermesMixtral: "Nous Hermes"
            case .fusionNetDpo: "FusionNet"
            case .smaug: "Smaug"
            case .codeLlama70b: "CodeLlama (Large)"
            case .miqu: "Miqu"
            case .internlm2: "InternLM2"
            }
        }

        var detail: String {
            switch self {
            case .dolphinMixtral: "v2.7, on Mixtral 8x7b"
            case .deepSeekCoder33: "33b variant, on Llama2"
            case .deepSeekCoder7: "v1.5, on Llama2"
            case .mythoMax: "vL2 13b variant"
            case .whisper: "Large v3"
            case .sauerkrautSolar: "on Solar 10.7b"
            case .dolphin70b: "on Llama 70b (x2)"
            case .dolphinTiny: "v2.8, on TinyLlama"
            case .openChat: "v3.5(0106)"
            case .nousHermesMixtral: "v2, on Mixtral 8x7b"
            case .fusionNetDpo: "DPO finetune"
            case .smaug: "on Momo 72b, based on Qwen 72b"
            case .codeLlama70b: "70b variant, on Llama2"
            case .miqu: "70b Mistral"
            case .internlm2: "Limarp Chat 20b"
            }
        }

        var id: String {
            switch self {
            case .dolphinMixtral: "43678C6F-FB70-4EDB-9C15-3B75E7C483FA"
            case .deepSeekCoder33: "73FD5E35-94F3-4923-9E28-070564DF5B6E"
            case .mythoMax: "AA4B3287-CA79-466F-8F84-87486D701256"
            case .whisper: "0FCCC65B-BD2B-470C-AFE2-637FABDA95EE"
            case .sauerkrautSolar: "195B279E-3CAA-4E53-9CD3-59D5DE5B40A2"
            case .dolphin70b: "0D70BC73-9559-4778-90A6-E5F2E4B71213"
            case .dolphinTiny: "5CDE7417-9281-4186-9C53-921674E8DCC0"
            case .openChat: "983CD5E9-F843-4D76-8D7B-2FB5A40841BE"
            case .nousHermesMixtral: "DA3F2AB9-963B-44CD-B3D4-CABDCB8C3145"
            case .fusionNetDpo: "2859B29B-19E1-47DE-817F-6A62A79AF7CF"
            case .smaug: "3D29AB99-02EA-44BD-881A-81C838BBBC66"
            case .deepSeekCoder7: "57A70BFB-4005-4B53-9404-3A2B107A6677"
            case .codeLlama70b: "41B93F86-721B-4560-A398-A6E69BFCA99B"
            case .miqu: "656CA7E2-6E18-4786-9AA8-C04B1424E01C"
            case .internlm2: "289A5C9F-4046-4C21-9EA3-D29DCAFA83CD"
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
                   version: 2)
        }
    }
}
