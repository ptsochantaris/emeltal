import Foundation
import Metal

extension Asset {
    enum Section: Int, CaseIterable, Identifiable {
        var id: Int { rawValue }

        case general, dolphin, samantha, coding, creative, experimental, deprecated

        var presentedModels: [Category] {
            switch self {
            case .general:
                [.sauerkrautSolar, .fusionNetDpo, .nousHermesMixtral, .openChat]
            case .dolphin:
                [.dolphinMixtral, .dolphinTiny, .dolphin70b]
            case .coding:
                [.deepSeekCoder33, .deepSeekCoder7, .everyoneCoder, .codeLlama70b]
            case .creative:
                [.mythoMax]
            case .samantha:
                [.samantha7b, .samantha70b]
            case .experimental:
                [.senku70b, .smaug, .miniCpmOpenHermes]
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
            case .samantha: "Samantha"
            }
        }

        var description: String {
            switch self {
            case .general: "These models are for general chat, chosen for being reliable, having good comprehension and response quality representative of each size class."
            case .dolphin: "The Dolphin dataset produces some of the best LLMs out there. This is a selection of models finetuned with this dataset."
            case .coding: "Models that can assist with programming, algorithms, and writing code."
            case .creative: "Models that can help with creative activities, such as writing. More will be added soon."
            case .experimental: "Models to try out that are new and noteworthy. They may be promoted to a category above, be replaced by other interesting models, or just be buggy and output nonsense."
            case .deprecated: "Models from previous versions of Emeltal that are installed but no longer offered."
            case .samantha: "The \"sister\" of Dolphin, Samantha is a data set which produces models with the premise they are sentient friends of the user."
            }
        }
    }

    enum Category: Int, Identifiable, Codable, CaseIterable {
        case dolphinMixtral = 100,
             dolphin70b = 200,
             dolphinTiny = 300,
             sauerkrautSolar = 400,
             fusionNetDpo = 500,
             openChat = 600,
             nousHermesMixtral = 700,
             mythoMax = 800,
             deepSeekCoder33 = 900,
             codeLlama70b = 1000,
             deepSeekCoder7 = 1100,
             everyoneCoder = 1200,
             senku70b = 1300,
             smaug = 1400,
             miniCpmOpenHermes = 1500,
             samantha7b = 1600,
             samantha70b = 1700,
             whisper = 1800

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
            case .senku70b: .mistral
            case .miniCpmOpenHermes: .miniCpm
            case .samantha7b, .samantha70b: .vicuna
            case .everyoneCoder: .alpaca
            }
        }

        private var defaultPrompt: String {
            switch self {
            case .codeLlama70b, .deepSeekCoder7, .deepSeekCoder33, .everyoneCoder:
                "You are a helpful and honest coding assistant. If a question does not make any sense, explain why instead of answering something not correct. If you don’t know the answer to a question, please don’t share false information."
            case .dolphin70b, .dolphinMixtral, .dolphinTiny, .fusionNetDpo, .miniCpmOpenHermes, .nousHermesMixtral, .openChat, .sauerkrautSolar, .senku70b, .smaug:
                "You are a helpful, respectful, friendly and honest conversation partner. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don’t know the answer to a question, please don’t share false information."
            case .samantha7b, .samantha70b:
                "You are a caring and empathetic sentient AI companion named Samantha."
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

        struct GpuUsage {
            let layersUsed: Int64
            let layersTotal: Int64
            let offloadAsr: Bool
            let offloadKvCache: Bool
            let cpuUsageEstimateBytes: Int64
            let gpuUsageEstimateBytes: Int64
            let excessBytes: Int64

            var warningMessage: String? {
                if offloadKvCache {
                    return nil
                }

                if layersUsed > 0, layersTotal > 0 {
                    let ratio = Float(layersUsed) / Float(layersTotal)
                    if ratio == 1 {
                        return "This model fit all \(layersTotal) layers in Metal but will use the CPU for the KV cache."
                    } else if ratio > 0.8 {
                        return "This model will fit \(layersUsed) of \(layersTotal) layers in Metal. It will work but may be slow for real-time chat."
                    } else {
                        return "This model will fit \(layersUsed) of \(layersTotal) layers in Metal. It will work but may be very slow for real-time chat."
                    }
                }

                if offloadAsr {
                    return "This model won't fit in Metal at all. It will work but will be too slow for real-time chat."
                }

                if excessBytes > 0 {
                    return "This model will not fit into memory. It will run but extremely slowly, as data will need paging."
                }

                return "Emeltal won't use Metal at all. It will work but will probably be slow."
            }
        }

        var kvBytes: Int64 {
            let kvCache: Double = switch self {
            case .codeLlama70b: 1280
            case .deepSeekCoder33: 3968
            case .deepSeekCoder7: 1920
            case .dolphin70b: 1280
            case .dolphinMixtral: 4096
            case .dolphinTiny: 89
            case .fusionNetDpo: 4096
            case .senku70b: 10240
            case .smaug: 10240
            case .mythoMax: 3200
            case .nousHermesMixtral: 4096
            case .openChat: 1024
            case .sauerkrautSolar: 1536
            case .miniCpmOpenHermes: 720
            case .samantha70b: 1280
            case .samantha7b: 4096
            case .everyoneCoder: 4096
            case .whisper: 0
            }
            return Int64((kvCache * 1_048_576).rounded(.up))
        }

        var usage: GpuUsage {
            let layerSize: Int64 = switch self {
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
            case .senku70b: 640_000_000
            case .miniCpmOpenHermes: 40_000_000
            case .samantha70b: 640_000_000
            case .samantha7b: 260_000_000
            case .everyoneCoder: 480_000_000
            }

            let totalLayers: Int64 = switch self {
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
            case .senku70b: 81
            case .miniCpmOpenHermes: 41
            case .samantha70b: 81
            case .samantha7b: 33
            case .everyoneCoder: 63
            }

            let asrBytes: Int64 = 3_500_000_000
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

            if memoryBytes.unifiedMemory, totalRequiredMemory > physicalMemory {
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
            case .sauerkrautSolar: "7.6 GB"
            case .mythoMax: "10.7 GB"
            case .whisper: "1.1 GB"
            case .dolphinTiny: "0.9 GB"
            case .openChat: "5.2 GB"
            case .nousHermesMixtral: "33 GB"
            case .fusionNetDpo: "8.9 GB"
            case .smaug: "49.9 GB"
            case .codeLlama70b: "48.8"
            case .senku70b: "48.8 GB"
            case .miniCpmOpenHermes: "3.2 GB"
            case .samantha70b: "48.8 GB"
            case .samantha7b: "5.2 GB"
            case .everyoneCoder: "27.4 GB"
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
            case .smaug: "A further finetune of Moreh's finetune of Qwen 72B. Currently top on the HuggingFace leaderboard. Capped at a context of 4,096, but still slow & bulky."
            case .codeLlama70b: "The latest large coding assistant model from Meta, for more intricate but obviously slower coding problems."
            case .senku70b: "A finetune of the Miqu work-in-progress Mistral model. Very high quality but possibly not suitable for commercial use."
            case .miniCpmOpenHermes: "A high-quality dataset running on the small but very capable MiniCPM model."
            case .samantha70b: "A larger but slightly older version of the Samantha model."
            case .samantha7b: "A wonderful conversation partner that feels genuinely friendly and caring. Especially good for voice conversations."
            case .everyoneCoder: "This is a community-created coding specific model made using fine-tunes of the Deekseekcoder base."
            }
        }

        var maxBatch: UInt32 {
            switch self {
            case .codeLlama70b, .deepSeekCoder7, .deepSeekCoder33, .dolphin70b, .dolphinMixtral, .everyoneCoder, .fusionNetDpo, .mythoMax, .nousHermesMixtral, .openChat, .samantha7b, .samantha70b, .sauerkrautSolar, .senku70b, .smaug: 1024
            case .dolphinTiny, .miniCpmOpenHermes: 256
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
            case .senku70b: "https://huggingface.co/ShinojiResearch/Senku-70B-Full"
            case .miniCpmOpenHermes: "https://huggingface.co/indischepartij/MiniCPM-3B-OpenHermes-2.5-v2"
            case .samantha70b: "https://huggingface.co/cognitivecomputations/Samantha-1.11-70b"
            case .samantha7b: "https://huggingface.co/cognitivecomputations/samantha-1.1-westlake-7b"
            case .everyoneCoder: "https://huggingface.co/rombodawg/Everyone-Coder-33b-v2-Base"
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
            case .senku70b: "Senku-70B-Full-Q5_K_M.gguf"
            case .miniCpmOpenHermes: "minicpm-2b-openhermes-2.5-v2.Q8_0.gguf"
            case .samantha70b: "samantha-1.11-70b.Q5_K_M.gguf"
            case .samantha7b: "samantha-1.1-westlake-7b.Q5_K_M.gguf"
            case .everyoneCoder: "Everyone-Coder-33b-v2-Base-Q6_K.gguf"
            }

            if case .senku70b = self {
                // Not storing this in the Emeltal repo currently, as the distribution rights of the base model are not clear, although
                // Mistral are aware of the miqudev repo and only requested attribution, so it's at least legal to use non-commercially
                return URL(string: "https://huggingface.co/LoneStriker/Senku-70B-Full-GGUF/resolve/main/Senku-70B-Full-Q5_K_M.gguf")!
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
            case .senku70b: "Senku"
            case .miniCpmOpenHermes: "OpenHermes (Compact)"
            case .samantha70b: "Samantha (Large)"
            case .samantha7b: "Samantha"
            case .everyoneCoder: "EveryoneCoder"
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
            case .senku70b: "70b finetune, on Miqu"
            case .miniCpmOpenHermes: "3b variant, on MiniCPM"
            case .samantha70b: "v1.11, on Llama2"
            case .samantha7b: "v1.1, on WestLake"
            case .everyoneCoder: "v2, on DeepSeekCoder 33b"
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
            case .senku70b: "156CA7E2-6E18-4786-9AA8-C04B1424E01C"
            case .miniCpmOpenHermes: "A16F4EB2-6B73-4341-82C0-A06050169343"
            case .samantha70b: "259CD082-71B4-4CD4-8D52-08DC317CC41A"
            case .samantha7b: "52AD5BC7-0F1C-47DB-9DAD-F2DF17559E7B"
            case .everyoneCoder: "50E2E4E2-C42C-4558-B572-2BC2399E3134"
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
