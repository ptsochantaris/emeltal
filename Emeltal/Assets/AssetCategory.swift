import Foundation
import Metal

extension Asset {
    enum Category: Identifiable, Codable {
        case dolphinMixtral, deepSeekCoder, mythoMax, sauerkrautSolar, dolphin70b, dolphinPhi2, whisper

        var format: Template.Format {
            switch self {
            case .deepSeekCoder: .alpaca
            case .dolphin70b, .dolphinMixtral, .dolphinPhi2: .chatml
            case .mythoMax: .alpaca
            case .sauerkrautSolar: .userAssistant
            case .whisper: .alpaca
            }
        }

        private var defaultPrompt: String {
            "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don’t know the answer to a question, please don’t share false information."
        }

        var useGpuOnThisSystem: Bool {
            guard let device = MTLCreateSystemDefaultDevice() else {
                fatalError("Failed to get the system's default Metal device.")
            }
            let vramSize = device.recommendedMaxWorkingSetSize / 1_000_000_000
            // log("Checking if current model selection can run on GPU (\(vramSize) GB)")
            return vramSize > vramRequiredToFitInGpu
        }

        var vramRequiredToFitInGpu: Int {
            switch self {
            case .dolphinMixtral: 41
            case .deepSeekCoder: 36
            case .sauerkrautSolar: 12
            case .dolphinPhi2: 5
            case .mythoMax: 16
            case .whisper: 2
            case .dolphin70b: 51
            }
        }

        var sizeDescription: String {
            switch self {
            case .dolphin70b: "48.8 GB"
            case .dolphinMixtral: "32.2 GB"
            case .deepSeekCoder: "27.4 GB"
            case .sauerkrautSolar: "7.6 GB"
            case .mythoMax: "10.7 GB"
            case .whisper: "1.1 GB"
            case .dolphinPhi2: "2.3 GB"
            }
        }

        var aboutText: String {
            switch self {
            case .deepSeekCoder:
                "This no-nonsense model focuses specifically on code-related generation and questions"
            case .dolphinMixtral:
                "The current state of the art, with multifaceted expertise and good conversational ability."
            case .mythoMax:
                "MythoMax is a model designed to be both imaginative, and useful for creativity and writing."
            case .sauerkrautSolar:
                "One of the highest performing models for chat. A great starting point."
            case .dolphin70b:
                "An extra large size version of Dolphin for those with a lot of memory, curiosity and/or patience."
            case .whisper:
                "OpenAI's industry leading speech recognition. Lets you talk directly to the model if you prefer. Ensure you have a good mic and 'voice isolation' is selected from the menubar for best results."
            case .dolphinPhi2:
                "The Doplhin chatbot running on Microsoft's compact Phi-2 model, great for systems with constrained storage or processing requirements."
            }
        }

        var maxBatch: UInt32 {
            switch self {
            case .deepSeekCoder, .dolphin70b, .dolphinMixtral, .dolphinPhi2, .mythoMax, .sauerkrautSolar: 1024
            case .whisper: 0
            }
        }

        private var defaultTopK: Int {
            50
        }

        private var defaultTopP: Float {
            0.5
        }

        private var defaultTemperature: Float {
            0.7
        }

        private var defaultRepeatPenatly: Float {
            switch self {
            case .deepSeekCoder: 1
            case .dolphin70b, .dolphinMixtral, .dolphinPhi2: 1.17
            case .mythoMax: 1.17
            case .sauerkrautSolar: 1.17
            case .whisper: 0
            }
        }

        private var defaultFrequencyPenalty: Float {
            switch self {
            case .deepSeekCoder: 0
            case .dolphin70b, .dolphinMixtral, .dolphinPhi2: 0.1
            case .mythoMax: 0.1
            case .sauerkrautSolar: 0.1
            case .whisper: 0
            }
        }

        private var defaultPresentPenalty: Float {
            1
        }

        var emeltalRepo: URL {
            URL(string: "https://huggingface.co/acertainbru/emeltal-collection")!
        }

        var originalRepoUrl: URL {
            let uri = switch self {
            case .dolphinMixtral:
                "https://huggingface.co/cognitivecomputations/dolphin-2.6-mixtral-8x7b"
            case .deepSeekCoder:
                "https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct"
            case .mythoMax:
                "https://huggingface.co/Gryphe/MythoMax-L2-13b"
            case .whisper:
                "https://huggingface.co/ggerganov/whisper.cpp"
            case .sauerkrautSolar:
                "https://huggingface.co/VAGOsolutions/SauerkrautLM-SOLAR-Instruct"
            case .dolphin70b:
                "https://huggingface.co/cognitivecomputations/dolphin-2.2-70b"
            case .dolphinPhi2:
                "https://huggingface.co/cognitivecomputations/dolphin-2_6-phi-2"
            }
            return URL(string: uri)!
        }

        var fetchUrl: URL {
            let fileName = switch self {
            case .dolphinMixtral:
                "dolphin-2.6-mixtral-8x7b.Q5_K_M.gguf"
            case .deepSeekCoder:
                "deepseek-coder-33b-instruct.Q6_K.gguf"
            case .mythoMax:
                "mythomax-l2-13b.Q6_K.gguf"
            case .whisper:
                "ggml-large-v3-q5_0.bin"
            case .sauerkrautSolar:
                "sauerkrautlm-solar-instruct.Q5_K_M.gguf"
            case .dolphin70b:
                "dolphin-2.2-70b.Q5_K_M.gguf"
            case .dolphinPhi2:
                "dolphin-2_6-phi-2.Q6_K.gguf"
            }
            return emeltalRepo
                .appendingPathComponent("resolve", conformingTo: .directory)
                .appendingPathComponent("main", conformingTo: .directory)
                .appendingPathComponent(fileName)
        }

        var displayName: String {
            switch self {
            case .dolphinMixtral: "Dolphin"
            case .deepSeekCoder: "DeepSeek Coder"
            case .mythoMax: "MythoMax Writing Assistant"
            case .whisper: "Whisper"
            case .sauerkrautSolar: "Sauerkraut"
            case .dolphin70b: "Dolphin (Large)"
            case .dolphinPhi2: "Dolphin (Compact)"
            }
        }

        var detail: String {
            switch self {
            case .dolphinMixtral: "v2.6, on Mixtral 8x7b"
            case .deepSeekCoder: "33b variant, on Llama2"
            case .mythoMax: "vL2 13b variant"
            case .whisper: "Large v3"
            case .sauerkrautSolar: "on Solar 10.7b"
            case .dolphin70b: "on Llama 70b (x2)"
            case .dolphinPhi2: "v2.6, on Phi-2"
            }
        }

        var id: String {
            switch self {
            case .dolphinMixtral: "43588C6F-FB70-4EDB-9C15-3B75E7C483FA"
            case .deepSeekCoder: "73FD5E35-94F3-4923-9E28-070564DF5B6E"
            case .mythoMax: "AA4B3287-CA79-466F-8F84-87486D701256"
            case .whisper: "0FCCC65B-BD2B-470C-AFE2-637FABDA95EE"
            case .sauerkrautSolar: "195B279E-3CAA-4E53-9CD3-59D5DE5B40A2"
            case .dolphin70b: "0D70BC73-9559-4778-90A6-E5F2E4B71213"
            case .dolphinPhi2: "72ACC367-207D-4BCA-83F0-2767827D8F64"
            }
        }

        var defaultParams: Params {
            Params(topK: defaultTopK,
                   topP: defaultTopP,
                   systemPrompt: defaultPrompt,
                   temperature: defaultTemperature,
                   repeatPenatly: defaultRepeatPenatly,
                   frequencyPenatly: defaultFrequencyPenalty,
                   presentPenatly: defaultPresentPenalty)
        }
    }
}
