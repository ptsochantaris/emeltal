import Foundation
import Metal

extension Asset {
    enum Category: Identifiable, Codable {
        case dolphinMixtral, deepSeekCoder, mythoMax, sauerkrautSolar, shiningValiant, dolphinPhi2, whisper

        var format: Template.Format {
            switch self {
            case .shiningValiant: .llama
            case .deepSeekCoder: .alpaca
            case .dolphinMixtral: .chatml
            case .mythoMax: .alpaca
            case .sauerkrautSolar: .userAssistant
            case .whisper: .alpaca
            case .dolphinPhi2: .chatml
            }
        }

        private var defaultPrompt: String {
            switch self {
            case .deepSeekCoder: "You are an intelligent and helpful coding assistant."
            case .sauerkrautSolar: ""
            case .dolphinMixtral, .dolphinPhi2: "You are Dolphin, a helpful, informative and friendly AI assistant."
            case .mythoMax: "You are an intelligent and helpful writing assistant."
            case .shiningValiant: "You are an intelligent, helpful AI assistant."
            case .whisper: ""
            }
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
            case .dolphinMixtral: 34
            case .deepSeekCoder: 37
            case .sauerkrautSolar: 9
            case .dolphinPhi2: 5
            case .mythoMax: 12
            case .whisper: 2
            case .shiningValiant: 60
            }
        }

        var sizeDescription: String {
            switch self {
            case .shiningValiant: "48.8 GB"
            case .dolphinMixtral: "32.2 GB"
            case .deepSeekCoder: "35.4 GB"
            case .sauerkrautSolar: "7.6 GB"
            case .mythoMax: "10.6 GB"
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
            case .shiningValiant:
                "A large-size model focused on knowledge, enthusiasm, and personality."
            case .whisper:
                "OpenAI's industry leading speech recognition. Lets you talk directly to the model if you prefer. Ensure you have a good mic and 'voice isolation' is selected from the menubar for best results."
            case .dolphinPhi2:
                "The Doplhin chatbot running on Microsoft's compact Phi-2 model, great for systems with constrained storage or processing requirements."
            }
        }

        var maxBatch: UInt32 {
            switch self {
            case .deepSeekCoder, .dolphinMixtral, .dolphinPhi2, .mythoMax, .sauerkrautSolar, .shiningValiant: 1024
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
            case .dolphinMixtral: 1.17
            case .mythoMax: 1.17
            case .sauerkrautSolar: 1.17
            case .shiningValiant: 1.17
            case .dolphinPhi2: 1.17
            case .whisper: 0
            }
        }

        private var defaultFrequencyPenalty: Float {
            switch self {
            case .deepSeekCoder: 0
            case .dolphinMixtral: 0.1
            case .mythoMax: 0.1
            case .sauerkrautSolar: 0.1
            case .shiningValiant: 0.1
            case .dolphinPhi2: 0.1
            case .whisper: 0
            }
        }

        private var defaultPresentPenalty: Float {
            1
        }

        var repoUrl: URL {
            let uri = switch self {
            case .dolphinMixtral:
                "https://huggingface.co/TheBloke/dolphin-2.6-mixtral-8x7b-GGUF"
            case .deepSeekCoder:
                "https://huggingface.co/TheBloke/deepseek-coder-33B-instruct-GGUF"
            case .mythoMax:
                "https://huggingface.co/TheBloke/MythoMax-L2-13B-GGUF"
            case .whisper:
                "https://huggingface.co/ggerganov/whisper.cpp"
            case .sauerkrautSolar:
                "https://huggingface.co/TheBloke/SauerkrautLM-SOLAR-Instruct-GGUF"
            case .shiningValiant:
                "https://huggingface.co/TheBloke/ShiningValiant-1.3-GGUF"
            case .dolphinPhi2:
                "https://huggingface.co/TheBloke/dolphin-2_6-phi-2-GGUF"
            }
            return URL(string: uri)!
        }

        var fetchUrl: URL {
            let fileName = switch self {
            case .dolphinMixtral:
                "dolphin-2.6-mixtral-8x7b.Q5_K_M.gguf"
            case .deepSeekCoder:
                "deepseek-coder-33b-instruct.Q8_0.gguf"
            case .mythoMax:
                "mythomax-l2-13b.Q8_0.gguf"
            case .whisper:
                "ggml-large-v3-q5_0.bin"
            case .sauerkrautSolar:
                "sauerkrautlm-solar-instruct.Q5_K_M.gguf"
            case .shiningValiant:
                "shiningvaliant-1.2.Q5_K_M.gguf"
            case .dolphinPhi2:
                "dolphin-2_6-phi-2.Q6_K.gguf"
            }
            return repoUrl
                .appendingPathComponent("resolve", conformingTo: .directory)
                .appendingPathComponent("main", conformingTo: .directory)
                .appendingPathComponent(fileName)
        }

        var displayName: String {
            switch self {
            case .dolphinMixtral: "Dolphin (Large)"
            case .deepSeekCoder: "DeepSeek Coder"
            case .mythoMax: "MythoMax Writing Assistant"
            case .whisper: "Whisper"
            case .sauerkrautSolar: "Sauerkraut"
            case .shiningValiant: "Shining Valiant"
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
            case .shiningValiant: "quantised v1.4"
            case .dolphinPhi2: "v2.6, on Phi-2"
            }
        }

        var id: String {
            switch self {
            case .dolphinMixtral: "43588C6F-FB70-4EDB-9C15-3B75E7C483FA"
            case .deepSeekCoder: "A4A183C7-718C-4E47-B880-F437DAFA9D5C"
            case .mythoMax: "AA4B3287-CA79-466F-8F84-87486D701256"
            case .whisper: "0FCCC65B-BD2B-470C-AFE2-637FABDA95EE"
            case .sauerkrautSolar: "195B279E-3CAA-4E53-9CD3-59D5DE5B40A2"
            case .shiningValiant: "25C3A6DB-A824-4011-9E8F-330D3B6310C7"
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
