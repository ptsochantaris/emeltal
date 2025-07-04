import Foundation

extension Model {
    enum Category: Int, CaseIterable, Identifiable {
        var id: Int { rawValue }

        case qwen, dolphin, samantha, coding, creative, gemma, llamas, glm, system, experimental, apple

        var title: String {
            switch self {
            case .dolphin: "Dolphin"
            case .coding: "Coding"
            case .qwen: "Qwen"
            case .creative: "Creative"
            case .samantha: "Samantha"
            case .llamas: "Llamas"
            case .system: "Internal"
            case .gemma: "Gemma"
            case .glm: "GLM"
            case .apple: "Apple"
            case .experimental: "Experimental"
            }
        }

        var displayable: Bool {
            switch self {
            case .apple, .coding, .creative, .dolphin, .experimental, .gemma, .glm, .llamas, .qwen, .samantha: true
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
            case .gemma:
                "Google claims Gemma 3 is the most natural chatbot in its size category."
            case .apple:
                "Models published by Apple, focusing on maximising performance at each size category."
            case .glm:
                "ChatGLM is made by the THUDM group at Tsinghua University and is quite good at logic and coding tasks."
            case .system:
                ""
            }
        }
    }
}
