import Foundation

extension Model {
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
}
