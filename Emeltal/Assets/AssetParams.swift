import Foundation

extension Asset {
    struct Params: Codable {
        var topK: Int
        var topP: Float
        var systemPrompt: String
        var temperature: Float
        var temperatureRange: Float
        var temperatureExponent: Float
        var repeatPenatly: Float
        var frequencyPenatly: Float
        var presentPenatly: Float

        enum SamplingType {
            case topK, topP, temperature, entropy, greedy
        }

        var samplingType: SamplingType {
            if temperature > 0 {
                if temperatureRange > 0 {
                    .entropy
                } else {
                    .temperature
                }
            } else if topP < 1 {
                .topP
            } else if topK < 100 {
                .topK
            } else {
                .greedy
            }
        }
    }
}
