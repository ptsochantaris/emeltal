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
        var version: Int?
    }
}
