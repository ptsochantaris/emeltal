import Foundation

extension Asset {
    struct Params: Codable {
        var topK: Int
        var topP: Float
        var systemPrompt: String
        var temperature: Float
        var repeatPenatly: Float
        var frequencyPenatly: Float
        var presentPenatly: Float
    }
}
