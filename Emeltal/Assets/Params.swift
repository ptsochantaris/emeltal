import Foundation

extension Model {
    struct Params: Codable, Sendable {
        static let currentVersion = 4

        enum Descriptors {
            struct Descriptor {
                let title: String
                let min: Float
                let max: Float
                let disabled: Float
            }

            static let topK = Descriptor(title: "Top-K", min: 0, max: 200, disabled: 0)
            static let topP = Descriptor(title: "Top-P", min: 0, max: 2, disabled: 0)
            static let temperature = Descriptor(title: "Temperature", min: 0, max: 2, disabled: 0)
            static let temperatureRange = Descriptor(title: "Range", min: 0, max: 1, disabled: 0)
            static let temperatureExponent = Descriptor(title: "Exponent", min: 1, max: 4, disabled: 0)
            static let repeatPenatly = Descriptor(title: "Repeat Penalty", min: 1, max: 4, disabled: 1)
            static let frequencyPenatly = Descriptor(title: "Frequency Penalty", min: 0, max: 4, disabled: 0)
            static let presentPenatly = Descriptor(title: "Presence Penalty", min: 1, max: 4, disabled: 1)
        }

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

        static var empty: Params {
            Params(topK: 0,
                   topP: 0,
                   systemPrompt: "",
                   temperature: 0,
                   temperatureRange: 0,
                   temperatureExponent: 0,
                   repeatPenatly: 0,
                   frequencyPenatly: 0,
                   presentPenatly: 0)
        }
    }
}
