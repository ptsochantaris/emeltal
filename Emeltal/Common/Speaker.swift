#if canImport(AppKit)
    import AppKit
#endif
import AVFoundation
import Foundation

final actor Speaker {
    private let synth = AVSpeechSynthesizer()

    var havePreferredVoice = false

    private let voice: AVSpeechSynthesisVoice

    private static func pickFavourite(from voices: [AVSpeechSynthesisVoice]) -> AVSpeechSynthesisVoice? {
        if let premiumFemale = voices.filter({ $0.quality == .premium && $0.gender == .female }).first {
            return premiumFemale
        }
        if let premiumMale = voices.filter({ $0.quality == .premium && $0.gender == .male }).first {
            return premiumMale
        }
        if let enhancedFemale = voices.filter({ $0.quality == .enhanced && $0.gender == .female }).first {
            return enhancedFemale
        }
        if let enhancedMale = voices.filter({ $0.quality == .enhanced && $0.gender == .male }).first {
            return enhancedMale
        }
        if let female = voices.first(where: { $0.gender == .female }) {
            return female
        }
        return voices.first
    }

    init() async throws {
        if let preferred = AVSpeechSynthesisVoice(identifier: "com.apple.voice.premium.en-US.Zoe") {
            havePreferredVoice = true
            voice = preferred
        } else {
            let allVoices = AVSpeechSynthesisVoice.speechVoices()
            let enVoices = allVoices.filter { $0.language.hasPrefix("en-US") && !$0.voiceTraits.contains(.isNoveltyVoice) && !$0.voiceTraits.contains(.isPersonalVoice) }
            if let enVoice = Self.pickFavourite(from: enVoices) {
                log("Selected voice: \(enVoice.identifier)")
                voice = enVoice
            } else if let anyVoice = Self.pickFavourite(from: allVoices) {
                log("Fallback voice: \(anyVoice.identifier)")
                voice = anyVoice
            } else {
                throw "Could not find any TTS voices in the system"
            }
        }

        add(text: "")
    }

    func cancelIfNeeded() {
        synth.stopSpeaking(at: .immediate)
    }

    func waitForCompletion() async {
        while synth.isSpeaking {
            try? await Task.sleep(for: .seconds(0.1))
        }
    }

    func add(text: String) {
        let textToPlay = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let utterance = AVSpeechUtterance(string: textToPlay)
        utterance.voice = voice
        if textToPlay.hasSuffix(".") || textToPlay.hasSuffix("!") || textToPlay.hasSuffix("?") {
            utterance.postUtteranceDelay = 0.2
        }
        synth.speak(utterance)
    }

    #if canImport(AppKit)
        let startEffect = NSSound(contentsOf: Bundle.main.url(forResource: "MicStart", withExtension: "caf")!, byReference: true)!
        let endEffect = NSSound(contentsOf: Bundle.main.url(forResource: "MicStop", withExtension: "caf")!, byReference: true)!

        func playEffect(_ sound: NSSound) {
            sound.currentTime = 0
            sound.volume = 0.2
            sound.play()
            DispatchQueue.main.asyncAfter(deadline: .now() + sound.duration + 0.1) {
                sound.stop()
            }
        }
    #endif
}
