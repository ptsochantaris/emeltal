#if canImport(AppKit)
    import AppKit
#endif
import AVFoundation
import Foundation

final actor Speaker {
    var havePreferredVoice = false

    private let voice: AVSpeechSynthesisVoice
    private let synth = AVSpeechSynthesizer()
    private var muted = false

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

    func setMute(_ mute: Bool) {
        muted = mute
        if mute {
            cancelIfNeeded()
        }
    }

    init() throws {
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

        Task {
            await add(text: "")
            #if os(iOS)
                // needs warmup
                await playEffect(.startListening)
            #endif
        }
    }

    func cancelIfNeeded() {
        synth.stopSpeaking(at: .immediate)
    }

    func waitForCompletion() async {
        var count = 0
        while count < 4 {
            try? await Task.sleep(for: .seconds(0.1))
            if synth.isSpeaking {
                count = 0
            } else {
                count += 1
            }
        }
        log("Speaker stopped speaking")
    }

    func add(text: String) {
        if muted { return }
        let textToPlay = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let utterance = AVSpeechUtterance(string: textToPlay)
        utterance.voice = voice
        if textToPlay.hasSuffix(".") || textToPlay.hasSuffix("!") || textToPlay.hasSuffix("?") {
            utterance.postUtteranceDelay = 0.2
        }
        synth.speak(utterance)
    }

    enum Effect {
        case startListening, endListening
    }

    private static let startCaf = Bundle.main.url(forResource: "MicStart", withExtension: "caf")!
    private static let endCaf = Bundle.main.url(forResource: "MicStop", withExtension: "caf")!
    #if canImport(AppKit)
        private static let soundEffectVolume: Float = 0.2
        private static let startEffect = NSSound(contentsOf: startCaf, byReference: true)!
        private static let endEffect = NSSound(contentsOf: endCaf, byReference: true)!
    #else
        private static let soundEffectVolume: Float = 0.8
        private static let startEffect = try! AVAudioPlayer(contentsOf: startCaf)
        private static let endEffect = try! AVAudioPlayer(contentsOf: endCaf)
    #endif
    func playEffect(_ effect: Effect) {
        if muted { return }
        let sound = switch effect {
        case .startListening: Self.startEffect
        case .endListening: Self.endEffect
        }
        #if os(iOS)
            sound.prepareToPlay()
        #endif
        sound.currentTime = 0
        sound.volume = Self.soundEffectVolume
        sound.play()
        DispatchQueue.main.asyncAfter(deadline: .now() + sound.duration + 0.1) {
            sound.stop()
        }
    }
}
