#if canImport(AppKit)
    import AppKit
#endif
@preconcurrency import AVFAudio
import Combine
import Foundation

extension AVSpeechUtterance: @retroactive @unchecked Sendable {}

final actor Speaker {
    var havePreferredVoice = false

    private let voice: AVSpeechSynthesisVoice
    private let synth = AVSpeechSynthesizer()
    private var muted = false
    private let watcher = UtteranceWatcher()
    private let effectPlayer = AVAudioPlayerNode()

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

    private final actor UtteranceWatcher: NSObject, AVSpeechSynthesizerDelegate {
        private var utterances = Set<AVSpeechUtterance>()

        private func remove(utterance: AVSpeechUtterance) { utterances.remove(utterance) }

        func add(utterance: AVSpeechUtterance) { utterances.insert(utterance) }

        func reset() { utterances.removeAll() }

        nonisolated func speechSynthesizer(_: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
            Task {
                await remove(utterance: utterance)
            }
        }

        nonisolated func speechSynthesizer(_: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
            Task {
                await remove(utterance: utterance)
            }
        }

        func waitForZero() async {
            while !utterances.isEmpty {
                try? await Task.sleep(for: .seconds(0.1))
            }
            log("Speaker stopped speaking")
        }

        deinit {
            log("UtteranceWatcher deinit")
        }
    }

    init() throws(EmeltalError) {
        #if os(iOS)
            synth.usesApplicationAudioSession = true
        #endif
        synth.delegate = watcher
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
                throw .message("Could not find any TTS voices in the system (counted: \(allVoices.count))")
            }
        }
    }

    func warmup() throws {
        Task {
            synth.write(AVSpeechUtterance(string: "Warmup")) { _ in }
            log("TTS warmup done")
            log("Starting sound effect loop")
            try await effectPlayerLoop()
        }
    }

    func cancelIfNeeded() {
        synth.stopSpeaking(at: .immediate)
        Task {
            await watcher.reset()
        }
    }

    func waitForCompletion() async {
        await watcher.waitForZero()
    }

    func add(text: String) async {
        if muted { return }
        let utterance = utterance(for: text)
        await watcher.add(utterance: utterance)
        synth.speak(utterance)
    }

    func render(text: String) async -> Data? {
        let utterance = utterance(for: text)
        return await withCheckedContinuation { continuation in
            synth.write(utterance) { audioBuffer in
                if let buf = audioBuffer as? AVAudioPCMBuffer, let speech = buf.floatChannelData?[0] {
                    let data = Data(bytes: speech, count: Int(buf.frameLength))
                    continuation.resume(returning: data)
                } else {
                    continuation.resume(returning: nil)
                }
            }
        }
    }

    private func utterance(for text: String) -> AVSpeechUtterance {
        let textToPlay = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let utterance = AVSpeechUtterance(string: textToPlay)
        utterance.rate = 0.51
        utterance.voice = voice
        if textToPlay.hasSuffix(".") || textToPlay.hasSuffix("!") || textToPlay.hasSuffix("?") || textToPlay.hasSuffix(":") || textToPlay.hasSuffix("\n") {
            utterance.postUtteranceDelay = 0.18
        }
        return utterance
    }

    enum Effect {
        private static let startCaf = Bundle.main.url(forResource: "MicStart", withExtension: "caf")!
        private static let endCaf = Bundle.main.url(forResource: "MicStop", withExtension: "caf")!

        case startListening, endListening

        var audioFile: AVAudioFile {
            switch self {
            case .startListening:
                try! AVAudioFile(forReading: Self.startCaf)

            case .endListening:
                try! AVAudioFile(forReading: Self.endCaf)
            }
        }

        var preferredVolume: Float {
            switch self {
            case .startListening: 0.1
            case .endListening: 0.4
            }
        }
    }

    func shutdown() {
        effectQueue.continuation.finish()
    }

    private let effectQueue = AsyncStream.makeStream(of: Effect.self, bufferingPolicy: .unbounded)

    func play(effect: Effect) {
        effectQueue.continuation.yield(effect)
    }

    private func effectPlayerLoop() async throws {
        for await effect in effectQueue.stream {
            if muted { continue }

            let manager = AudioEngineManager.shared
            try? await manager.willUseEngine()
            guard await manager.getEngine().isRunning else {
                // weird
                continue
            }

            let (effectPlayer, msec) = await manager.getEffectPlayer(scheduling: effect)

            // log("Playing effect \(effect); duration: \(msec) ms; volume: \(effectPlayer.volume)")
            try? await Task.sleep(nanoseconds: (msec + 100) * NSEC_PER_MSEC)
            // log("Stopping player")

            effectPlayer.stop()
            await manager.doneUsingEngine()
        }
        log("Speaker shutdown")
    }
}
