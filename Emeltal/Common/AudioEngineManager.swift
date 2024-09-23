@preconcurrency import AVFoundation
import Foundation
import PopTimer

final actor AudioEngineManager {
    static let shared = AudioEngineManager()

    private let engine = AVAudioEngine()
    private let effectPlayer = AVAudioPlayerNode()

    func getEngine() -> AVAudioEngine {
        engine
    }

    func getEffectPlayer(scheduling effect: Speaker.Effect) -> (AVAudioPlayerNode, UInt64) {
        let sound = effect.audioFile
        let msec = UInt64(Double(sound.length * 1000) / sound.processingFormat.sampleRate)

        effectPlayer.volume = effect.preferredVolume
        effectPlayer.play()

        Task {
            await effectPlayer.scheduleFile(sound, at: nil)
        }

        return (effectPlayer, msec)
    }

    init() {
        engine.attach(effectPlayer)
        engine.connect(effectPlayer, to: engine.mainMixerNode, format: engine.mainMixerNode.outputFormat(forBus: 0))

        // Hack: Add and remove tap to enable mic input
        engine.inputNode.installTap(onBus: 0, bufferSize: 4096, format: nil) { _, _ in }
        engine.inputNode.removeTap(onBus: 0)

        engine.prepare()
        log("Audio engine instantiated")
    }

    private var count = 0

    func willUseEngine() throws {
        engineShutdown.abort()
        if engine.isRunning {
            // log("Audio engine already running")
        } else {
            log("Starting audio engine")
            try engine.start()
        }
        count += 1
    }

    private lazy var engineShutdown = PopTimer(timeInterval: 1) { [weak self] in
        guard let self else { return }
        if engine.isRunning {
            log("Pausing audio engine")
            engine.stop()
            engine.prepare()
        } else {
            // log("Audio engine already paused")
        }
    }

    func doneUsingEngine() {
        count -= 1
        assert(count >= 0)
        if count == 0 {
            engineShutdown.push()
        } else {
            // log("Audio engine still in use, will keep running")
        }
    }
}
