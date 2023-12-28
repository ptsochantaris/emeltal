import AVFoundation
import Foundation

final actor AudioEngineManager {
    static let shared = AudioEngineManager()

    let engine = AVAudioEngine()

    private var count = 0

    init() {
        let input = engine.inputNode
        input.isVoiceProcessingAGCEnabled = false
        input.isVoiceProcessingBypassed = false
        input.isVoiceProcessingInputMuted = false
    }

    func config(block: (AVAudioEngine) throws -> Void) throws {
        if engine.isRunning {
            engine.stop()
            try block(engine)
            try engine.start()
        } else {
            try block(engine)
            engine.prepare()
        }
    }

    func willUseEngine() throws {
        if engine.isRunning {
            log("Audio engine already running")
        } else {
            log("Starting audio engine")
            engine.prepare()
            try engine.start()
        }
        count += 1
    }

    func doneUsingEngine() {
        count -= 1
        if count == 0 {
            if engine.isRunning {
                log("Stopping audio engine")
                engine.stop()
                engine.prepare()
            } else {
                log("Audio engine already stopped")
            }
        } else {
            log("Audio engine still in use, will keep running")
        }
    }
}
