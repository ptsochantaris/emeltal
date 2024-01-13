import Foundation

final class GGMLExecutor: SerialExecutor {
    static let shared = GGMLExecutor()

    private let ggmlQueue = DispatchQueue(label: "build.bru.emeltal.ggml", qos: .userInitiated)

    func enqueue(_ job: consuming ExecutorJob) {
        let j = UnownedJob(job)
        let e = asUnownedSerialExecutor()
        ggmlQueue.async {
            j.runSynchronously(on: e)
        }
    }
}

@globalActor actor GGMLActor {
    static var shared = GGMLActor()

    static var sharedUnownedExecutor = GGMLExecutor.shared.asUnownedSerialExecutor()

    nonisolated var unownedExecutor: UnownedSerialExecutor {
        Self.sharedUnownedExecutor
    }
}

@GGMLActor
final class WhisperContext {
    private var context: OpaquePointer
    private let manager: AssetManager

    init(manager: AssetManager) async throws {
        self.manager = manager

        var params = whisper_context_default_params()
        params.use_gpu = true

        guard let context = whisper_init_from_file_with_params(manager.asset.localModelPath.path, params) else {
            throw "Could not initialise context"
        }

        self.context = context
        // whisper_print_system_info()
    }

    deinit {
        whisper_free(context)
    }

    func warmup() {
        _ = transcribe(samples: [Float](repeating: 0, count: 32000))
    }

    private static let enCString: UnsafePointer<Int8> = {
        let buffer = malloc(3)!
        "en".withCString {
            _ = memcpy(buffer, $0, 3)
        }
        return UnsafePointer<Int8>(OpaquePointer(buffer))
    }()

    private let params: whisper_full_params = {
        var params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY)
        params.n_threads = 1
        params.print_timestamps = false
        params.print_realtime = false
        params.print_progress = false
        params.print_special = false
        params.no_timestamps = true
        params.no_context = true
        params.suppress_blank = true
        params.suppress_non_speech_tokens = true
        params.language = enCString
        params.single_segment = true
        params.temperature = 0.4
        return params
    }()

    func transcribe(samples: [Float]) -> String {
        samples.withUnsafeBufferPointer { floats in
            log("Transcribing audio")
            // whisper_reset_timings(context)
            if whisper_full(context, params, floats.baseAddress, Int32(floats.count)) < 0 {
                fatalError("Failed to run the whisper model")
            } else {
                // whisper_print_timings(context)
                var transcription = ""
                for i in 0 ..< whisper_full_n_segments(context) {
                    transcription += String(cString: whisper_full_get_segment_text(context, i))
                }
                return transcription
            }
        }
    }
}
