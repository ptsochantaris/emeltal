import Foundation

@GGMLActor
final class WhisperContext {
    private var context: OpaquePointer

    init(asset: Model) async throws(EmeltalError) {
        var params = whisper_context_default_params()
        params.flash_attn = true
        params.use_gpu = await asset.memoryEstimate.offloadAsr

        let modelPath = await asset.localModelPath.path
        guard let context = whisper_init_from_file_with_params(modelPath, params) else {
            throw .message("Could not initialise context")
        }

        self.context = context
    }

    func shutdown() {
        whisper_free(context)
    }

    deinit {
        log("Whisper context deinit")
    }

    func warmup() throws {
        _ = try transcribe(samples: [Float](repeating: 0, count: 32000))
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
        params.suppress_nst = true
        params.language = enCString
        params.single_segment = true
        params.temperature = 0.4
        return params
    }()

    func transcribe(samples: [Float]) throws -> String {
        log("Transcribing audio")
        try samples.withUnsafeBufferPointer { floats in
            if whisper_full(context, params, floats.baseAddress, Int32(floats.count)) < 0 {
                throw EmeltalError.message("Failed to run the whisper model")
            }
        }
        return (0 ..< whisper_full_n_segments(context)).map {
            String(cString: whisper_full_get_segment_text(context, $0))
        }.joined()
    }
}
