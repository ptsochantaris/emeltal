import Accelerate
@preconcurrency import AVFoundation
import Foundation

final actor Mic {
    enum State: Equatable, Sendable {
        static func == (lhs: Self, rhs: Self) -> Bool {
            switch lhs {
            case .quiet:
                if case .quiet = rhs {
                    return true
                }
            case let .talking(voiceDetectedL, _):
                if case let .talking(voiceDetectedR, _) = rhs {
                    return voiceDetectedL == voiceDetectedR
                }
            }
            return false
        }

        case quiet(prefixBuffer: [Float]), talking(voiceDetected: Bool, quietCount: Int)

        var isQuiet: Bool {
            if case .quiet = self {
                return true
            }
            return false
        }
    }

    static var havePermission: Bool {
        #if canImport(AppKit)
            AVCaptureDevice.authorizationStatus(for: .audio) == .authorized
        #else
            AVAudioApplication.shared.recordPermission == .granted
        #endif
    }

    var state = State.quiet(prefixBuffer: []) {
        didSet {
            switch oldValue {
            case .quiet:
                switch state {
                case .quiet:
                    break

                case .talking:
                    log("Starting to listen")
                    stateStream.continuation.yield(state)
                }

            case let .talking(voiceDetectedCurrent, _):
                switch state {
                case .quiet:
                    log("Finished speaking")
                    stateStream.continuation.yield(.quiet(prefixBuffer: []))

                case let .talking(voiceDetectedNew, _):
                    if voiceDetectedCurrent, !voiceDetectedNew {
                        log("Stopped or paused?")
                        stateStream.continuation.yield(state)
                    } else if !voiceDetectedCurrent, voiceDetectedNew {
                        log("Was a pause, still listening")
                        stateStream.continuation.yield(state)
                    }
                }
            }
        }
    }

    let stateStream = AsyncStream.makeStream(of: State.self, bufferingPolicy: .unbounded)

    private var buffer = [Float]()

    func warmup() async {
        try? await start(detectVoice: false)
        _ = try? await stop(temporary: false)
        log("Mic warmup done")
    }

    enum TapState {
        case none, added(usingVoiceDetection: Bool), stopping

        var isStopping: Bool {
            if case .stopping = self {
                return true
            }
            return false
        }
    }

    private var tapState = TapState.none
    private var engineInUse = false
    private func isUsingEngine(_ using: Bool) async throws {
        let oldState = engineInUse
        engineInUse = using
        // state updated, can go async from here
        if !oldState, using {
            try await AudioEngineManager.shared.willUseEngine()
        } else if oldState, !using {
            await AudioEngineManager.shared.doneUsingEngine()
        }
    }

    private func removeTap() async {
        guard case .added = tapState else {
            return
        }
        tapState = .stopping
        while tapState.isStopping {
            try? await Task.sleep(for: .seconds(0.1))
        }
        await AudioEngineManager.shared.getEngine().inputNode.removeTap(onBus: 0)
    }

    private static let transcriptionSampleRate = 16000
    private static let micBufferSize: UInt32 = 8192
    private static let fft = FFT(bufferSize: Int(micBufferSize), minFrequency: 1500, maxFrequency: 3500, numberOfBands: 1, windowType: .none, sampleRate: transcriptionSampleRate)
    private static let outputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(transcriptionSampleRate), channels: 1, interleaved: true)!
    private static let outputFrames = AVAudioFrameCount(outputFormat.sampleRate)

    private func addTap(useVoiceDetection: Bool) async throws {
        switch tapState {
        case .none:
            break

        case .stopping:
            while tapState.isStopping {
                try? await Task.sleep(for: .seconds(0.1))
            }

        case let .added(usingVoiceDetection):
            if useVoiceDetection == usingVoiceDetection {
                return
            }
            await removeTap()
        }

        tapState = .added(usingVoiceDetection: useVoiceDetection)

        let input = await AudioEngineManager.shared.getEngine().inputNode
        let inputFormat = input.outputFormat(forBus: 0)
        let sampleRate = AVAudioFrameCount(inputFormat.sampleRate)

        let converter = AVAudioConverter(from: inputFormat, to: Self.outputFormat)!

        let convertedBufferFrames = Int(Self.outputFrames * Self.micBufferSize / sampleRate)
        let convertedBufferBytes = convertedBufferFrames * MemoryLayout<Float>.size
        let convertedBuffer = AVAudioPCMBuffer(pcmFormat: Self.outputFormat, frameCapacity: AVAudioFrameCount(convertedBufferFrames))!
        var voiceDetected = !useVoiceDetection
        var micLevels = [Float]()
        var startedLevel: Float = 0

        let audioPointer = convertedBuffer.floatChannelData![0]
        var audioProcessingBuffer = UnsafeMutableBufferPointer(start: audioPointer, count: convertedBufferFrames)

        var voiceFilter = vDSP.Biquad(
            coefficients: [0.7781271848311052, -1.5562543696622104, 0.7781271848311052, -1.494679407120035, 0.6178293322043856],
            channelCount: 1,
            sectionCount: 1,
            ofType: Float.self
        )!

        input.installTap(onBus: 0, bufferSize: Self.micBufferSize, format: inputFormat) { [weak self] incomingBuffer, _ in
            guard let self else { return }

            if useVoiceDetection {
                let power = Self.fft.fftForwardSingleBandMagnitude(incomingBuffer.floatChannelData![0])
                micLevels.append(power)
                if micLevels.count > 8 {
                    micLevels.remove(at: 0)
                    let oldLevels = micLevels[0] + micLevels[1] + micLevels[2]
                    let newLevels = micLevels[5] + micLevels[6] + micLevels[7]
                    if voiceDetected {
                        if oldLevels > newLevels {
                            let ratio = oldLevels / newLevels
                            if ratio > 20 {
                                log("Voice end (drop in power)")
                                voiceDetected = false
                            }
                        } else if abs(startedLevel - newLevels) < 0.3, abs(startedLevel - oldLevels) < 0.3 {
                            log("Voice end (low volume - was: \(newLevels) vs started at \(oldLevels)")
                            voiceDetected = false
                        }
                    } else {
                        if newLevels > oldLevels {
                            let ratio = newLevels / oldLevels
                            if ratio > 10 {
                                log("Voice start")
                                startedLevel = oldLevels
                                voiceDetected = true
                            }
                        }
                    }
                }
            }

            var error: NSError?
            var reported = AVAudioConverterInputStatus.haveData
            converter.convert(to: convertedBuffer, error: &error) { _, outStatus in
                outStatus.pointee = reported
                reported = .noDataNow
                return incomingBuffer
            }
            if error != nil { return }

            voiceFilter.apply(input: audioProcessingBuffer, output: &audioProcessingBuffer)

            let segment = [Float](unsafeUninitializedCapacity: convertedBufferFrames) { buffer, initializedCount in
                memcpy(buffer.baseAddress, audioPointer, convertedBufferBytes)
                initializedCount = convertedBufferFrames
            }

            queueWorkaround(segment: segment, isVoiceHeard: voiceDetected)
        }
    }

    private nonisolated func queueWorkaround(segment: [Float], isVoiceHeard: Bool) {
        Task {
            await append(segment: segment, isVoiceHeard: isVoiceHeard)
        }
    }

    private var remoteMode = false

    func setRemoteMode(_ remote: Bool) {
        remoteMode = remote
    }

    private enum RunState {
        case stopped, paused, recording
    }

    private var runState = RunState.stopped

    func start(detectVoice: Bool) async throws {
        if runState == .recording {
            return
        }
        runState = .recording
        buffer.removeAll()
        state = .quiet(prefixBuffer: [])

        if remoteMode {
            log("Mic running (remote mode)")
            return
        }

        try await isUsingEngine(true)
        try await addTap(useVoiceDetection: detectVoice)
        log("Mic running")
    }

    func addToBuffer(_ data: Data) {
        let floatSize = MemoryLayout<Float>.size
        let floatCount = data.count / floatSize
        log("Received \(floatCount) samples from external source")
        let floats = [Float](unsafeUninitializedCapacity: floatCount) { floatBuffer, initializedCount in
            _ = data.copyBytes(to: floatBuffer)
            initializedCount = floatCount
        }
        buffer.append(contentsOf: floats)
    }

    private func append(segment: [Float], isVoiceHeard: Bool) {
        switch state {
        case let .quiet(prefixBuffer):
            var newBuffer = prefixBuffer + segment
            if newBuffer.count > Self.transcriptionSampleRate {
                newBuffer.removeFirst(1000)
            }
            if isVoiceHeard {
                state = .talking(voiceDetected: true, quietCount: 0)
                buffer.append(contentsOf: newBuffer)
            } else {
                state = .quiet(prefixBuffer: newBuffer)
            }
        case let .talking(_, currentQuietCount):
            buffer.append(contentsOf: segment)
            if isVoiceHeard {
                state = .talking(voiceDetected: true, quietCount: 0)
            } else {
                if currentQuietCount > 19 {
                    state = .quiet(prefixBuffer: [])
                } else {
                    state = .talking(voiceDetected: false, quietCount: currentQuietCount + 1)
                }
            }
        }
        if tapState.isStopping {
            tapState = .none
        }
    }

    func stop(temporary: Bool) async throws -> [Float] {
        switch runState {
        case .stopped:
            return []

        case .paused:
            if temporary {
                return []
            }

        case .recording:
            break
        }

        await removeTap()
        if temporary {
            runState = .paused
        } else {
            runState = .stopped
            try await isUsingEngine(false)
        }

        let ret = buffer
        buffer.removeAll()
        log("Mic stopped, have \(ret.count) samples, temporary: \(temporary)")
        return ret
    }
}
