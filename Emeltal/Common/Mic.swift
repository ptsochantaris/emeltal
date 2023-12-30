import Accelerate
import AVFoundation
import Combine
import Foundation

final actor Mic: NSObject {
    enum State: Equatable {
        static func == (lhs: Self, rhs: Self) -> Bool {
            switch lhs {
            case .quiet:
                if case .quiet = rhs {
                    return true
                }
            case .talking:
                if case .talking = rhs {
                    return true
                }
            }
            return false
        }

        case quiet(prefixBuffer: [Float]), talking(quietPeriods: Int)

        var isQuiet: Bool {
            if case .quiet = self {
                return true
            }
            return false
        }
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
                    statePublisher.send(.talking(quietPeriods: 0))
                }

            case let .talking(quietPeriods1):
                switch state {
                case .quiet:
                    log("Finished speaking")
                    statePublisher.send(.quiet(prefixBuffer: []))

                case let .talking(quietPeriods2):
                    if quietPeriods1 == 0, quietPeriods2 != 0 {
                        log("Stopped or paused?")
                    } else if quietPeriods1 != 0, quietPeriods2 == 0 {
                        log("Was a pause, still listening")
                    }
                }
            }
        }
    }

    let statePublisher = CurrentValueSubject<State, Never>(.quiet(prefixBuffer: []))

    private var buffer = [Float]()
    static let SampleRate = 16000

    func warmup() async {
        try? await start(detectVoice: false)
        _ = try? await stop(temporary: false)
        log("Mic warmup done")
    }

    enum TapState {
        case none, added(usingVoiceDetection: Bool)
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
        await AudioEngineManager.shared.getEngine().inputNode.removeTap(onBus: 0)
        tapState = .none
    }

    private static var voiceFilter = vDSP.Biquad(
        coefficients: [0.7781271848311052, -1.5562543696622104, 0.7781271848311052, -1.494679407120035, 0.6178293322043856],
        channelCount: 1,
        sectionCount: 1,
        ofType: Float.self
    )!

    private func addTap(useVoiceDetection: Bool) async throws {
        switch tapState {
        case .none:
            break

        case let .added(usingVoiceDetection):
            if useVoiceDetection == usingVoiceDetection {
                return
            }
            await removeTap()
        }

        tapState = .added(usingVoiceDetection: useVoiceDetection)

        let bufferSize: UInt32 = 4096
        let fft = FFT(bufferSize: Int(bufferSize), minFrequency: 1500, maxFrequency: 3500, numberOfBands: 1, windowType: .none, sampleRate: Self.SampleRate)
        var currentMicLevel: Float = 0
        var lastMicLevel: Float?
        var voiceDetected = !useVoiceDetection

        let audioEngine = await AudioEngineManager.shared.getEngine()
        let input = audioEngine.inputNode

        let outputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(Self.SampleRate), channels: 1, interleaved: true)!
        let outputFrames = AVAudioFrameCount(outputFormat.sampleRate)

        var cachedConverter: AVAudioConverter?

        input.installTap(onBus: 0, bufferSize: bufferSize, format: nil) { [weak self] incomingBuffer, _ in
            guard let self else { return }

            if useVoiceDetection {
                let power = fft.fftForwardSingleBandMagnitude(incomingBuffer.floatChannelData![0])
                if let lastMicLevel {
                    currentMicLevel = lastMicLevel * 0.4 + power * 0.6
                    let currentLevelRoc = abs(currentMicLevel - lastMicLevel)
                    if voiceDetected {
                        if currentLevelRoc < 0.08 {
                            voiceDetected = false
                        }
                    } else if currentLevelRoc > 80 {
                        voiceDetected = true
                    }
                    // print(currentMicLevel, abs(currentLevelRoc))
                } else {
                    currentMicLevel = power
                }
                lastMicLevel = currentMicLevel
            }

            let inputFormat = incomingBuffer.format
            let converter: AVAudioConverter
            let sampleRate = inputFormat.sampleRate
            if let c = cachedConverter, c.inputFormat.sampleRate == sampleRate {
                converter = c
            } else {
                log("Creating a sample rate converter for incoming mic")
                converter = AVAudioConverter(from: inputFormat, to: outputFormat)!
                cachedConverter = converter
            }

            var error: NSError?
            var reported = AVAudioConverterInputStatus.haveData
            let convertedBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: outputFrames * bufferSize / AVAudioFrameCount(sampleRate))!
            converter.convert(to: convertedBuffer, error: &error) { _, outStatus in
                outStatus.pointee = reported
                reported = .noDataNow
                return incomingBuffer
            }
            assert(error == nil)

            let numSamples = Int(convertedBuffer.frameLength)
            var buffer = UnsafeMutableBufferPointer(start: convertedBuffer.floatChannelData![0], count: numSamples)
            Self.voiceFilter.apply(input: buffer, output: &buffer)

            let segment = [Float](unsafeUninitializedCapacity: numSamples) { buffer, initializedCount in
                memcpy(buffer.baseAddress, convertedBuffer.floatChannelData![0], numSamples * MemoryLayout<Float>.size)
                initializedCount = numSamples
            }

            Task { [voiceDetected] in
                await self.append(segment: segment, voiceDetected: voiceDetected)
            }
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

    private func append(segment: [Float], voiceDetected: Bool) {
        switch state {
        case let .quiet(prefixBuffer):
            var newBuffer = prefixBuffer + segment
            if newBuffer.count > Self.SampleRate {
                newBuffer.removeFirst(1000)
            }
            if voiceDetected {
                state = .talking(quietPeriods: 0)
                buffer.append(contentsOf: newBuffer)
            } else {
                state = .quiet(prefixBuffer: newBuffer)
            }
        case let .talking(quietPeriods):
            buffer.append(contentsOf: segment)
            if voiceDetected {
                state = .talking(quietPeriods: 0)
            } else {
                let newCount = quietPeriods + 1
                if newCount > 10 {
                    state = .quiet(prefixBuffer: [])
                } else {
                    state = .talking(quietPeriods: newCount)
                }
            }
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
