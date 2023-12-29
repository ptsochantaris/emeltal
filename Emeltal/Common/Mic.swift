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
    private let SampleRate = 16000

    func warmup() async {
        try? await start()
        _ = try? await stop(temporary: false)
        log("Mic warmup done")
    }

    private var addedTap = false
    private var usingEngine = false

    private func removeTap() async {
        guard addedTap else { return }
        let audioEngine = AudioEngineManager.shared.engine
        audioEngine.inputNode.removeTap(onBus: 0)
        addedTap = false
    }

    private nonisolated func voiceFilter(_ buffer: UnsafeMutablePointer<Float>, len: Int) -> [Float] {
        // b0 b1 b2 a1 a2
        var r = vDSP.Biquad(coefficients: [1, -1.8769581297076159, 0.8827620567886247, -1.8798600932481204, 0.9399300466240602],
                            channelCount: 1,
                            sectionCount: 1,
                            ofType: Float.self)!

        let buffer = UnsafeMutableBufferPointer(start: buffer, count: len)
        return r.apply(input: buffer)
    }

    private func addTap() throws {
        if addedTap {
            return
        }
        addedTap = true

        let audioEngine = AudioEngineManager.shared.engine
        let input = audioEngine.inputNode

        let inputFormat = input.outputFormat(forBus: 0)
        let incomingSampleRate = AVAudioFrameCount(inputFormat.sampleRate)

        let outputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(SampleRate), channels: 1, interleaved: true)!
        let outputFrames = AVAudioFrameCount(outputFormat.sampleRate)

        let converter = AVAudioConverter(from: inputFormat, to: outputFormat)!

        input.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { [weak self] incomingBuffer, _ in
            guard let self else { return }

            let inNumberFrames = incomingBuffer.frameLength
            let convertedBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: outputFrames * inNumberFrames / incomingSampleRate)!

            var error: NSError?
            var reported = AVAudioConverterInputStatus.haveData
            let status = converter.convert(to: convertedBuffer, error: &error) { _, outStatus in
                outStatus.pointee = reported
                reported = .noDataNow
                return incomingBuffer
            }
            assert(status != .error)

            let numSamples = Int(convertedBuffer.frameLength)
            let convertedData = convertedBuffer.floatChannelData![0]
            let filteredCopy = voiceFilter(convertedData, len: numSamples)
            let avgValue = vDSP.meanMagnitude(filteredCopy)
            let v: Float = if avgValue == 0 {
                -100
            } else {
                20.0 * log10f(avgValue)
            }
            Task {
                await self.append(segment: filteredCopy, instantEnergy: v)
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

    func start() async throws {
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

        try addTap()
        if !usingEngine {
            usingEngine = true
            try await AudioEngineManager.shared.willUseEngine()
        }
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

    private var lastEnergy: Float?
    private var voiceLevel: Float = 0
    private func append(segment: [Float], instantEnergy: Float) {
        let energy: Float
        let reference: Float
        if let lastEnergy {
            reference = lastEnergy
            energy = (0.5 * instantEnergy) + (0.5 * lastEnergy)
        } else {
            reference = instantEnergy
            energy = instantEnergy
        }
        lastEnergy = energy

        switch state {
        case let .quiet(prefixBuffer):
            var newBuffer = prefixBuffer + segment
            if newBuffer.count > SampleRate {
                newBuffer.removeFirst(1000)
            }
            #if os(macOS)
                let startDiff: Float = 6
            #else
                let startDiff: Float = 10
            #endif
            let diff = max(0, energy - reference)
            log("Scanning for spike over \(startDiff.rounded()): \(diff.rounded()) - slow level: \(energy.rounded()) - ref: \(reference.rounded())")
            if diff > startDiff {
                voiceLevel = (reference * 0.7) + (instantEnergy * 0.3)
                state = .talking(quietPeriods: 0)
                buffer.append(contentsOf: newBuffer)
            } else {
                state = .quiet(prefixBuffer: newBuffer)
            }
        case let .talking(quietPeriods):
            log("Scanning for quiet below \(voiceLevel.rounded()) - slow level: \(energy.rounded())")
            if energy < voiceLevel {
                let count = quietPeriods + segment.count
                if count > SampleRate * 2 {
                    state = .quiet(prefixBuffer: [])
                } else {
                    state = .talking(quietPeriods: count)
                    buffer.append(contentsOf: segment)
                }
            } else {
                state = .talking(quietPeriods: 0)
                buffer.append(contentsOf: segment)
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
            if usingEngine {
                usingEngine = false
                await AudioEngineManager.shared.doneUsingEngine()
            }
            runState = .stopped
        }

        let ret = buffer
        buffer.removeAll()
        log("Mic stopped, have \(ret.count) samples, temporary: \(temporary)")
        return ret
    }
}
