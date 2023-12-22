import Accelerate
import AVFoundation
import Combine
import Foundation

final actor Mic: NSObject, AVCaptureAudioDataOutputSampleBufferDelegate {
    var state = MicState.quiet(prefixBuffer: []) {
        didSet {
            switch oldValue {
            case .quiet:
                switch state {
                case .quiet:
                    break

                case .listening:
                    log("Starting to listen")
                    statePublisher.send(.listening(quietPeriods: 0))
                }

            case let .listening(quietPeriods1):
                switch state {
                case .quiet:
                    log("Stopped listening")
                    statePublisher.send(.quiet(prefixBuffer: []))

                case let .listening(quietPeriods2):
                    if quietPeriods1 == 0, quietPeriods2 != 0 {
                        log("Stopped or paused?")
                    } else if quietPeriods1 != 0, quietPeriods2 == 0 {
                        log("Was a pause, still listening")
                    }
                }
            }
        }
    }

    let statePublisher = CurrentValueSubject<MicState, Never>(.quiet(prefixBuffer: []))

    private var _session: AVCaptureSession?
    private let audio = AVCaptureAudioDataOutput()
    private var buffer = [Float]()
    private let processQueue = DispatchQueue(label: "build.bru.emeltal.mic")
    private let SampleRate = 16000

    private func getSession() async throws -> AVCaptureSession {
        if let _session {
            return _session
        }

        guard await AVCaptureDevice.requestAccess(for: .audio),
              let mic = AVCaptureDevice.default(.microphone, for: .audio, position: .unspecified)
        else {
            throw "Capture device not accessible"
        }

        log("Using mic: \(mic.description)")

        let input = try AVCaptureDeviceInput(device: mic)

        let audio = AVCaptureAudioDataOutput()
        audio.setSampleBufferDelegate(self, queue: processQueue)
        audio.audioSettings = [
            AVNumberOfChannelsKey: 1,
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: SampleRate
        ]

        let session = AVCaptureSession()
        session.addOutput(audio)
        session.addInput(input)
        _session = session
        return session
    }

    // TODO: detect and advise to turn on voice isolation

    func start() async throws {
        let session = try await getSession()
        if session.isRunning {
            return
        }
        buffer.removeAll()
        state = .quiet(prefixBuffer: [])
        session.startRunning()
        log("Mic running")
    }

    nonisolated func captureOutput(_: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let numberOfSamples = sampleBuffer.numSamples
        if numberOfSamples < 1 {
            return
        }
        try? sampleBuffer.withAudioBufferList { listPointer, _ in
            guard let channel = connection.audioChannels.first, let dataPointer = listPointer.unsafePointer.pointee.mBuffers.mData else {
                return
            }
            let segmentCopy = [Float](unsafeUninitializedCapacity: numberOfSamples) { buffer, initializedCount in
                _ = memcpy(buffer.baseAddress!, dataPointer, numberOfSamples)
                initializedCount = numberOfSamples
            }
            let energy = channel.peakHoldLevel
            Task {
                await append(segment: segmentCopy, energy: energy)
            }
        }
    }

    private var lastEnergy: Float = 1000
    private let voiceSensitivity: Float = 20
    private func append(segment: [Float], energy: Float) {
        switch state {
        case let .quiet(prefixBuffer):
            var newBuffer = prefixBuffer + segment
            if newBuffer.count > 16000 {
                newBuffer.removeFirst(1000)
            }
            if lastEnergy < -voiceSensitivity, energy > -voiceSensitivity {
                state = .listening(quietPeriods: 0)
                buffer.append(contentsOf: newBuffer)
            } else {
                state = .quiet(prefixBuffer: newBuffer)
            }
        case let .listening(quietPeriods):
            if energy < -voiceSensitivity {
                let count = quietPeriods + segment.count
                if count > SampleRate * 2 {
                    state = .quiet(prefixBuffer: [])
                } else {
                    state = .listening(quietPeriods: count)
                    buffer.append(contentsOf: segment)
                }
            } else {
                state = .listening(quietPeriods: 0)
                buffer.append(contentsOf: segment)
            }
        }
        lastEnergy = energy
    }

    func stop() async throws -> [Float] {
        let session = try await getSession()
        guard session.isRunning else {
            return []
        }
        session.stopRunning()
        let ret = buffer
        buffer.removeAll()
        log("Mic stopped")
        _session = nil
        return ret
    }
}
