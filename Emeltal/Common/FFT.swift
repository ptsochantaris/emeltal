import Accelerate
import Foundation

// Derived from TempiFFT at: https://github.com/jscalo/tempi-fft
// Simplified and optimised to only handle one fixed buffer size

final class FFT {
    enum WindowType {
        case none, hanning, hamming
    }

    struct Band {
        let magnitude: Float
        let frequency: Float
    }

    private(set) var bands: [Band]
    private var complexBuffer: DSPSplitComplex
    private var magnitudes: UnsafeMutableBufferPointer<Float>

    private let size: Int
    private let bandwidth: Float
    private let halfSize: Int
    private let fftSetup: vDSP.FFT<DSPSplitComplex>
    private let window: [Float]?
    private let numberOfBands: Int
    private let realBuffer: UnsafeMutableBufferPointer<Float>
    private let imaginaryBuffer: UnsafeMutableBufferPointer<Float>
    private let magLowerRange, magUpperRange: Int
    private let ratio: Float

    init(bufferSize inSize: Int, minFrequency: Float, maxFrequency: Float, numberOfBands: Int, windowType: WindowType, sampleRate: Int) {
        self.numberOfBands = numberOfBands

        let sizeFloat = Float(inSize)

        #if DEBUG
            // Check if the size is a power of two
            let lg2 = logbf(sizeFloat)
            assert(remainderf(sizeFloat, powf(2.0, lg2)) == 0, "size must be a power of 2")
        #endif

        size = inSize
        halfSize = inSize / 2

        // create fft setup
        let log2Size = Int(log2f(sizeFloat))
        fftSetup = vDSP.FFT(log2n: vDSP_Length(log2Size), radix: .radix2, ofType: DSPSplitComplex.self)!

        realBuffer = UnsafeMutableBufferPointer.allocate(capacity: halfSize)
        imaginaryBuffer = UnsafeMutableBufferPointer.allocate(capacity: halfSize)
        complexBuffer = DSPSplitComplex(realp: realBuffer.baseAddress!, imagp: imaginaryBuffer.baseAddress!)

        magnitudes = UnsafeMutableBufferPointer.allocate(capacity: halfSize)
        bands = [Band](repeating: Band(magnitude: 0, frequency: 0), count: numberOfBands)

        switch windowType {
        case .hamming:
            window = vDSP.window(ofType: Float.self, usingSequence: .hamming, count: size, isHalfWindow: false)
        case .hanning:
            window = vDSP.window(ofType: Float.self, usingSequence: .hanningNormalized, count: size, isHalfWindow: false)
        default:
            window = nil
        }

        let nyquistFrequency = Float(sampleRate / 2)
        bandwidth = nyquistFrequency / Float(halfSize)
        magLowerRange = Int(Float(halfSize) * minFrequency / nyquistFrequency)
        magUpperRange = Int(Float(halfSize) * maxFrequency / nyquistFrequency)
        ratio = Float(magUpperRange - magLowerRange) / Float(numberOfBands)
    }

    deinit {
        magnitudes.deallocate()
        realBuffer.deallocate()
        imaginaryBuffer.deallocate()
    }

    func fftForward(_ buffer: UnsafeMutablePointer<Float>) {
        if let window {
            vDSP_vmul(buffer, 1, window, 1, buffer, 1, vDSP_Length(size))
        }
        buffer.withMemoryRebound(to: DSPComplex.self, capacity: halfSize) { pointer in
            vDSP_ctoz(pointer, 2, &complexBuffer, 1, vDSP_Length(halfSize))
        }
        fftSetup.forward(input: complexBuffer, output: &complexBuffer)
        vDSP.squareMagnitudes(complexBuffer, result: &magnitudes)

        for i in 0 ..< numberOfBands {
            let magsStartIdx = Int(floorf(Float(i) * ratio)) + magLowerRange
            let magsEndIdx = Int(floorf(Float(i + 1) * ratio)) + magLowerRange
            let magsAvg: Float = if magsEndIdx == magsStartIdx {
                // Can happen when numberOfBands < # of magnitudes. No need to average anything.
                magnitudes[magsStartIdx]
            } else {
                vDSP.mean(magnitudes[magsStartIdx ..< magsEndIdx])
            }
            let freq = (bandwidth * Float(magsStartIdx) + bandwidth * Float(magsEndIdx)) / 2
            bands[i] = Band(magnitude: magsAvg, frequency: freq)
        }
    }

    func fftForwardSingleBandMagnitude(_ buffer: UnsafeMutablePointer<Float>) -> Float {
        if let window {
            vDSP_vmul(buffer, 1, window, 1, buffer, 1, vDSP_Length(size))
        }
        buffer.withMemoryRebound(to: DSPComplex.self, capacity: halfSize) { pointer in
            vDSP_ctoz(pointer, 2, &complexBuffer, 1, vDSP_Length(halfSize))
        }
        fftSetup.forward(input: complexBuffer, output: &complexBuffer)
        vDSP.squareMagnitudes(complexBuffer, result: &magnitudes)
        return vDSP.mean(magnitudes)
    }
}

extension Float {
    var toDB: Float {
        // ceil to 128db in order to avoid log10'ing 0
        10 * log10f(max(self, 0.000000000001))
    }
}
