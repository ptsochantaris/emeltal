import Foundation

struct UTF8Builder {
    enum Result {
        case moreRequired, result(String)
    }

    enum RunLength: Int {
        case one = 1, two, three, four
    }

    var bytes = [UInt8]()
    var expectedRunLength: RunLength = .one

    mutating func parseByte(_ byte: UInt8) -> Result {
        // Not escaped
        if byte & 0b1000_0000 == 0 {
            expectedRunLength = .one
            bytes = [byte & 0b0111_1111]
        }

        // start of 4 byte sequence
        else if byte & 0b1111_0000 == 0b1111_0000 {
            expectedRunLength = .four
            bytes = [byte & 0b0000_1111]
            return .moreRequired
        }

        // start of 3 byte sequence
        else if byte & 0b1110_0000 == 0b1110_0000 {
            expectedRunLength = .three
            bytes = [byte & 0b0001_1111]
            return .moreRequired
        }

        // start of 2 byte sequence
        else if byte & 0b1100_0000 == 0b1100_0000 {
            expectedRunLength = .two
            bytes = [byte & 0b0011_1111]
            return .moreRequired
        }

        // Continuing sequence
        else if byte & 0b1100_0000 == 0b1000_0000 {
            bytes.append(byte & 0b0011_1111)
        }

        // Malformed, throw away existing sequence and treat as first unescaped char
        else {
            expectedRunLength = .one
            bytes = [byte & 0b0111_1111]
        }

        if bytes.count < expectedRunLength.rawValue {
            return .moreRequired
        }

        let rawScalarValue = switch expectedRunLength {
        case .one:
            UInt32(bytes[0])
        case .two:
            UInt32(bytes[0]) << 6 | UInt32(bytes[1])
        case .three:
            UInt32(bytes[0]) << 12 | UInt32(bytes[1]) << 6 | UInt32(bytes[2])
        case .four:
            UInt32(bytes[0]) << 18 | UInt32(bytes[1]) << 12 | UInt32(bytes[2]) << 6 | UInt32(bytes[3])
        }

        if let scalar = UnicodeScalar(rawScalarValue) {
            let c = Character(scalar)
            return .result(String([c]))
        } else {
            return .result("")
        }
    }
}
