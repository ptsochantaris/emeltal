import Foundation
import Network

extension NWProtocolFramer.Message {
    convenience init(command: EmeltalLink.Command) {
        self.init(definition: EmeltalLink.definition)
        self["EmeltalCommand"] = command.rawValue
    }

    var command: EmeltalLink.Command? {
        guard let v = self["EmeltalCommand"] as? UInt64 else {
            return nil
        }
        return EmeltalLink.Command(rawValue: v)
    }
}

final class EmeltalLink: NWProtocolFramerImplementation {
    enum Command: UInt64 {
        case utterance = 0
        case audioBlock = 1
        case audioBlockLast = 2
        case unknown = 999
    }

    static var params: NWParameters {
        let options = NWProtocolFramer.Options(definition: definition)
        let parameters = NWParameters.applicationService
        parameters.defaultProtocolStack.applicationProtocols.insert(options, at: 0)
        return parameters
    }

    struct Header {
        let command: Command
        let length: UInt64
        static let uint64size = MemoryLayout<UInt64>.size
        static let dataSize = uint64size * 2

        var data: Data {
            var command = command
            var length = length
            return Data(bytes: &command, count: Self.uint64size)
                + Data(bytes: &length, count: Self.uint64size)
        }

        init(command: Command, length: UInt64) {
            self.command = command
            self.length = length
        }

        init(_ buffer: UnsafeMutableRawBufferPointer) {
            let tempCommand = buffer.load(as: UInt64.self)
            command = Command(rawValue: tempCommand) ?? .unknown
            length = buffer.load(fromByteOffset: Self.uint64size, as: UInt64.self)
        }
    }

    static let label = "Emeltal Link"
    static let definition = NWProtocolFramer.Definition(implementation: EmeltalLink.self)

    init(framer _: NWProtocolFramer.Instance) {}
    func wakeup(framer _: NWProtocolFramer.Instance) {}
    func stop(framer _: NWProtocolFramer.Instance) -> Bool { true }
    func cleanup(framer _: NWProtocolFramer.Instance) {}
    func start(framer _: NWProtocolFramer.Instance) -> NWProtocolFramer.StartResult { .ready }

    func handleInput(framer: NWProtocolFramer.Instance) -> Int {
        while true {
            var header: Header?
            let parsed = framer.parseInput(minimumIncompleteLength: Header.dataSize, maximumLength: Header.dataSize) { buffer, _ in
                guard let buffer, buffer.count >= Header.dataSize else { return 0 }
                header = Header(buffer)
                return Header.dataSize
            }

            guard parsed, let header else {
                return Header.dataSize
            }

            let message = NWProtocolFramer.Message(command: header.command)
            if !framer.deliverInputNoCopy(length: Int(header.length), message: message, isComplete: true) {
                return 0
            }
        }
    }

    func handleOutput(framer: NWProtocolFramer.Instance, message: NWProtocolFramer.Message, messageLength: Int, isComplete _: Bool) {
        guard let command = message.command else {
            return
        }
        let header = Header(command: command, length: UInt64(messageLength))
        framer.writeOutput(data: header.data)
        do {
            try framer.writeOutputNoCopy(length: messageLength)
        } catch {
            log("Error writing network message: \(error.localizedDescription)")
        }
    }
}
