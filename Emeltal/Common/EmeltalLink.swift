import Foundation
import Network

extension NWProtocolFramer.Message {
    convenience init(transmission: EmeltalConnector.Transmission) {
        self.init(definition: EmeltalLink.definition)
        self["EmeltalTransmission"] = transmission
    }

    var transmission: EmeltalConnector.Transmission? {
        self["EmeltalTransmission"] as? EmeltalConnector.Transmission
    }
}

final class EmeltalLink: NWProtocolFramerImplementation {
    static var params: NWParameters {
        let options = NWProtocolFramer.Options(definition: definition)
        let parameters = NWParameters.applicationService
        parameters.defaultProtocolStack.applicationProtocols.insert(options, at: 0)
        return parameters
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
            var transmission: EmeltalConnector.Transmission?
            let dataSize = EmeltalConnector.Transmission.dataSize
            let parsed = framer.parseInput(minimumIncompleteLength: dataSize, maximumLength: dataSize) { buffer, _ in
                guard let buffer, buffer.count >= dataSize else { return 0 }
                transmission = EmeltalConnector.Transmission(buffer)
                return dataSize
            }

            guard parsed, let transmission else {
                return dataSize
            }

            let message = NWProtocolFramer.Message(transmission: transmission)
            if !framer.deliverInputNoCopy(length: Int(transmission.length), message: message, isComplete: true) {
                return 0
            }
        }
    }

    func handleOutput(framer: NWProtocolFramer.Instance, message: NWProtocolFramer.Message, messageLength: Int, isComplete _: Bool) {
        guard let transmission = message.transmission else {
            return
        }
        framer.writeOutput(data: transmission.data)
        do {
            try framer.writeOutputNoCopy(length: messageLength)
        } catch {
            log("Error writing network message: \(error.localizedDescription)")
        }
    }
}
