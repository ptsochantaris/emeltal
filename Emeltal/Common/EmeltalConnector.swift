import Foundation
import Network

@globalActor
actor NetworkActor {
    static let shared = NetworkActor()
}

@NetworkActor
final class EmeltalConnector {
    private static let networkQueue = DispatchQueue(label: "build.bru.emeltal.connector.network-queue")

    enum Payload: UInt64 {
        case unknown = 0, generatedSentence, appMode, recordedSpeech, recordedSpeechLast
    }

    nonisolated init() {}

    struct Header {
        let payload: Payload
        let length: UInt64

        static let uint64size = MemoryLayout<UInt64>.size
        static let dataSize = uint64size * 2

        var data: Data {
            var payload = payload.rawValue
            var length = length
            return Data(bytes: &payload, count: Self.uint64size)
                + Data(bytes: &length, count: Self.uint64size)
        }

        init(payload: Payload, length: Int) {
            self.payload = payload
            self.length = UInt64(length)
        }

        init(_ buffer: UnsafeMutableRawBufferPointer) {
            let tempPayload = buffer.load(as: UInt64.self)
            payload = Payload(rawValue: tempPayload) ?? .unknown
            length = buffer.load(fromByteOffset: Self.uint64size, as: UInt64.self)
        }
    }

    struct Nibble {
        let payload: Payload
        let data: Data?
    }

    enum State {
        case boot, searching, connecting, unConnected, connected(NWConnection), error(Error)
    }

    private let (inputStream, continuation) = AsyncStream.makeStream(of: Nibble.self, bufferingPolicy: .unbounded)

    var state = State.boot {
        didSet {
            log("New network state: \(state)")
        }
    }

    private func update(from browser: NWBrowser, connection: NWConnection, connectionState: NWConnection.State) {
        switch connectionState {
        case .preparing, .setup:
            break

        case .cancelled:
            state = .searching
            browser.start(queue: Self.networkQueue)

        case .ready:
            connectionEstablished(connection)

        case let .waiting(error):
            state = .error(error)

        case let .failed(error):
            state = .error(error)
            state = .searching
            browser.start(queue: Self.networkQueue)

        @unknown default:
            break
        }
    }

    private func update(from browser: NWBrowser, results: Set<NWBrowser.Result>) {
        guard case .searching = state, let result = results.first else {
            return
        }

        state = .connecting
        browser.cancel()

        let connection = NWConnection(to: result.endpoint, using: LinkProtocol.params)
        connection.stateUpdateHandler = { [weak self] newState in
            Task { [weak self] in
                guard let self else { return }
                await update(from: browser, connection: connection, connectionState: newState)
            }
        }
        connection.start(queue: Self.networkQueue)
    }

    private func update(from connection: NWConnection, to update: NWConnection.State) {
        switch update {
        case .cancelled:
            state = .unConnected

        case .ready:
            connectionEstablished(connection)

        case .preparing, .setup:
            break

        case let .failed(error), let .waiting(error):
            state = .error(error)

        @unknown default:
            break
        }
    }

    func startServer() -> AsyncStream<Nibble> {
        let service = NWListener.Service(name: "Emeltal", type: "_emeltal._tcp", domain: nil)
        let listener = try! NWListener(service: service, using: LinkProtocol.params)
        listener.newConnectionHandler = { connection in
            connection.stateUpdateHandler = { [weak self] change in
                Task { [weak self] in
                    guard let self else { return }
                    await update(from: connection, to: change)
                }
            }
            connection.start(queue: Self.networkQueue)
        }
        listener.start(queue: Self.networkQueue)
        return inputStream
    }

    private func connectionEstablished(_ connection: NWConnection) {
        state = .connected(connection)
        receive(connection: connection)
    }

    func startClient() -> AsyncStream<Nibble> {
        state = .searching
        let emeltalTcp = NWBrowser.Descriptor.bonjour(type: "_emeltal._tcp", domain: nil)
        let browser = NWBrowser(for: emeltalTcp, using: LinkProtocol.params)
        browser.browseResultsChangedHandler = { [weak self] results, _ in
            Task { [weak self] in
                guard let self else { return }
                await update(from: browser, results: results)
            }
        }
        browser.start(queue: Self.networkQueue)
        return inputStream
    }

    private func receive(connection: NWConnection) {
        connection.receiveMessage { [weak self] content, contentContext, isComplete, error in
            guard let self else { return }

            if let error {
                connection.cancel()
                log("Receiving error: \(error.localizedDescription)")
                return
            }

            if isComplete,
               let content,
               let msg = contentContext?.protocolMetadata(definition: LinkProtocol.definition) as? NWProtocolFramer.Message,
               let header = msg["EmeltalHeader"] as? EmeltalConnector.Header {
                continuation.yield(Nibble(payload: header.payload, data: content))
            }
            Task { [weak self] in
                guard let self else { return }
                await receive(connection: connection)
            }
        }
    }

    func send(_ payload: Payload, content: Data?) {
        guard case let .connected(nWConnection) = state else {
            return
        }

        let message = NWProtocolFramer.Message(definition: LinkProtocol.definition)
        message["EmeltalHeader"] = Header(payload: payload, length: content?.count ?? 0)

        let context = NWConnection.ContentContext(identifier: "Emeltal", metadata: [message])
        nWConnection.send(content: content, contentContext: context, isComplete: true, completion: .idempotent)
    }

    private final class LinkProtocol: NWProtocolFramerImplementation {
        static var params: NWParameters {
            let options = NWProtocolFramer.Options(definition: definition)
            let parameters = NWParameters.applicationService
            parameters.defaultProtocolStack.applicationProtocols.insert(options, at: 0)
            return parameters
        }

        static let label = "Emeltal Link"
        static let definition = NWProtocolFramer.Definition(implementation: LinkProtocol.self)

        init(framer _: NWProtocolFramer.Instance) {}
        func wakeup(framer _: NWProtocolFramer.Instance) {}
        func stop(framer _: NWProtocolFramer.Instance) -> Bool { true }
        func cleanup(framer _: NWProtocolFramer.Instance) {}
        func start(framer _: NWProtocolFramer.Instance) -> NWProtocolFramer.StartResult { .ready }

        func handleInput(framer: NWProtocolFramer.Instance) -> Int {
            while true {
                var header: EmeltalConnector.Header?
                let dataSize = EmeltalConnector.Header.dataSize
                let parsed = framer.parseInput(minimumIncompleteLength: dataSize, maximumLength: dataSize) { buffer, _ in
                    guard let buffer, buffer.count >= dataSize else { return 0 }
                    header = EmeltalConnector.Header(buffer)
                    return dataSize
                }

                guard parsed, let header else {
                    return dataSize
                }

                let message = NWProtocolFramer.Message(definition: Self.definition)
                message["EmeltalHeader"] = header
                if !framer.deliverInputNoCopy(length: Int(header.length), message: message, isComplete: true) {
                    return 0
                }
            }
        }

        func handleOutput(framer: NWProtocolFramer.Instance, message: NWProtocolFramer.Message, messageLength: Int, isComplete _: Bool) {
            guard let header = message["EmeltalHeader"] as? EmeltalConnector.Header else {
                return
            }
            framer.writeOutput(data: header.data)
            do {
                try framer.writeOutputNoCopy(length: messageLength)
            } catch {
                log("Error writing network message: \(error.localizedDescription)")
            }
        }
    }
}
