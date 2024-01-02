import Combine
import Foundation
import Network
import PopTimer
import SwiftUI

@globalActor
actor NetworkActor {
    static let shared = NetworkActor()
}

@NetworkActor
final class EmeltalConnector {
    private static let networkQueue = DispatchQueue(label: "build.bru.emeltal.connector.network-queue")

    enum Payload: UInt64 {
        case unknown = 0, spokenSentence, appMode, recordedSpeech, recordedSpeechDone, toggleListeningMode, buttonDown, appActivationState, heartbeat, textInitial, textDiff, textInput, hello, requestReset
    }

    private lazy var popTimer = PopTimer(timeInterval: 4) { [weak self] in
        Task { @NetworkActor [weak self] in
            guard let self else { return }
            send(.heartbeat, content: emptyData)
        }
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
            var tempPayload: UInt64 = 0
            var tempLength: UInt64 = 0
            withUnsafeMutableBytes(of: &tempPayload) {
                $0.copyMemory(from: UnsafeRawBufferPointer(start: buffer.baseAddress!, count: Self.uint64size))
            }
            withUnsafeMutableBytes(of: &tempLength) {
                $0.copyMemory(from: UnsafeRawBufferPointer(start: buffer.baseAddress!.advanced(by: Self.uint64size), count: Self.uint64size))
            }
            payload = Payload(rawValue: tempPayload) ?? .unknown
            length = tempLength
        }
    }

    struct Nibble {
        let payload: Payload
        let data: Data?
    }

    enum State {
        case boot, searching, connecting, unConnected, connected(NWConnection), error(Error)

        var isConnected: Bool {
            switch self {
            case .boot, .connecting, .error, .searching, .unConnected:
                false
            case .connected:
                true
            }
        }

        var isConnectionActive: Bool {
            switch self {
            case .boot, .error, .unConnected:
                false
            case .connected, .connecting, .searching:
                true
            }
        }

        var label: String {
            switch self {
            case .boot, .unConnected: "Starting"
            case .connecting: "Connecting"
            case .connected: "Connected"
            case .error: "Error"
            case .searching: "Searching"
            }
        }

        var color: Color {
            switch self {
            case .boot, .unConnected: .gray
            case .connecting: .yellow
            case .connected: .green
            case .error: .red
            case .searching: .yellow
            }
        }
    }

    private var state = State.boot {
        didSet {
            statePublisher.send(state)
            log("[Connector] State: \(state)")
            if case .connected = state {
                popTimer.push()
            }
        }
    }

    let statePublisher = CurrentValueSubject<State, Never>(State.boot)

    func invalidate() {
        if case let .connected(nWConnection) = state {
            nWConnection.cancel()
        }
    }

    func startServer() -> AsyncStream<Nibble> {
        let (inputStream, continuation) = AsyncStream.makeStream(of: Nibble.self, bufferingPolicy: .unbounded)

        let service = NWListener.Service(name: "Emeltal", type: "_emeltal._tcp", domain: nil)
        let listener = try! NWListener(service: service, using: LinkProtocol.params)
        listener.newConnectionHandler = { connection in
            connection.stateUpdateHandler = { [weak self] change in
                Task { @NetworkActor [weak self] in
                    guard let self else { return }

                    switch change {
                    case .cancelled:
                        state = .unConnected

                    case .ready:
                        connectionEstablished(connection, continuation: continuation)

                    case .preparing, .setup:
                        break

                    case let .failed(error), let .waiting(error):
                        if case let .connected(nWConnection) = state, connection !== nWConnection {
                            break
                        }
                        state = .error(error)

                    @unknown default:
                        break
                    }
                }
            }
            connection.start(queue: Self.networkQueue)
        }
        listener.start(queue: Self.networkQueue)
        return inputStream
    }

    private func connectionEstablished(_ connection: NWConnection, continuation: AsyncStream<Nibble>.Continuation) {
        state = .connected(connection)
        send(.hello, content: emptyData)
        receive(connection: connection, continuation: continuation)
    }

    func startClient() -> AsyncStream<Nibble> {
        let (inputStream, continuation) = AsyncStream.makeStream(of: Nibble.self, bufferingPolicy: .unbounded)

        state = .searching
        let emeltalTcp = NWBrowser.Descriptor.bonjour(type: "_emeltal._tcp", domain: nil)
        let browser = NWBrowser(for: emeltalTcp, using: LinkProtocol.params)
        browser.browseResultsChangedHandler = { [weak self] results, _ in
            Task { @NetworkActor [weak self] in
                guard let self, case .searching = state, let result = results.first else { return }

                state = .connecting
                browser.cancel()

                let connection = NWConnection(to: result.endpoint, using: LinkProtocol.params)
                connection.stateUpdateHandler = { [weak self] newState in
                    Task { @NetworkActor [weak self] in
                        guard let self else { return }

                        switch newState {
                        case .preparing, .setup:
                            break

                        case .cancelled:
                            state = .searching

                        case .ready:
                            connectionEstablished(connection, continuation: continuation)

                        case let .waiting(error):
                            state = .error(error)

                        case let .failed(error):
                            state = .error(error)
                            _ = startClient()

                        @unknown default:
                            break
                        }
                    }
                }
                connection.start(queue: Self.networkQueue)
            }
        }
        browser.start(queue: Self.networkQueue)
        return inputStream
    }

    private func receive(connection: NWConnection, continuation: AsyncStream<Nibble>.Continuation) {
        connection.receiveMessage { [weak self] content, contentContext, isComplete, error in
            guard let self else { return }

            Task { @NetworkActor [weak self] in
                guard let self else { return }
                popTimer.push()
            }

            if let error {
                if case .cancelled = connection.state {
                    continuation.finish()
                } else {
                    connection.cancel()
                    log("[Connector] Receiving error: \(error.localizedDescription)")
                }
                return
            }

            if isComplete,
               let content,
               let msg = contentContext?.protocolMetadata(definition: LinkProtocol.definition) as? NWProtocolFramer.Message,
               let header = msg["EmeltalHeader"] as? EmeltalConnector.Header {
                continuation.yield(Nibble(payload: header.payload, data: content))
            }

            Task { @NetworkActor [weak self] in
                guard let self else { return }
                receive(connection: connection, continuation: continuation)
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

        log("[Connector] Did send \(payload) - \(content?.count ?? 0) bytes")
        popTimer.push()
    }

    private final class LinkProtocol: NWProtocolFramerImplementation {
        static var params: NWParameters {
            let options = NWProtocolFramer.Options(definition: definition)
            let tcpOptions = NWProtocolTCP.Options()
            tcpOptions.keepaliveCount = 1
            tcpOptions.keepaliveInterval = 1
            tcpOptions.keepaliveIdle = 1
            let parameters = NWParameters(tls: nil, tcp: tcpOptions)
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
                log("[Connector] Error writing network message: \(error.localizedDescription)")
            }
        }
    }
}
