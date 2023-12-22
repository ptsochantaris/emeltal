import Foundation
import Network

final class EmeltalConnector {
    struct Transmission {
        enum Payload: UInt64 {
            case unknown = 0, generatedSentence, appMode, recordedSpeech, recordedSpeechLast
        }

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

        init(payload: Payload, length: UInt64) {
            self.payload = payload
            self.length = length
        }

        init(_ buffer: UnsafeMutableRawBufferPointer) {
            let tempPayload = buffer.load(as: UInt64.self)
            payload = Payload(rawValue: tempPayload) ?? .unknown
            length = buffer.load(fromByteOffset: Self.uint64size, as: UInt64.self)
        }
    }

    enum State {
        case boot, searching, connecting, unConnected, connected(NWConnection), error(Error)
    }

    private let (inputStream, continuation) = AsyncStream.makeStream(of: (Transmission.Payload, Data?).self, bufferingPolicy: .unbounded)

    var state = State.boot {
        didSet {
            log("New network state: \(state)")
        }
    }

    func setupNetworkAdvertiser() -> AsyncStream<(Transmission.Payload, Data?)> {
        let service = NWListener.Service(name: "Emeltal", type: "_emeltal._tcp", domain: nil)
        let listener = try! NWListener(service: service, using: EmeltalLink.params)
        listener.newConnectionHandler = { connection in
            connection.stateUpdateHandler = { update in
                Task { @MainActor [weak self] in
                    guard let self else { return }

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
            }
            connection.start(queue: .main)
        }
        listener.start(queue: .main)
        return inputStream
    }

    private func connectionEstablished(_ connection: NWConnection) {
        state = .connected(connection)
        receive(connection: connection)
    }

    func setupNetworkListener() -> AsyncStream<(Transmission.Payload, Data?)> {
        state = .searching
        let emeltalTcp = NWBrowser.Descriptor.bonjour(type: "_emeltal._tcp", domain: nil)
        let browser = NWBrowser(for: emeltalTcp, using: EmeltalLink.params)
        browser.browseResultsChangedHandler = { [weak self] results, _ in
            if let self, case .searching = state, let result = results.first {
                state = .connecting
                foundServer(on: browser, result: result)
                browser.cancel()
            }
        }
        browser.start(queue: .main)
        return inputStream
    }

    private func foundServer(on browser: NWBrowser, result: NWBrowser.Result) {
        let connection = NWConnection(to: result.endpoint, using: EmeltalLink.params)
        connection.stateUpdateHandler = { connectionState in
            Task { @MainActor [weak self] in
                guard let self else { return }

                switch connectionState {
                case .preparing, .setup:
                    break

                case .cancelled:
                    state = .searching
                    browser.start(queue: .main)

                case .ready:
                    connectionEstablished(connection)

                case let .waiting(error):
                    state = .error(error)

                case let .failed(error):
                    state = .error(error)
                    state = .searching
                    browser.start(queue: .main)

                @unknown default:
                    break
                }
            }
        }
        connection.start(queue: .main)
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
               let msg = contentContext?.protocolMetadata(definition: EmeltalLink.definition) as? NWProtocolFramer.Message,
               let payload = msg.transmission?.payload {
                continuation.yield((payload, content))
            }
            receive(connection: connection)
        }
    }

    func send(_ payload: Transmission.Payload, content: Data?) {
        guard case let .connected(nWConnection) = state else {
            return
        }

        let transmission = Transmission(payload: payload, length: UInt64(content?.count ?? 0))

        let context = NWConnection.ContentContext(identifier: "Emeltal", metadata: [
            NWProtocolFramer.Message(transmission: transmission)
        ])

        nWConnection.send(content: content,
                          contentContext: context,
                          isComplete: true,
                          completion: .idempotent)
    }
}
