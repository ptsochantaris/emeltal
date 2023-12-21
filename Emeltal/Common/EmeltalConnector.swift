import Foundation
import Network

final class EmeltalConnector {
    enum Transmission {
        case speak(String)
        case hear(Data, Bool)
    }

    enum State {
        case boot, searching, connecting, unConnected, connected(NWConnection), error(Error)
    }

    private let (inputStream, continuation) = AsyncStream.makeStream(of: Transmission.self, bufferingPolicy: .unbounded)

    var state = State.boot {
        didSet {
            log("New network state: \(state)")
        }
    }

    func setupNetworkAdvertiser() -> AsyncStream<Transmission> {
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

    func setupNetworkListener() -> AsyncStream<Transmission> {
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

            if isComplete, let content, let msg = contentContext?.protocolMetadata(definition: EmeltalLink.definition) as? NWProtocolFramer.Message {
                switch msg.command {
                case .none, .unknown:
                    break
                case .audioBlock:
                    continuation.yield(.hear(content, false))
                case .audioBlockLast:
                    continuation.yield(.hear(content, true))
                case .utterance:
                    if let text = String(data: content, encoding: .utf8) {
                        continuation.yield(.speak(text))
                    }
                }
            }
            receive(connection: connection)
        }
    }

    func sendUtterance(_ sentence: String) {
        guard case let .connected(nWConnection) = state else {
            return
        }

        let context = NWConnection.ContentContext(identifier: "utterance", metadata: [
            NWProtocolFramer.Message(command: .utterance)
        ])

        nWConnection.send(content: sentence.data(using: .utf8),
                          contentContext: context,
                          isComplete: true,
                          completion: .idempotent)
    }
}
