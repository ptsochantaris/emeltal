import Foundation
import Network

final class ClientConnector: EmeltalConnector {
    func startClient() -> AsyncStream<Nibble> {
        let (inputStream, continuation) = AsyncStream.makeStream(of: Nibble.self, bufferingPolicy: .unbounded)

        state = .searching
        let emeltalTcp = NWBrowser.Descriptor.bonjour(type: "_emeltal._tcp", domain: nil)
        let browser = NWBrowser(for: emeltalTcp, using: EmeltalProtocol.params)
        browser.browseResultsChangedHandler = { [weak self] results, _ in
            Task { @NetworkActor [weak self] in
                guard let self, case .searching = state, let result = results.first else { return }

                state = .connecting
                browser.cancel()

                let connection = NWConnection(to: result.endpoint, using: EmeltalProtocol.params)
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
}
