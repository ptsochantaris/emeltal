import Foundation
import Network

final class ClientConnector: EmeltalConnector {
    private var browser: NWBrowser?

    private func browserDone() {
        browser?.cancel()
        browser = nil
    }

    override func shutdown() {
        browserDone()
        super.shutdown()
    }

    deinit {
        log("ClientConnector deinit")
    }

    func startClient() -> AsyncStream<Nibble> {
        let emeltalTcp = NWBrowser.Descriptor.bonjour(type: "_emeltal._tcp", domain: nil)
        browser = NWBrowser(for: emeltalTcp, using: EmeltalProtocol.params)

        let (inputStream, continuation) = AsyncStream.makeStream(of: Nibble.self, bufferingPolicy: .unbounded)

        state = .searching
        browser?.browseResultsChangedHandler = { [weak self] results, _ in
            Task { @NetworkActor [weak self] in
                guard let self, case .searching = state, let result = results.first else { return }

                state = .connecting
                browserDone()

                let connection = NWConnection(to: result.endpoint, using: EmeltalProtocol.params)
                connection.stateUpdateHandler = { [weak self] newState in
                    Task { @NetworkActor [weak self] in
                        guard let self else { return }

                        switch newState {
                        case .preparing, .setup:
                            break

                        case .cancelled:
                            log("Client connection cancelled")
                            continuation.finish()
                            state = .searching

                        case .ready:
                            connectionEstablished(connection, continuation: continuation)

                        case let .waiting(error):
                            state = .error(error)

                        case .failed:
                            state = .searching

                        @unknown default:
                            break
                        }
                    }
                }
                connection.start(queue: Self.networkQueue)
            }
        }
        browser?.start(queue: Self.networkQueue)
        return inputStream
    }
}
