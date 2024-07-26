import Foundation
import Network

final class ClientConnector: EmeltalConnector {
    private var browser: NWBrowser?

    private func browserDone() {
        browser?.cancel()
        browser = nil
    }

    override func shutdown() {
        log("ClientConnector shutdown")
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
        let queue = networkQueue
        browser?.browseResultsChangedHandler = { [weak self] results, _ in
            Task { @NetworkActor [weak self] in
                guard let self, case .searching = self.state, let result = results.first else { return }

                self.state = .connecting
                self.browserDone()

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
                            self.state = .searching

                        case .ready:
                            self.connectionEstablished(connection, continuation: continuation)

                        case let .waiting(error):
                            self.state = .error(error)

                        case .failed:
                            self.state = .searching

                        @unknown default:
                            break
                        }
                    }
                }
                connection.start(queue: queue)
            }
        }
        browser?.start(queue: queue)
        return inputStream
    }
}
