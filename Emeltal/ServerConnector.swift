import Foundation
import Network

final class ServerConnector: EmeltalConnector {
    private var listener: NWListener?

    override func shutdown() {
        log("ServerConnector shutdown")
        listener?.cancel()
        listener = nil
        super.shutdown()
    }

    deinit {
        log("ServerConnector deinit")
    }

    func startServer() -> AsyncStream<Nibble> {
        let service = NWListener.Service(name: "Emeltal", type: "_emeltal._tcp", domain: nil)
        listener = try! NWListener(service: service, using: EmeltalProtocol.params)

        let (inputStream, continuation) = AsyncStream.makeStream(of: Nibble.self, bufferingPolicy: .unbounded)
        let queue = networkQueue

        listener?.newConnectionHandler = { [weak self] connection in
            connection.stateUpdateHandler = { [weak self] change in
                Task { @NetworkActor [weak self] in
                    guard let self else { return }

                    switch change {
                    case .cancelled:
                        self.state = .unConnected

                    case .ready:
                        self.connectionEstablished(connection, continuation: continuation)

                    case .preparing, .setup:
                        break

                    case let .failed(error), let .waiting(error):
                        if case let .connected(nWConnection) = self.state, connection !== nWConnection {
                            break
                        }
                        self.state = .error(error)

                    @unknown default:
                        break
                    }
                }
            }
            connection.start(queue: queue)
        }
        listener?.stateUpdateHandler = { state in
            switch state {
            case .cancelled:
                continuation.finish()
                log("Server listener cancelled")

            case .failed, .ready, .setup, .waiting:
                break

            @unknown default:
                break
            }
        }
        listener?.start(queue: queue)
        return inputStream
    }
}
