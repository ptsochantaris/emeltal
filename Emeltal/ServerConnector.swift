import Foundation
import Network

final class ServerConnector: EmeltalConnector {
    func startServer() -> AsyncStream<Nibble> {
        let (inputStream, continuation) = AsyncStream.makeStream(of: Nibble.self, bufferingPolicy: .unbounded)

        let service = NWListener.Service(name: "Emeltal", type: "_emeltal._tcp", domain: nil)
        let listener = try! NWListener(service: service, using: EmeltalProtocol.params)
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
}
