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

    @NetworkActor
    private func connectionStateChanged(to newState: NWConnection.State, connection: NWConnection, continuation: AsyncStream<Nibble>.Continuation) {
        switch newState {
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

    @NetworkActor
    private func listenerConnected(connection: NWConnection, continuation: AsyncStream<Nibble>.Continuation, queue: DispatchQueue) async {
        connection.stateUpdateHandler = { [weak self] change in
            guard let self else { return }
            Task {
                await connectionStateChanged(to: change, connection: connection, continuation: continuation)
            }
        }
        connection.start(queue: queue)
    }

    func startServer() -> AsyncStream<Nibble> {
        let service = NWListener.Service(name: "Emeltal", type: "_emeltal._tcp", domain: nil)
        listener = try? NWListener(service: service, using: EmeltalProtocol.params)

        let (inputStream, continuation) = AsyncStream.makeStream(of: Nibble.self, bufferingPolicy: .unbounded)
        let queue = networkQueue
        listener?.newConnectionHandler = { [weak self] connection in
            guard let self else { return }
            Task {
                await listenerConnected(connection: connection, continuation: continuation, queue: queue)
            }
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
