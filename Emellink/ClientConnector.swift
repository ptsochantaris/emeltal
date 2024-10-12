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
            guard let self else { return }
            Task {
                await browserResultsChanged(results: results, continuation: continuation, queue: queue)
            }
        }
        browser?.start(queue: queue)
        return inputStream
    }

    @NetworkActor
    private func browserResultsChanged(results: Set<NWBrowser.Result>, continuation: AsyncStream<Nibble>.Continuation, queue: DispatchQueue) async {
        guard case .searching = state, let result = results.first else { return }

        state = .connecting
        browserDone()

        let connection = NWConnection(to: result.endpoint, using: EmeltalProtocol.params)
        connection.stateUpdateHandler = { [weak self] newState in
            guard let self else { return }
            Task {
                await connectionStateChanged(to: newState, connection: connection, continuation: continuation)
            }
        }
        connection.start(queue: queue)
    }

    @NetworkActor
    private func connectionStateChanged(to newState: NWConnection.State, connection: NWConnection, continuation: AsyncStream<Nibble>.Continuation) {
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
