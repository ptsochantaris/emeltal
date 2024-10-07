import Foundation
import Network
import PopTimer
import SwiftUI

@globalActor
actor NetworkActor {
    static let shared = NetworkActor()
}

@NetworkActor
class EmeltalConnector {
    let networkQueue = DispatchQueue(label: "build.bru.emeltal.connector.network-queue")

    enum Payload: UInt64 {
        case unknown = 0, spokenSentence, appMode, recordedSpeech, recordedSpeechDone, toggleListeningMode, buttonTap, appActivationState, heartbeat, textInitial, textDiff, textInput, hello, requestReset, responseDone
    }

    struct Nibble {
        let payload: Payload
        let data: Data?
    }

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

    enum State: Sendable {
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

    final let stateStream = AsyncStream.makeStream(of: State.self, bufferingPolicy: .unbounded)

    nonisolated init() {}

    private lazy var popTimer = PopTimer(timeInterval: 4) { [weak self] in
        Task { @NetworkActor [weak self] in
            guard let self else { return }
            self.send(.heartbeat, content: emptyData)
        }
    }

    final var state = State.boot {
        didSet {
            stateStream.continuation.yield(state)
            log("[Connector] State: \(state)")
            if case .connected = state {
                popTimer.push()
            }
        }
    }

    func shutdown() {
        if case let .connected(nWConnection) = state {
            nWConnection.cancel()
        }
    }

    final func connectionEstablished(_ connection: NWConnection, continuation: AsyncStream<Nibble>.Continuation) {
        state = .connected(connection)
        send(.hello, content: emptyData)
        receive(connection: connection, continuation: continuation)
    }

    private final func receive(connection: NWConnection, continuation: AsyncStream<Nibble>.Continuation) {
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
               let msg = contentContext?.protocolMetadata(definition: EmeltalProtocol.definition) as? NWProtocolFramer.Message,
               let header = msg["EmeltalHeader"] as? EmeltalConnector.Header {
                continuation.yield(Nibble(payload: header.payload, data: content))
            }

            Task { @NetworkActor [weak self] in
                guard let self else { return }
                receive(connection: connection, continuation: continuation)
            }
        }
    }

    final func send(_ payload: Payload, content: Data?) {
        guard case let .connected(nWConnection) = state else {
            return
        }

        let message = NWProtocolFramer.Message(definition: EmeltalProtocol.definition)
        message["EmeltalHeader"] = Header(payload: payload, length: content?.count ?? 0)

        let context = NWConnection.ContentContext(identifier: "Emeltal", metadata: [message])
        nWConnection.send(content: content, contentContext: context, isComplete: true, completion: .idempotent)

        log("[Connector] Did send \(payload) - \(content?.count ?? 0) bytes")
        popTimer.push()
    }

    deinit {
        log("EmeltalConnector deinit")
    }
}
