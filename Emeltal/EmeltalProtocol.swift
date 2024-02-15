import CryptoKit
import Foundation
import Network

final class EmeltalProtocol: NWProtocolFramerImplementation {
    static var params: NWParameters {
        let tcpOptions = NWProtocolTCP.Options()
        tcpOptions.keepaliveCount = 1
        tcpOptions.keepaliveInterval = 1
        tcpOptions.keepaliveIdle = 1

        let authenticationKey = SymmetricKey(data: "NotARealSecret".data(using: .utf8)!)
        let labelData = label.data(using: .utf8)!
        let authenticationCode = HMAC<SHA256>.authenticationCode(for: labelData, using: authenticationKey)
        let authenticationDispatchData = authenticationCode.withUnsafeBytes { DispatchData(bytes: $0) }
        let dispatchData = labelData.withUnsafeBytes { DispatchData(bytes: $0) }

        let tlsOptions = NWProtocolTLS.Options()
        sec_protocol_options_add_pre_shared_key(tlsOptions.securityProtocolOptions, authenticationDispatchData as __DispatchData, dispatchData as __DispatchData)
        sec_protocol_options_append_tls_ciphersuite(tlsOptions.securityProtocolOptions, tls_ciphersuite_t(rawValue: UInt16(TLS_PSK_WITH_AES_128_GCM_SHA256))!)

        let parameters = NWParameters(tls: tlsOptions, tcp: tcpOptions)
        parameters.includePeerToPeer = true

        let options = NWProtocolFramer.Options(definition: definition)
        parameters.defaultProtocolStack.applicationProtocols.insert(options, at: 0)

        return parameters
    }

    static let label = "Emeltal Link"
    static let definition = NWProtocolFramer.Definition(implementation: EmeltalProtocol.self)

    init(framer _: NWProtocolFramer.Instance) {}
    func wakeup(framer _: NWProtocolFramer.Instance) {}
    func stop(framer _: NWProtocolFramer.Instance) -> Bool { true }
    func cleanup(framer _: NWProtocolFramer.Instance) {}
    func start(framer _: NWProtocolFramer.Instance) -> NWProtocolFramer.StartResult { .ready }

    func handleInput(framer: NWProtocolFramer.Instance) -> Int {
        while true {
            var header: EmeltalConnector.Header?
            let dataSize = EmeltalConnector.Header.dataSize
            let parsed = framer.parseInput(minimumIncompleteLength: dataSize, maximumLength: dataSize) { buffer, _ in
                guard let buffer, buffer.count >= dataSize else { return 0 }
                header = EmeltalConnector.Header(buffer)
                return dataSize
            }

            guard parsed, let header else {
                return dataSize
            }

            let message = NWProtocolFramer.Message(definition: Self.definition)
            message["EmeltalHeader"] = header
            if !framer.deliverInputNoCopy(length: Int(header.length), message: message, isComplete: true) {
                return 0
            }
        }
    }

    func handleOutput(framer: NWProtocolFramer.Instance, message: NWProtocolFramer.Message, messageLength: Int, isComplete _: Bool) {
        guard let header = message["EmeltalHeader"] as? EmeltalConnector.Header else {
            return
        }
        framer.writeOutput(data: header.data)
        do {
            try framer.writeOutputNoCopy(length: messageLength)
        } catch {
            log("[Connector] Error writing network message: \(error.localizedDescription)")
        }
    }

    deinit {
        log("EmeltalProtocol deinit")
    }
}
