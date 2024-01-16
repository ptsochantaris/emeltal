import Foundation

final class Turn: Codable {
    let id: llama_seq_id
    var length = 0

    enum CodingKeys: CodingKey {
        case id
        case length
    }

    init(id: llama_seq_id) {
        self.id = id
    }

    func append(tokens: [llama_token], in context: OpaquePointer, andPredict: Bool, offset: Int) -> UnsafeMutablePointer<Float>? {
        let promptCount = tokens.count
        // print("Seq ID \(id): \(promptCount) tokens, offset: \(offset): ", terminator: "")
        var b = llama_batch_init(Int32(promptCount), 0, 1)
        defer {
            length += tokens.count
            llama_batch_free(b)
        }
        b.n_tokens = Int32(promptCount)
        for (i, token) in tokens.enumerated() {
            let pos = Int32(i + offset)
            // print("[\(pos): \(token)] ", terminator: "")
            b.token[i] = token
            b.pos[i] = pos
            b.n_seq_id[i] = 1
            b.seq_id[i]![0] = 0
            b.logits[i] = 0
        }
        // print()
        if andPredict {
            b.logits[promptCount - 1] = 1
            llama_decode(context, b)
            return llama_get_logits_ith(context, Int32(promptCount) - 1)
        } else {
            llama_decode(context, b)
            return nil
        }
    }

    private static let singleTokenBatch: llama_batch = {
        var b = llama_batch_init(1, 0, 1)
        b.n_tokens = 1
        b.n_seq_id[0] = 1
        b.seq_id[0]![0] = 0
        b.logits[0] = 1
        return b
    }()

    func appendAndPredict(token: llama_token, in context: OpaquePointer, pos: Int) -> UnsafeMutablePointer<Float> {
        // print("+[\(pos): \(token)] ", terminator: "")
        Self.singleTokenBatch.token[0] = token
        Self.singleTokenBatch.pos[0] = Int32(pos)
        llama_decode(context, Self.singleTokenBatch)
        length += 1
        return llama_get_logits_ith(context, 0)!
    }
}
