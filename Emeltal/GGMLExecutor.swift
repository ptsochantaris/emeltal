import Foundation

private final class GGMLExecutor: SerialExecutor {
    static let shared = GGMLExecutor()

    private let ggmlQueue = DispatchQueue(label: "build.bru.emeltal.ggml", qos: .userInitiated)

    func enqueue(_ job: consuming ExecutorJob) {
        let j = UnownedJob(job)
        let e = asUnownedSerialExecutor()
        ggmlQueue.async {
            j.runSynchronously(on: e)
        }
    }
}

@globalActor actor GGMLActor {
    static var shared = GGMLActor()

    static var sharedUnownedExecutor = GGMLExecutor.shared.asUnownedSerialExecutor()

    nonisolated var unownedExecutor: UnownedSerialExecutor {
        Self.sharedUnownedExecutor
    }
}
