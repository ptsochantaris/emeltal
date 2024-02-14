import Foundation

@globalActor actor GGMLActor {
    static let shared = GGMLActor()

    private final class Executor: SerialExecutor {
        private let ggmlQueue = DispatchQueue(label: "build.bru.emeltal.ggml", qos: .userInitiated)

        func enqueue(_ job: consuming ExecutorJob) {
            let j = UnownedJob(job)
            let e = asUnownedSerialExecutor()
            ggmlQueue.async {
                j.runSynchronously(on: e)
            }
        }
    }

    private static let executor = Executor()
    static let sharedUnownedExecutor = executor.asUnownedSerialExecutor()

    nonisolated var unownedExecutor: UnownedSerialExecutor {
        Self.sharedUnownedExecutor
    }
}
