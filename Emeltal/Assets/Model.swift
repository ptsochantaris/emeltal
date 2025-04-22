import Foundation
import PopTimer

@MainActor
@Observable
final class Model: Hashable, Identifiable, Sendable {
    let id: String
    let category: Category
    let variant: Variant

    private var saveTimer: PopTimer?

    var params: Params {
        didSet {
            saveTimer?.push()
        }
    }

    private(set) var status: Status

    static var modelsDir = appDocumentsUrl.appendingPathComponent("models", conformingTo: .directory)

    nonisolated static func == (lhs: Model, rhs: Model) -> Bool {
        lhs.id == rhs.id
    }

    nonisolated func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }

    init(category: Category, variant: Variant) {
        let myId = "\(category.id)-\(variant.id)"
        id = myId
        self.category = category
        self.variant = variant
        status = .checking

        if let modelParams = Persisted.modelParams, let list = try? JSONDecoder().decode([ParamsHolder].self, from: modelParams), let mine = list.first(where: { $0.modelId == myId }) {
            params = mine.params
        } else {
            params = variant.defaultParams
        }

        saveTimer = PopTimer(timeInterval: 0.1) { [weak self] in
            self?.save()
        }

        updateInstalledStatus()
    }

    func unInstall() {
        let fm = FileManager.default
        try? fm.removeItem(at: localModelPath)
        try? fm.removeItem(at: localStatePath)
        status = .checking
        updateInstalledStatus()
    }

    func updateInstalledStatus() {
        if FileManager.default.fileExists(atPath: localModelPath.path) {
            let fetcher = AssetFetcher(fetching: self)
            status = .installed(fetcher)
            return
        }

        let task = Task.detached { [weak self] in
            guard let self else { return Model.Status.notReady }
            log("Checking availability of \(variant.displayName)")

            var request = URLRequest(url: variant.fetchUrl)
            request.httpMethod = "head"
            let response = try? await URLSession.shared.data(for: request).1 as? HTTPURLResponse
            let newStatus: Model.Status = if let code = response?.statusCode, code >= 200, code < 300 {
                variant.recommended ? .recommended : .available
            } else {
                .notReady
            }
            return newStatus
        }

        Task {
            let newStatus = await task.value
            if !Task.isCancelled, status != newStatus {
                status = newStatus
                log("Status for \(variant.displayName) determined to be \(newStatus)")
            }
        }
    }

    var localModelPath: URL {
        let modelDir = Self.modelsDir
        let fm = FileManager.default
        if !fm.fileExists(atPath: modelDir.path) {
            try! fm.createDirectory(at: modelDir, withIntermediateDirectories: true)
        }
        return modelDir.appendingPathComponent(variant.fileName)
    }

    var localStatePath: URL {
        let fm = FileManager.default
        let statePath = appDocumentsUrl.appendingPathComponent("states-\(id)", conformingTo: .directory)
        if !fm.fileExists(atPath: statePath.path) {
            try? fm.createDirectory(at: statePath, withIntermediateDirectories: true)
        }
        return statePath
    }

    func mlTemplate(in context: LlamaContext) -> Template? {
        Template(format: variant.format,
                 system: params.systemPrompt,
                 bosToken: context.bosToken)
    }

    func resetToDefaults() {
        params = variant.defaultParams
    }

    func save() {
        var list = if let modelParams = Persisted.modelParams, let list = try? JSONDecoder().decode([ParamsHolder].self, from: modelParams) {
            list
        } else {
            [ParamsHolder]()
        }

        let myParams = ParamsHolder(modelId: id, params: params)
        if let index = list.firstIndex(where: { $0.modelId == myParams.modelId }) {
            list[index] = myParams
        } else {
            list.append(myParams)
        }
        Persisted.modelParams = try? JSONEncoder().encode(list)
        log("Saved params for model \(id)")
    }

    func cancelInstall() {
        guard case let .installing(fetcher) = status else { return }
        fetcher.cancel()
        status = .available
    }

    func install() {
        let fetcher = AssetFetcher(fetching: self)
        status = .installing(fetcher)
    }

    private var cachedMemoryEstimate: MemoryEstimate?
    var memoryEstimate: MemoryEstimate {
        if let cachedMemoryEstimate { return cachedMemoryEstimate }
        let calculated = variant.memoryEstimate
        cachedMemoryEstimate = calculated
        return calculated
    }

    func sanityCheckEstimates(whisper: AssetFetcher) async {
        guard case let .installed(fetcher) = status else {
            return
        }

        let estimated = memoryEstimate.gpuUsageEstimateBytes
        let estimatedString = memoryFormatter.string(fromByteCount: estimated)
        print(">> Model \(variant), estimated GPU usage: \(estimatedString)")

        let state = ConversationState(llm: fetcher, whisper: whisper)
        while state.statusMessage != nil {
            try? await Task.sleep(for: .seconds(0.2))
        }
        if let (_, used, _, _) = variant.memoryBytes {
            let usedString = memoryFormatter.string(fromByteCount: used)
            let diff = estimated - used
            let diffString = memoryFormatter.string(fromByteCount: diff)
            let warning: String = if diff < 200_000_000 {
                "** SMALL **"
            } else if diff > 700_000_000 {
                "** LARGE **"
            } else {
                ""
            }
            print(">> Model \(variant), actual: \(usedString), diff: \(diffString) \(warning)")
            print(">> -----------------------------------------------------------------------------")
        }

        await state.shutdown()
    }
}
