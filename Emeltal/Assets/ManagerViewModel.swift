import Foundation

@MainActor
@Observable
final class ManagerViewModel {
    static let shared = ManagerViewModel()

    let whisper = AssetFetcher(fetching: Model(category: .system, variant: .whisper))

    var selected: Model {
        didSet {
            Persisted.selectedAssetId = selected.id
        }
    }

    let models = [
        Model(category: .dolphin, variant: .dolphinMixtral),
        Model(category: .dolphin, variant: .dolphin70b),
        Model(category: .dolphin, variant: .dolphinTiny),
        Model(category: .creative, variant: .mythoMax),
        Model(category: .creative, variant: .neuralStory7b),
        Model(category: .coding, variant: .codestral),
        Model(category: .coding, variant: .dolphinCoder),
        Model(category: .coding, variant: .deepSeekCoder7),
        Model(category: .coding, variant: .codeLlama70b),
        Model(category: .coding, variant: .everyoneCoder),
        Model(category: .samantha, variant: .samantha7b),
        Model(category: .samantha, variant: .samantha70b),
        Model(category: .llamas, variant: .llama3),
        Model(category: .llamas, variant: .llama3large),
        Model(category: .llamas, variant: .llama3compact),
        Model(category: .llamas, variant: .llama3tiny),
        Model(category: .qwen, variant: .qwen2regular),
        Model(category: .qwen, variant: .qwen2large),
        Model(category: .qwen, variant: .qwen2small),
        Model(category: .experimental, variant: .supernovaMedius),
        Model(category: .system, variant: .whisper)
    ]

    init() {
        selected = if let id = Persisted.selectedAssetId {
            models.first { $0.id == id } ?? models.first!
        } else {
            models.first!
        }
    }

    func models(for category: Model.Category) -> [Model] {
        #if os(macOS)
            models.filter { $0.category == category }
        #else
            models.filter { $0.category == category && (!$0.variant.macOnly) }
        #endif
    }

    func category(for variant: Model.Variant) -> Model.Category? {
        models.first { $0.variant == variant }?.category
    }

    func cleanupNonInstalledAssets() {
        log("Checking for stale assets…")
        let modelsDir = Model.modelsDir
        let appDocumentsUrl = appDocumentsUrl

        let modelPaths = Set(models.map(\.localModelPath)).map(\.lastPathComponent)
        let fm = FileManager.default
        let modelDirs = (try? fm.contentsOfDirectory(at: modelsDir, includingPropertiesForKeys: nil)) ?? []
        let onDiskModelPaths = Set(modelDirs.map(\.lastPathComponent).filter { !$0.hasPrefix(".") })
        let unusedFiles = onDiskModelPaths.subtracting(modelPaths)
        for file in unusedFiles {
            log("Removing stale unused model file: \(file)")
            try? fm.removeItem(at: modelsDir.appendingPathComponent(file))
        }

        let statePaths = Set(models.map(\.localStatePath)).map(\.lastPathComponent)
        let stateDirs = (try? fm.contentsOfDirectory(at: appDocumentsUrl, includingPropertiesForKeys: nil)) ?? []
        let onDiskStatePaths = Set(stateDirs.map(\.lastPathComponent).filter { $0.hasPrefix("states-") })
        let unusedStateDirs = onDiskStatePaths.subtracting(statePaths)
        for file in unusedStateDirs {
            log("Removing stale state dir: \(file)")
            try? fm.removeItem(at: appDocumentsUrl.appendingPathComponent(file))
        }
    }
}
