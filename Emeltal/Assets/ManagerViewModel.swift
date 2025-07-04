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
        Model(category: .dolphin, variant: .dolphinThreeR1),
        Model(category: .dolphin, variant: .dolphinThree8b),
        Model(category: .dolphin, variant: .dolphinThree3b),
        Model(category: .dolphin, variant: .dolphinThreeTiny),
        Model(category: .dolphin, variant: .dolphinNemo),
        Model(category: .dolphin, variant: .dolphin72b),

        Model(category: .creative, variant: .mythoMax),
        Model(category: .creative, variant: .neuralStory7b),

        Model(category: .coding, variant: .qwen25coder),
        Model(category: .coding, variant: .codestral),
        Model(category: .coding, variant: .dolphinCoder),
        Model(category: .coding, variant: .deepSeekCoder7),
        Model(category: .coding, variant: .codeLlama70b),
        Model(category: .coding, variant: .olympicCoder),
        Model(category: .coding, variant: .everyoneCoder),

        Model(category: .samantha, variant: .samantha7b),
        Model(category: .samantha, variant: .samantha70b),

        Model(category: .llamas, variant: .llama4scout),
        Model(category: .llamas, variant: .llama3),
        Model(category: .llamas, variant: .llama3large),
        Model(category: .llamas, variant: .llama3compact),
        Model(category: .llamas, variant: .llama3tiny),
        Model(category: .llamas, variant: .llamaNemotron),

        Model(category: .qwen, variant: .qwen3compact),
        Model(category: .qwen, variant: .qwen3regular),
        Model(category: .qwen, variant: .qwen3tiny),
        Model(category: .qwen, variant: .qwen25regular),
        Model(category: .qwen, variant: .qwen25large),
        Model(category: .qwen, variant: .qwen25medium),
        Model(category: .qwen, variant: .qwen25small),
        Model(category: .qwen, variant: .qwenQwQ32),

        Model(category: .gemma, variant: .gemma327),
        Model(category: .gemma, variant: .gemma312),
        Model(category: .gemma, variant: .gemma34),
        Model(category: .gemma, variant: .gemma31),

        Model(category: .glm, variant: .glm4),
        Model(category: .glm, variant: .glmz1),

        Model(category: .apple, variant: .sage),

        Model(category: .experimental, variant: .am1),
        Model(category: .experimental, variant: .magistral),
        Model(category: .experimental, variant: .mistral2503),
        Model(category: .experimental, variant: .dsro70),
        Model(category: .experimental, variant: .athene),
        Model(category: .experimental, variant: .supernovaMedius),
        Model(category: .experimental, variant: .smol),
        Model(category: .experimental, variant: .shuttle),
        Model(category: .experimental, variant: .calme),

        Model(category: .system, variant: .whisper)
    ]

    init() {
        selected = if let id = Persisted.selectedAssetId {
            models.first { $0.id == id } ?? models[0]
        } else {
            models[0]
        }
    }

    func models(for category: Model.Category) -> [Model] {
        #if os(macOS)
            models.filter { $0.category == category }
        #else
            models.filter { $0.category == category && $0.variant.memoryEstimate.excessBytes == 0 }
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

        /*
         Task {
             // TODO: remove!
             for model in models.sorted(by: { $0.memoryEstimate.gpuUsageEstimateBytes < $1.memoryEstimate.gpuUsageEstimateBytes }) where model.category.displayable {
                 await model.sanityCheckEstimates(whisper: whisper)
             }
         }
          */
    }
}
