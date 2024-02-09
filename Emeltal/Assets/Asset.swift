import Foundation

@Observable
final class Asset: Codable, Identifiable {
    let id: String
    let category: Category
    var isInstalled = false
    var isDeprecated = false

    var params: Params {
        didSet {
            Persisted.assetList = Asset.assetList().map {
                if $0.id == id {
                    return self
                }
                return $0
            }
        }
    }

    static func assetList(for section: Section? = nil) -> [Asset] {
        let categories: [Category] = if let section {
            section.presentedModels
        } else {
            Category.allCases.filter(\.selectable)
        }

        var persistedList = (Persisted.assetList ?? [Asset]()).filter { category in
            if categories.contains(where: { $0.id == category.id }) {
                true
            } else if category.isDeprecated {
                section == .deprecated
            } else {
                section == nil
            }
        }

        for defaultAsset in categories.map({ Asset(defaultFor: $0) }) {
            if persistedList.allSatisfy({ $0.id != defaultAsset.id }) {
                persistedList.append(defaultAsset)
            }
        }

        return persistedList.sorted { $0.category.rawValue < $1.category.rawValue }
    }

    static func cleanupNonInstalledAssets() {
        log("Checking for stale assets…")

        let persistedList = Self.assetList()

        // Clean up deprecated model files
        let potentialModelPaths = Set(persistedList.map(\.localModelPath) + [Asset(defaultFor: .whisper).localModelPath])
        let fm = FileManager.default
        let onDiskModelPaths = Set((try? fm.contentsOfDirectory(at: modelsDir, includingPropertiesForKeys: nil)) ?? []).filter { !$0.lastPathComponent.hasPrefix(".") }
        let unusedFiles = onDiskModelPaths.subtracting(potentialModelPaths)
        for file in unusedFiles {
            log("Removing stale unused model file: \(file.path)")
            try? fm.removeItem(at: file)
        }

        let potentialStatePaths = Set(persistedList.map(\.localStatePath))
        let onDiskStatePaths = Set((try? fm.contentsOfDirectory(at: appDocumentsUrl, includingPropertiesForKeys: nil))?.filter { $0.lastPathComponent.hasPrefix("states-") } ?? [])
        let unusedStateDirs = onDiskStatePaths.subtracting(potentialStatePaths)
        for file in unusedStateDirs {
            log("Removing stale state dir: \(file.path)")
            try? fm.removeItem(at: file)
        }
    }

    static func == (lhs: Asset, rhs: Asset) -> Bool {
        lhs.id == rhs.id
    }

    enum CodingKeys: CodingKey {
        case id
        case category
        case params
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(category, forKey: .category)
        try container.encode(params, forKey: .params)
    }

    init(defaultFor category: Asset.Category) {
        id = category.id
        self.category = category
        params = category.defaultParams
        updateInstalledStatus()
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        let loadedCategory = try container.decode(Category.self, forKey: .category)
        category = loadedCategory

        let loadedParams = try container.decode(Params.self, forKey: .params)
        if loadedParams.version == nil {
            var migratedParams = loadedCategory.defaultParams
            migratedParams.systemPrompt = loadedParams.systemPrompt
            params = migratedParams
        } else {
            params = loadedParams
        }
        updateInstalledStatus()
    }

    private func updateInstalledStatus() {
        isInstalled = FileManager.default.fileExists(atPath: localModelPath.path)
        isDeprecated = isInstalled && Asset.Category.allCases.allSatisfy { $0.id != id }
    }

    func unInstall() {
        let fm = FileManager.default
        try? fm.removeItem(at: localModelPath)
        try? fm.removeItem(at: localStatePath)
        updateInstalledStatus()
    }

    static var modelsDir = appDocumentsUrl.appendingPathComponent("models", conformingTo: .directory)

    var localModelPath: URL {
        let modelDir = Self.modelsDir
        let fm = FileManager.default
        if !fm.fileExists(atPath: modelDir.path) {
            try! fm.createDirectory(at: modelDir, withIntermediateDirectories: true)
        }
        return modelDir.appendingPathComponent(category.fetchUrl.lastPathComponent)
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
        Template(format: category.format,
                 system: params.systemPrompt,
                 bosToken: context.bosToken)
    }
}
