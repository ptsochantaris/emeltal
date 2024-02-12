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
        let assetList: [Asset]
        let persistedAssets = Persisted.assetList ?? [Asset]()

        if let section {
            if section == .deprecated {
                let allSupportedCategories = Category.allCases.filter(\.selectable)
                assetList = persistedAssets.filter { persistedAsset in
                    guard persistedAsset.isInstalled else { return false }
                    let isSupported = allSupportedCategories.contains { persistedAsset.id == $0.id }
                    return !isSupported
                }

            } else {
                let sectionCategories = section.presentedModels
                assetList = sectionCategories.map { sectionCategory in
                    if let persisted = persistedAssets.first(where: { $0.id == sectionCategory.id }), persisted.isInstalled {
                        persisted
                    } else {
                        Asset(defaultFor: sectionCategory)
                    }
                }
            }
        } else {
            let allSupportedCategories = Category.allCases.filter(\.selectable)
            let supportedList = allSupportedCategories.map { supportedCategory in
                if let installed = persistedAssets.first(where: { $0.id == supportedCategory.id }) {
                    installed
                } else {
                    Asset(defaultFor: supportedCategory)
                }
            }
            let deprecatedList = persistedAssets.filter { persistedAsset in
                guard persistedAsset.isInstalled else { return false }
                let isSupported = allSupportedCategories.contains { persistedAsset.id == $0.id }
                return !isSupported
            }

            assetList = supportedList + deprecatedList
        }

        return assetList.sorted { $0.category.rawValue < $1.category.rawValue }
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
