import Foundation

@Observable
final class Asset: Codable, Identifiable {
    let id: String
    let category: Category
    var isInstalled = false

    var params: Params {
        didSet {
            Asset.assetList = Asset.assetList.map {
                if $0.id == id {
                    return self
                }
                return $0
            }
        }
    }

    static var assetList: [Asset] {
        get {
            let categories: [Category] = [.sauerkrautSolar, .dolphinMixtral, .mythoMax, .deepSeekCoder, .shiningValiant, .dolphinPhi2]

            let peristsedList = (Persisted.assetList ?? [Asset]()).filter { category in
                if !categories.contains(where: { $0.id == category.id }), !category.isInstalled {
                    false
                } else {
                    true
                }
            }
            let newItems = categories
                .map { Asset(defaultFor: $0) }
                .filter { defaultAsset in
                    !peristsedList.contains(where: { $0.id == defaultAsset.id })
                }
            return peristsedList + newItems
        }
        set {
            Persisted.assetList = newValue
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
        category = try container.decode(Category.self, forKey: .category)
        params = try container.decode(Params.self, forKey: .params)
        updateInstalledStatus()
    }

    private func updateInstalledStatus() {
        isInstalled = FileManager.default.fileExists(atPath: localModelPath.path)
    }

    func unInstall() {
        let fm = FileManager.default
        try? fm.removeItem(at: localModelPath)
        try? fm.removeItem(at: localStatePath)
        updateInstalledStatus()
    }

    var localModelPath: URL {
        let modelDir = appDocumentsUrl.appendingPathComponent("models", conformingTo: .directory)
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
        if category.format.acceptsSystemPrompt {
            Template(format: category.format,
                     system: params.systemPrompt,
                     bosToken: context.bosToken)
        } else {
            Template(format: category.format,
                     system: "",
                     bosToken: context.bosToken)
        }
    }
}
