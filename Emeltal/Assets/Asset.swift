import Foundation

struct Asset: RawRepresentable, Codable, Identifiable {
    let id: String
    let category: Category
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
            let list = Persisted.assetList ?? [Asset]()
            let categories: [Category] = [.solar, .dolphinMixtral, .mythoMax, .deepSeekCoder, .shiningValiant, .zephyr3b]
            let newItems = categories
                .map { Asset(defaultFor: $0) }
                .filter { defaultAsset in
                    !list.contains(where: { $0.id == defaultAsset.id })
                }
            return list + newItems
        }
        set {
            Persisted.assetList = newValue
        }
    }

    static func == (lhs: Self, rhs: Self) -> Bool {
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

    init?(rawValue: String) {
        guard let data = rawValue.data(using: .utf8), let instance = try? JSONDecoder().decode(Asset.self, from: data) else {
            return nil
        }
        self = instance
    }

    var rawValue: String {
        let data = (try? JSONEncoder().encode(self)) ?? Data()
        return String(data: data, encoding: .utf8) ?? ""
    }

    var isInstalled = false

    private mutating func updateInstalledStatus() {
        isInstalled = FileManager.default.fileExists(atPath: localModelPath.path)
    }

    mutating func unInstall() {
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
        Template(format: category.format,
                 system: params.systemPrompt,
                 bosToken: context.bosToken)
    }
}
