@preconcurrency import Foundation

@MainActor
@Observable
final class Asset: Codable, Identifiable, Sendable {
    enum Status: Codable, Sendable {
        case checking, available, installed, notReady
    }

    let id: String
    let category: Category
    private(set) var status = Status.checking

    private var booted = false

    var params = Params.empty {
        didSet {
            if booted {
                Persisted.update(asset: self)
            }
        }
    }

    var badgeText: String? {
        switch status {
        case .available:
            category.recommended ? "START HERE" : nil
        case .checking:
            nil
        case .installed:
            "INSTALLED"
        case .notReady:
            "NOT AVAILABLE"
        }
    }

    static func assetList(for section: Section? = nil) -> [Asset] {
        let assetList: [Asset]
        let persistedAssets = Persisted.assetList.filter(\.category.selectable)

        if let section {
            if section == .deprecated {
                let allSupportedCategories = Category.allCases.filter(\.selectable)
                assetList = persistedAssets.filter { persistedAsset in
                    guard persistedAsset.status == .installed else { return false }
                    let isSupported = allSupportedCategories.contains { persistedAsset.id == $0.id }
                    return !isSupported
                }

            } else {
                let sectionCategories = section.presentedModels
                assetList = sectionCategories.map { sectionCategory in
                    if let persisted = persistedAssets.first(where: { $0.id == sectionCategory.id }), persisted.status == .installed {
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
                guard persistedAsset.status == .installed else { return false }
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

    nonisolated func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(category, forKey: .category)
        let paramsCopy = MainActor.assumeIsolated {
            params
        }
        try container.encode(paramsCopy, forKey: .params)
    }

    init(defaultFor category: Asset.Category) {
        id = category.id
        self.category = category
        params = category.defaultParams
        updateInstalledStatus()
        booted = true
    }

    nonisolated init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        let loadedCategory = try container.decode(Category.self, forKey: .category)
        category = loadedCategory

        let loadedParams = try container.decode(Params.self, forKey: .params)
        let loadedVersion = (loadedParams.version ?? 0)
        MainActor.assumeIsolated {
            if loadedVersion < Params.currentVersion {
                var migratedParams = loadedCategory.defaultParams
                migratedParams.systemPrompt = loadedParams.systemPrompt
                params = migratedParams
            } else {
                params = loadedParams
            }
            updateInstalledStatus()
            booted = true
        }
    }

    private static var statusCache = [String: (lastcheck: Date, status: Status, task: Task<Status, Never>?)]()

    private func updateInstalledStatus() {
        if let cacheEntry = Self.statusCache[id], cacheEntry.lastcheck.timeIntervalSinceNow > -600 {
            let cachedStatus = cacheEntry.status
            if cachedStatus == .checking, let existingTask = cacheEntry.task {
                Task { @MainActor in
                    let resolvedStatus = await existingTask.value
                    if !Task.isCancelled, status != resolvedStatus {
                        status = resolvedStatus
                    }
                }
            } else if cachedStatus != status {
                status = cachedStatus
            }
            return
        }

        if FileManager.default.fileExists(atPath: localModelPath.path) {
            Self.statusCache[id] = (Date.now, .installed, nil)
            status = .installed
            return
        }

        let task = Task { @MainActor [weak self] in
            guard let self else { return Status.notReady }
            log("Checking availability of \(category.displayName)")

            var request = URLRequest(url: category.fetchUrl)
            request.httpMethod = "head"
            let response = try? await URLSession.shared.data(for: request).1 as? HTTPURLResponse
            let newStatus = if let code = response?.statusCode, code >= 200, code < 300 {
                Status.available
            } else {
                Status.notReady
            }
            Self.statusCache[id] = (Date.now, newStatus, nil)
            return newStatus
        }

        Self.statusCache[id] = (Date.now, .checking, task)

        Task { @MainActor in
            let newStatus = await task.value
            if !Task.isCancelled, status != newStatus {
                status = newStatus
                log("Status for \(category.displayName) determined to be \(newStatus)")
            }
        }
    }

    func unInstall() {
        let fm = FileManager.default
        try? fm.removeItem(at: localModelPath)
        try? fm.removeItem(at: localStatePath)
        Self.statusCache[id] = nil
        status = .checking
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
