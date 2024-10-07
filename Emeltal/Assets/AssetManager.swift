import Foundation

@MainActor
@Observable
final class AssetManager: NSObject, URLSessionDownloadDelegate, Identifiable {
    enum Phase {
        case boot, fetching(downloaded: Int64, expected: Int64), error(error: Error), done

        var isOngoing: Bool {
            switch self {
            case .boot, .fetching:
                true
            case .done, .error:
                false
            }
        }

        var shouldShowToUser: Bool {
            switch self {
            case .boot, .error, .fetching:
                true
            case .done:
                false
            }
        }
    }

    let id: String
    let model: Model

    var builderDone: ((Phase) -> Void)?
    var phase: Phase

    private var urlSession: URLSession!
    private let localModelPath: URL

    init(fetching model: Model) async {
        id = model.id
        phase = .boot
        self.model = model
        localModelPath = model.localModelPath
        super.init()
        urlSession = URLSession(configuration: URLSessionConfiguration.background(withIdentifier: "build.bru.emeltal.background-download-\(model.id)"), delegate: self, delegateQueue: nil)
        await startup()
    }

    nonisolated func urlSession(_: URLSession, didCreateTask task: URLSessionTask) {
        Task { @MainActor in
            log("[\(model.variant.displayName)] Download task created: \(task.taskIdentifier)")
        }
    }

    nonisolated func urlSession(_: URLSession, downloadTask _: URLSessionDownloadTask, didWriteData _: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        Task { @MainActor in
            phase = .fetching(downloaded: totalBytesWritten, expected: totalBytesExpectedToWrite)
        }
    }

    deinit {
        log("AssetManager deinit")
    }

    nonisolated func urlSession(_: URLSession, downloadTask _: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        try? FileManager.default.moveItem(at: location, to: localModelPath)
    }

    private func handleNetworkError(_ error: Error, in task: URLSessionTask) {
        log("[\(model.variant.displayName)] Network error on \(task.originalRequest?.url?.absoluteString ?? "<no url>"): \(error.localizedDescription)")
        phase = .error(error: error)
        urlSession.invalidateAndCancel()
        builderDone?(phase)
        builderDone = nil
    }

    nonisolated func urlSession(_: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error {
            Task { @MainActor in
                handleNetworkError(error, in: task)
            }
            return
        }

        if let response = task.response as? HTTPURLResponse, response.statusCode >= 400 {
            let error = NSError(domain: "build.bru.mima.network", code: 1, userInfo: [NSLocalizedDescriptionKey: "Server returned code \(response.statusCode)"])
            Task { @MainActor in
                handleNetworkError(error, in: task)
            }
            return
        }

        Task { @MainActor in
            log("[\(model.variant.displayName)] Downloaded asset to \(localModelPath.path)...")
            phase = .done
            urlSession.invalidateAndCancel()
            model.updateInstalledStatus()
            builderDone?(phase)
            builderDone = nil
        }
    }

    private func startup() async {
        log("[\(model.variant.displayName)] Setting up asset for \(model.variant.displayName)")

        if FileManager.default.fileExists(atPath: model.localModelPath.path) {
            log("[\(model.variant.displayName)] Asset ready at \(model.localModelPath.path)...")
            phase = .done
            urlSession.invalidateAndCancel()
            builderDone?(phase)
            builderDone = nil
            return
        }

        log("[\(model.variant.displayName)] Need to fetch asset...")
        phase = .fetching(downloaded: 0, expected: 0)

        let downloadTasks = await urlSession.tasks.2
        var related = downloadTasks.filter { $0.originalRequest?.url?.lastPathComponent == model.variant.fetchUrl.lastPathComponent }
        while related.count > 1 {
            if let task = related.popLast() {
                task.cancel()
            }
        }

        if !related.isEmpty {
            log("[\(model.variant.displayName)] Existing download for currently selected asset detected, continuing")
            return
        }

        do {
            log("[\(model.variant.displayName)] Requesting new asset transfer...")
            urlSession.downloadTask(with: model.variant.fetchUrl).resume()
        }
    }

    nonisolated func urlSessionDidFinishEvents(forBackgroundURLSession _: URLSession) {
        Task { @MainActor in
            log("[\(model.variant.displayName)] Background URL session events complete")
        }
    }
}
