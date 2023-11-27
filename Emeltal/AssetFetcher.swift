import Foundation

@MainActor
@Observable
final class AssetFetcher: NSObject, URLSessionDownloadDelegate, Identifiable {
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

    nonisolated var id: String { asset.id }

    @MainActor
    var builderDone: ((Phase) -> Void)?

    var phase: Phase
    let asset: Asset

    private var urlSession: URLSession!

    init(fetching asset: Asset) {
        phase = .boot
        self.asset = asset
        super.init()
        urlSession = URLSession(configuration: URLSessionConfiguration.background(withIdentifier: "build.bru.emeltal.background-download-\(asset.id)"), delegate: self, delegateQueue: nil)

        Task {
            await startup()
        }
    }

    deinit {
        print("Completed setup for \(asset.displayName)")
    }

    nonisolated func urlSession(_: URLSession, didCreateTask task: URLSessionTask) {
        print("Download task created: \(task.taskIdentifier)")
    }

    nonisolated func urlSession(_: URLSession, downloadTask _: URLSessionDownloadTask, didWriteData _: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        Task { @MainActor in
            phase = .fetching(downloaded: totalBytesWritten, expected: totalBytesExpectedToWrite)
        }
    }

    nonisolated func urlSession(_: URLSession, downloadTask _: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        try? FileManager.default.moveItem(at: location, to: asset.localPath)
    }

    private func handleNetworkError(_ error: Error, in task: URLSessionTask) {
        print("Network error on \(task.originalRequest?.url?.absoluteString ?? "<no url>"): \(error.localizedDescription)")
        phase = .error(error: error)
        urlSession.invalidateAndCancel()
        builderDone?(phase)
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

        Task {
            print("Downloaded asset to \(asset.localPath.path)...")
            Task { @MainActor in
                phase = .done
                urlSession.invalidateAndCancel()
                builderDone?(phase)
            }
        }
    }

    private func startup() async {
        print("Setting up asset for \(asset.displayName)")

        if FileManager.default.fileExists(atPath: asset.localPath.path) {
            print("Asset ready at \(asset.localPath.path)...")
            phase = .done
            urlSession.invalidateAndCancel()
            builderDone?(phase)
            return
        }

        print("Need to fetch asset...")
        phase = .fetching(downloaded: 0, expected: 0)

        let downloadTasks = await urlSession.tasks.2
        var related = downloadTasks.filter { $0.originalRequest?.url?.lastPathComponent == asset.fetchUrl.lastPathComponent }
        while related.count > 1 {
            if let task = related.popLast() {
                task.cancel()
            }
        }

        if !related.isEmpty {
            print("Existing download for currently selected asset detected, continuing")
            return
        }

        do {
            print("Requesting new asset transfer...")
            urlSession.downloadTask(with: asset.fetchUrl).resume()
        }
    }

    nonisolated func urlSessionDidFinishEvents(forBackgroundURLSession _: URLSession) {
        print("Background URL session events complete")
    }
}
