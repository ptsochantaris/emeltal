import Foundation
import Ink
import SwiftUI
import WebKit

#if canImport(AppKit)
    struct WebView: NSViewRepresentable {
        let handler: ConversationHandler
        func makeNSView(context: Context) -> WKWebView { context.coordinator.webView }
        func updateNSView(_: WKWebView, context _: Context) {}
        func makeCoordinator() -> Coordinator { Coordinator(handler: handler) }
    }
#else
    struct WebView: UIViewRepresentable {
        let handler: ConversationHandler
        func makeUIView(context: Context) -> WKWebView { context.coordinator.webView }
        func updateUIView(_: WKWebView, context _: Context) {}
        func makeCoordinator() -> Coordinator { Coordinator(handler: handler) }
    }
#endif

protocol ConversationHandler: Sendable {
    var messageLog: MessageLog { get }
    func shutdown() async
}

extension WebView {
    @MainActor
    final class Coordinator {
        let webView: WKWebView
        private let handler: ConversationHandler

        deinit {
            Task { [handler] in
                await handler.shutdown()
            }
            log("Coordinator deinit")
        }

        init(handler: ConversationHandler) {
            log("Coordinator init")

            let logView = Bundle.main.url(forResource: "log", withExtension: "html")!
            let config = WKWebViewConfiguration()
            config.suppressesIncrementalRendering = true
            webView = WKWebView(frame: .zero, configuration: config)
            webView.loadFileURL(logView, allowingReadAccessTo: logView.deletingLastPathComponent())

            #if canImport(AppKit)
                webView.setValue(false, forKey: "drawsBackground")
                webView.enclosingScrollView?.horizontalScrollElasticity = .none
            #else
                webView.isOpaque = false
                webView.scrollView.alwaysBounceHorizontal = false
            #endif

            self.handler = handler
            Task {
                await handler.messageLog.setWebView(webView)
            }
        }
    }
}

private final class LowPriorityExecutor: SerialExecutor {
    fileprivate static let sharedExecutor = LowPriorityExecutor()

    private let ggmlQueue = DispatchQueue(label: "build.bru.emeltal.low-priority", qos: .background)

    func enqueue(_ job: consuming ExecutorJob) {
        let j = UnownedJob(job)
        let e = asUnownedSerialExecutor()
        ggmlQueue.async {
            j.runSynchronously(on: e)
        }
    }

    deinit {
        log("LowPriorityExecutor executor deinit")
    }
}

final actor MessageLog {
    nonisolated var unownedExecutor: UnownedSerialExecutor {
        LowPriorityExecutor.sharedExecutor.asUnownedSerialExecutor()
    }

    private enum Change: Sendable {
        case append(text: String),
             commit,
             save(to: URL, @Sendable (Error?) -> Void),
             isEmpty(@Sendable (Bool) -> Void),
             setHistory(text: String),
             allText(@Sendable (String) -> Void)
    }

    private weak var webView: WKWebView?

    func setWebView(_ webView: WKWebView) {
        self.webView = webView
        Task { [weak self] in
            log("Messagelog queue started")
            guard let stream = self?.changeQueue.stream else { return }
            for await change in stream {
                await self?.process(change: change)
            }
            log("Messagelog queue done")
        }
    }

    nonisolated var isEmpty: Bool {
        get async {
            await withCheckedContinuation { [weak self] continuation in
                if let self {
                    changeQueue.continuation.yield(.isEmpty {
                        continuation.resume(returning: $0)
                    })
                } else {
                    continuation.resume(returning: true)
                }
            }
        }
    }

    nonisolated var allText: String {
        get async {
            await withCheckedContinuation { [weak self] continuation in
                if let self {
                    changeQueue.continuation.yield(.allText {
                        continuation.resume(returning: $0)
                    })
                } else {
                    continuation.resume(returning: "")
                }
            }
        }
    }

    nonisolated func appendText(_ text: String) {
        changeQueue.continuation.yield(.append(text: text))
    }

    nonisolated func reset() {
        changeQueue.continuation.yield(.setHistory(text: ""))
    }

    nonisolated func commitNewText() {
        changeQueue.continuation.yield(.commit)
    }

    nonisolated func save(to url: URL) async throws {
        try await withCheckedThrowingContinuation { [weak self] (continuation: CheckedContinuation<Void, Error>) in
            if let self {
                changeQueue.continuation.yield(.save(to: url) { exception in
                    if let exception {
                        continuation.resume(throwing: exception)
                    } else {
                        continuation.resume()
                    }
                })
            } else {
                continuation.resume()
            }
        }
    }

    nonisolated func setHistory(_ text: String) {
        changeQueue.continuation.yield(.setHistory(text: text))
    }

    nonisolated func setHistory(from url: URL?) {
        let historyString = if let url {
            (try? String(contentsOf: url)) ?? ""
        } else {
            ""
        }
        changeQueue.continuation.yield(.setHistory(text: historyString))
    }

    nonisolated func shutdown() {
        changeQueue.continuation.finish()
    }

    deinit {
        log("MessageLog deinit")
    }

    private var displayedHistoryCount = 0
    private var displayedBuildingCount = 0
    private var history = ""
    private var newText = ""

    private let changeQueue = AsyncStream.makeStream(of: Change.self, bufferingPolicy: .unbounded)
    private let parser = MarkdownParser()

    private func markdownToHtml(_ markdown: String) -> String {
        let source = markdown.trimmingCharacters(in: .whitespacesAndNewlines)
        if source.isEmpty {
            return ""
        }
        return parser.html(from: source)
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "'", with: "\\'")
            .replacingOccurrences(of: "\n", with: "\\n")
    }

    private func process(change: Change) async {
        switch change {
        case let .save(url, callback):
            do {
                try history.write(toFile: url.path, atomically: true, encoding: .utf8)
                callback(nil)
            } catch {
                callback(error)
            }
            return

        case let .allText(callback):
            callback(history + newText)
            return

        case let .isEmpty(callback):
            let isEmpty = newText.isEmpty && history.isEmpty
            callback(isEmpty)
            return

        case let .append(text):
            newText += text

        case .commit:
            history += newText
            newText = ""

        case let .setHistory(text):
            history = text
            newText = ""
        }

        guard let webView else {
            log("Update \(change) without active webview!")
            return
        }

        let html1: String?
        let newHistoryCount = history.count
        if displayedHistoryCount != newHistoryCount {
            html1 = markdownToHtml(history)
            displayedHistoryCount = newHistoryCount
        } else {
            html1 = nil
        }

        let html2: String?
        let newBuildingCount = newText.count
        if displayedBuildingCount != newBuildingCount {
            html2 = markdownToHtml(newText)
            displayedBuildingCount = newBuildingCount
        } else {
            html2 = nil
        }

        if (html1 ?? html2) != nil {
            let h1 = if let html1 { "'\(html1)'" } else { "null" }
            let h2 = if let html2 { "'\(html2)'" } else { "null" }
            let js = "setHTML(\(h1),\(h2));"

            await Task { @MainActor [weak webView] in
                guard let webView else { return }
                while webView.isLoading {
                    await Task.yield()
                    if Task.isCancelled {
                        return
                    }
                }
                do {
                    try await webView.evaluateJavaScript(js)
                } catch {
                    log("Error evaluating JS: \(error)")
                }
            }.value
        }
    }
}
