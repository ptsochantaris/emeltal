import Foundation
import Ink
import SwiftUI
import WebKit

private let parser = MarkdownParser()

private extension String {
    var markdownToHtml: String {
        let source = trimmingCharacters(in: .whitespacesAndNewlines)
        if source.isEmpty {
            return ""
        }
        return parser.html(from: source)
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "'", with: "\\'")
            .replacingOccurrences(of: "\n", with: "\\n")
    }
}

#if os(macOS)
    struct WebView: NSViewRepresentable {
        let messageLog: MessageLog
        func makeNSView(context: Context) -> WKWebView { context.coordinator.webView }
        func updateNSView(_: WKWebView, context: Context) { context.coordinator.update(from: messageLog) }
        func makeCoordinator() -> Coordinator { Coordinator() }
    }
#else
    struct WebView: UIViewRepresentable {
        let messageLog: MessageLog
        func makeUIView(context: Context) -> WKWebView { context.coordinator.webView }
        func updateUIView(_: WKWebView, context: Context) { context.coordinator.update(from: messageLog) }
        func makeCoordinator() -> Coordinator { Coordinator() }
    }
#endif

extension WebView {
    @MainActor
    final class Coordinator {
        let webView: WKWebView
        var displayedHistoryCount = 0
        var displayedBuildingCount = 0

        private let queue = AsyncStream.makeStream(of: String.self, bufferingPolicy: .unbounded)

        func update(from messageLog: MessageLog) {
            let html1: String?
            let newHistoryCount = messageLog.history.count
            if displayedHistoryCount != newHistoryCount {
                html1 = messageLog.history.markdownToHtml
                displayedHistoryCount = newHistoryCount
            } else {
                html1 = nil
            }

            let html2: String?
            let newBuildingCount = messageLog.newText.count
            if displayedBuildingCount != newBuildingCount {
                html2 = messageLog.newText.markdownToHtml
                displayedBuildingCount = newBuildingCount
            } else {
                html2 = nil
            }

            if (html1 ?? html2) != nil {
                let h1 = if let html1 { "'\(html1)'" } else { "null" }
                let h2 = if let html2 { "'\(html2)'" } else { "null" }
                queue.continuation.yield("setHTML(\(h1),\(h2));")
            }
        }

        deinit {
            log("Coordinator deinit")
        }

        init() {
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

            Task { [weak self] in
                guard let self else { return }
                while webView.isLoading {
                    await Task.yield()
                }
                for await js in queue.stream {
                    do {
                        let ret = try await webView.evaluateJavaScript(js)
                        if ret as? Bool == true {
                            log("Scrolled text view")
                        }
                    } catch {
                        log("Error evaluating JS: \(error)")
                    }
                }
                log("Webview output coordinator shutdown")
            }
        }
    }
}

@Observable
final class MessageLog {
    private(set) var history: String
    private(set) var newText: String

    var isEmpty: Bool {
        history.isEmpty && newText.isEmpty
    }

    init(string: String) {
        history = string
        newText = ""
    }

    init(path: URL?) {
        if let path {
            history = (try? String(contentsOf: path)) ?? ""
        } else {
            history = ""
        }
        newText = ""
    }

    func appendText(_ text: String) {
        newText += text
    }

    func reset() {
        history = ""
        newText = ""
    }

    func commitNewText() {
        history += newText
        newText = ""
    }

    func save(to url: URL) throws {
        try history.write(toFile: url.path, atomically: true, encoding: .utf8)
    }

    deinit {
        log("MessageLog deinit")
    }
}
