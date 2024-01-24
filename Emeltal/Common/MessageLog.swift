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
        return parser.html(from: source).replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "'", with: "\\'").replacingOccurrences(of: "\n", with: "\\n")
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
            var js = ""

            let newHistoryCount = messageLog.history.count
            if displayedHistoryCount != newHistoryCount {
                let html = messageLog.history.markdownToHtml
                js += "setHTML(historyElement, '\(html)');"
                displayedHistoryCount = newHistoryCount
            }

            let newBuildingCount = messageLog.newText.count
            if displayedBuildingCount != newBuildingCount {
                let html = messageLog.newText.markdownToHtml
                js += "setHTML(newElement, '\(html)');"
                displayedBuildingCount = newBuildingCount
            }

            if !js.isEmpty {
                js += "setTimeout(scrollToBottom, 1);"
                queue.continuation.yield(js)
            }
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

            Task {
                while webView.isLoading {
                    await Task.yield()
                }
                for await js in queue.stream {
                    do {
                        _ = try await webView.evaluateJavaScript(js)
                    } catch {
                        log("Error evaluating JS: \(error)")
                    }
                }
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

    func save(to url: URL) throws {
        history += newText
        newText = ""
        try history.write(toFile: url.path, atomically: true, encoding: .utf8)
    }
}
