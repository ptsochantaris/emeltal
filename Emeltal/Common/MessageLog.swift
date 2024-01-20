import Foundation
import Ink
import SwiftUI
import WebKit

private let parser = MarkdownParser()

private extension String {
    var markdownToHtml: String {
        if isEmpty {
            return ""
        }
        return parser.html(from: self).replacingOccurrences(of: "'", with: "\\'").replacingOccurrences(of: "\n", with: "\\n")
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
        let webView = WKWebView()
        var displayedHistoryCount = 0
        var displayedBuildingCount = 0

        private let queue = AsyncStream.makeStream(of: String.self, bufferingPolicy: .unbounded)

        func update(from messageLog: MessageLog) {
            var js = ""

            let newHistoryCount = messageLog.history.count
            if displayedHistoryCount != newHistoryCount {
                let html = messageLog.history.markdownToHtml
                js += "document.getElementById('__emeltal_internal_history').innerHTML = '\(html)';"
                displayedHistoryCount = newHistoryCount
            }

            let newBuildingCount = messageLog.newText.count
            if displayedBuildingCount != newBuildingCount {
                let html = messageLog.newText.markdownToHtml
                js += "document.getElementById('__emeltal_internal_new').innerHTML = '\(html)';"
                displayedBuildingCount = newBuildingCount
            }

            if !js.isEmpty {
                queue.continuation.yield(js)
            }
        }

        private func scrollToBottom() {
            #if canImport(AppKit)
                webView.scrollToEndOfDocument(nil)
            #else
                webView.scrollView.scrollRectToVisible(CGRect(x: 0, y: webView.scrollView.contentSize.height - 1, width: 1, height: 1), animated: false)
            #endif
        }

        init() {
            webView.loadHTMLString("""
            <!DOCTYPE html>
            <html>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
            :root {
                color-scheme: light dark;
                color: CanvasText;
                font-family: sans-serif;
                background: transparent;
            }
            p code {
                font-weight: bold;
                font-size: large;
            }
            h4 {
                border-color: gray;
                border-width: 0.5pt;
                border-bottom-style: solid;
                display: block;

                padding: 0;
                padding-bottom: 8pt;
                padding-left: 3pt;
                padding-right: 3pt;

                margin: 0;
                margin-top: 32pt;
                margin-bottom: -4pt;
                margin-left: -3pt;
                margin-right: -3pt;
            }
            blockquote {
                font-size: small;
                margin: 0;
                background: Canvas;
                padding-left: 12pt;
                padding-right: 12pt;
                padding-top: 3pt;
                padding-bottom: 3pt;
                border-radius: 16pt;
            }
            pre code {
                font-size: 110%;
                font-weight: regular;
                font-family: monospace;
                background: Canvas;
                padding: 12pt;
                display: inline-block;
                white-space: pre;
                -webkit-overflow-scrolling: touch;
                overflow-x: scroll;
                max-width: 100%;
                min-width: 100px;
                border-radius: 16pt;
                margin-left: -6pt;

            }
            </style>
            <body>
            <div id='__emeltal_internal_history'></div>
            <div id='__emeltal_internal_new'></div>
            </body>
            </html>
            """, baseURL: nil)

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
                scrollToBottom()
                for await js in queue.stream {
                    _ = try? await webView.evaluateJavaScript(js)
                    scrollToBottom()
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
