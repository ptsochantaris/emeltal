import Foundation
import SwiftUI

struct ManagerRow: View {
    let manager: AssetFetcher

    private static let fileFormatter = ByteCountFormatStyle(style: .file, allowedUnits: .all, spellsOutZero: true, includesActualByteCount: false, locale: .autoupdatingCurrent)

    var body: some View {
        HStack {
            VStack(alignment: .leading) {
                switch manager.phase {
                case .boot, .done:
                    Text("**\(manager.model.variant.displayName)** Starting")
                case .cancelled:
                    Text("**\(manager.model.variant.displayName)** Cancelled")
                case let .error(error):
                    Text("**\(manager.model.variant.displayName)** error: \(error.localizedDescription)")
                case let .fetching(downloaded, expected):
                    let progress: Double = (Double(downloaded) / Double(expected))
                    let downloadedString = Self.fileFormatter.format(downloaded)
                    let totalString = Self.fileFormatter.format(expected)
                    HStack(alignment: .top) {
                        Text("**\(manager.model.variant.displayName)**")
                        Spacer()
                        Text("\(downloadedString) / \(totalString)")
                    }
                    ProgressView(value: progress)
                    Text(manager.model.variant.fetchUrl.absoluteString)
                }
            }
            .multilineTextAlignment(.leading)
            Spacer()
        }
        .font(.caption)
        .padding()
        .background {
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .foregroundStyle(.secondary)
                .opacity(0.1)
        }
        .padding([.top, .bottom], 4)
    }
}
