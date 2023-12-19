import Foundation
import SwiftUI

@MainActor
struct ManagerRow: View {
    let manager: AssetManager

    var body: some View {
        HStack {
            VStack(alignment: .leading) {
                switch manager.phase {
                case .boot, .done:
                    Text("**\(manager.asset.displayName)** Starting")
                case let .error(error):
                    Text("**\(manager.asset.displayName)** error: \(error.localizedDescription)")
                case let .fetching(downloaded, expected):
                    let progress: Double = (Double(downloaded) / Double(expected))
                    let downloadedString = sizeFormatter.string(fromByteCount: downloaded)
                    let totalString = sizeFormatter.string(fromByteCount: expected)
                    HStack(alignment: .top) {
                        Text("**\(manager.asset.displayName)**")
                        Spacer()
                        Text("\(downloadedString) / \(totalString)")
                    }
                    ProgressView(value: progress)
                    Text(manager.asset.fetchUrl.absoluteString)
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
