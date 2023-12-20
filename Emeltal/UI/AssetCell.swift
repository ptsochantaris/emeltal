import Foundation
import SwiftUI

struct AssetCell: View {
    let asset: Asset
    let recommended: Bool
    @Binding var selected: Asset

    var body: some View {
        ZStack(alignment: .top) {
            RoundedRectangle(cornerSize: CGSize(width: 20, height: 20), style: .continuous)
                .foregroundStyle(.ultraThinMaterial)

            if selected == asset {
                RoundedRectangle(cornerSize: CGSize(width: 20, height: 20), style: .continuous)
                    .stroke(style: StrokeStyle(lineWidth: 4))
            }

            VStack(spacing: 8) {
                Text(asset.category.displayName)
                    .font(.title2)
                    .lineLimit(1)

                Text(asset.category.aboutText)

                Spacer(minLength: 0)

                HStack {
                    if recommended {
                        Text(" START HERE ")
                            .font(.caption2)
                            .padding(4)
                            .background {
                                Capsule(style: .continuous)
                                    .foregroundStyle(.ultraThinMaterial)
                            }
                    }
                    if asset.isInstalled {
                        Text(" INSTALLED ")
                            .font(.caption2)
                            .padding(4)
                            .background {
                                Capsule(style: .continuous)
                                    .foregroundStyle(.ultraThinMaterial)
                            }
                    }

                    Spacer()

                    Text(asset.category.sizeDescription)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }
            .multilineTextAlignment(.center)
            .padding()
            .frame(minHeight: 0)
        }
        .onTapGesture {
            selected = asset
        }
    }
}
