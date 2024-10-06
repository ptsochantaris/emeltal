import Foundation
import SwiftUI

struct AssetCell: View {
    let asset: Asset
    @Binding var selected: Asset

    @Environment(\.openURL) var openUrl

    var body: some View {
        ZStack(alignment: .top) {
            PickerEntryBackground()

            if selected == asset {
                RoundedRectangle(cornerSize: CGSize(width: 20, height: 20), style: .continuous)
                    .stroke(style: StrokeStyle(lineWidth: 3))
                    .foregroundStyle(.accent)
                    .padding(1)
            }

            let category = asset.category
            VStack(spacing: 8) {
                VStack(spacing: 2) {
                    Text(category.displayName)
                        .font(.title2)
                        .lineLimit(1)

                    HStack(spacing: 4) {
                        Text(category.detail)
                            .lineLimit(1)

                        Button {
                            openUrl(category.originalRepoUrl)
                        } label: {
                            Image(systemName: "questionmark.circle.fill")
                        }
                        .buttonStyle(.borderless)
                    }
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                }

                Text(category.aboutText)

                Spacer(minLength: 0)

                HStack {
                    if let label = asset.status.badgeText {
                        Text(label)
                            .font(.caption2)
                            .padding(4)
                            .padding(.horizontal, 4)
                            .background {
                                Capsule(style: .continuous)
                                    .foregroundStyle(.ultraThinMaterial)
                            }
                    }

                    Spacer()

                    Text(category.sizeDescription)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .padding(.trailing, 2)
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
