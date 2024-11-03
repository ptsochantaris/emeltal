import Foundation
import SwiftUI

struct AssetCell: View {
    let model: Model
    @Binding var selected: Model

    @Environment(\.openURL) var openUrl

    var body: some View {
        ZStack(alignment: .top) {
            PickerEntryBackground()

            if selected == model {
                RoundedRectangle(cornerSize: CGSize(width: 20, height: 20), style: .continuous)
                    .stroke(style: StrokeStyle(lineWidth: 3))
                    .foregroundStyle(.accent)
                    .padding(1)
            }

            let variant = model.variant
            VStack(spacing: 8) {
                VStack(spacing: 2) {
                    Text(variant.displayName)
                        .font(.title2)
                        .lineLimit(1)

                    HStack(spacing: 4) {
                        Text(variant.detail)
                            .lineLimit(1)

                        Button {
                            openUrl(variant.originalRepoUrl)
                        } label: {
                            Image(systemName: "questionmark.circle.fill")
                        }
                        .buttonStyle(.borderless)
                    }
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                }

                Text(variant.aboutText)

                Spacer(minLength: 0)

                HStack {
                    if let info = model.status.badgeInfo {
                        Text(info.label)
                            .font(.caption2)
                            .padding(4)
                            .padding(.horizontal, 4)
                            .background {
                                ZStack {
                                    Rectangle()
                                        .foregroundStyle(.material)

                                    if info.progress > 0 {
                                        GeometryReader { proxy in
                                            Color.white
                                                .opacity(0.24)
                                                .frame(width: proxy.size.width * info.progress)
                                        }
                                    }
                                }
                                .clipShape(.capsule)
                            }
                    }

                    Spacer()

                    Text(variant.sizeDescription)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .padding(.trailing, 2)
                }
            }
            .multilineTextAlignment(.center)
            .padding()
            .frame(minHeight: 0)
            .opacity(model.memoryEstimate.warningBeforeStart == nil ? 1 : 0.6)
        }
        .onTapGesture {
            selected = model
        }
    }
}
