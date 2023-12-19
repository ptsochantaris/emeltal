import Foundation
import SwiftUI

struct ModelPicker: View {
    let allowCancel: Bool
    @State var selectedAsset: Asset
    let selection: (Asset) -> Void

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            VStack {
                Text("The model you select will be downloaded and installed locally on your system. You can change your selection from the menu later. Please ensure you have enough disk space for the model you select.")
                    .multilineTextAlignment(.center)
                    .font(.subheadline)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding()
                    .padding([.leading, .trailing], 64)

                let recommended = Asset.solar
                let item = GridItem(spacing: 14)
                LazyVGrid(columns: [item, item], spacing: 14) {
                    ForEach(Asset.assetList) {
                        AssetCell(asset: $0, recommended: $0 == recommended, selected: $selectedAsset)
                    }
                }

                HStack {
                    if selectedAsset.isInstalled {
                        Button("Uninstall") {
                            uninstall(selectedAsset)
                        }
                    }
                    if selectedAsset.useGpuOnThisSystem {
                        Spacer()
                    } else {
                        Text("This model won't fit in this system's video memory and will need to use the CPU. It will work but it will be **too slow for real-time chat**.")
                            .foregroundStyle(.black)
                            .frame(maxWidth: .infinity)
                    }
                    if allowCancel {
                        Button("Cancel") {
                            dismiss()
                        }
                    }
                    Button(selectedAsset.isInstalled ? "Select" : "Install") {
                        selection(selectedAsset)
                    }
                }
                .padding([.leading, .trailing], selectedAsset.useGpuOnThisSystem ? 0 : 16)
                .padding([.top, .bottom], 8)
                .background(selectedAsset.useGpuOnThisSystem ? .clear : .accent)
                .cornerRadius(8.0)
            }
            .padding([.leading, .trailing])
            .padding(.bottom, 10)
            .foregroundStyle(.white)
            .background(Image(.canvas).resizable())
            .navigationTitle("Select an ML model")
        }
    }

    private func uninstall(_ asset: Asset) {
        asset.unInstall()
        selectedAsset = .mythoMax // force an update
        selectedAsset = .solar // force an update
        selectedAsset = asset
    }
}
