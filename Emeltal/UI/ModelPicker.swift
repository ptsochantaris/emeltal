import Foundation
import SwiftUI

struct ModelPicker: View {
    @Binding var selectedAsset: Asset
    let allowCancel: Bool
    let selection: () -> Void

    @State private var showOverrides = false
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

                let recommended = Asset.Category.solar
                let item = GridItem(spacing: 14)
                LazyVGrid(columns: [item, item], spacing: 14) {
                    ForEach(Asset.assetList) {
                        AssetCell(asset: $0, recommended: $0.category == recommended, selected: $selectedAsset)
                    }
                }

                if showOverrides {
                    Grid(alignment: .leading) {
                        GridRow {
                            Text("System Prompt")
                                .gridColumnAlignment(.trailing)
                            TextField("System Prompt", text: $selectedAsset.params.systemPrompt)
                                .textFieldStyle(PlainTextFieldStyle())
                                .padding([.top, .bottom], 4)
                                .padding([.leading, .trailing], 7)
                                .background {
                                    RoundedRectangle(cornerSize: CGSize(width: 8, height: 8), style: .continuous)
                                        .stroke(.secondary)
                                }
                        }

                        GridRow {
                            Spacer()
                            Text("(applies when creating, or after resetting, a conversation)")
                                .foregroundStyle(.secondary)
                                .font(.caption2)
                                .padding([.bottom], 4)
                        }

                        GridRow {
                            Text("Top K")
                                .gridColumnAlignment(.trailing)
                            Slider(value: .convert(from: $selectedAsset.params.topK), in: 1 ... 100)
                            Text(selectedAsset.params.topK, format: .number)
                        }

                        GridRow {
                            Text("Top P")
                                .gridColumnAlignment(.trailing)
                            Slider(value: .round(from: $selectedAsset.params.topP), in: 0 ... 1)
                            Text(selectedAsset.params.topP, format: .number)
                        }

                        GridRow {
                            Text("Temperature")
                                .gridColumnAlignment(.trailing)
                            Slider(value: .round(from: $selectedAsset.params.temperature), in: 0 ... 2)
                            Text(selectedAsset.params.temperature, format: .number)
                        }

                        GridRow {
                            Text("Repeat Penalty")
                                .gridColumnAlignment(.trailing)
                            Slider(value: .round(from: $selectedAsset.params.repeatPenatly), in: 0 ... 2)
                            Text(selectedAsset.params.repeatPenatly, format: .number)
                        }

                        GridRow {
                            Text("Frequency Penalty")
                                .gridColumnAlignment(.trailing)
                            Slider(value: .round(from: $selectedAsset.params.frequencyPenatly), in: 0 ... 2)
                            Text(selectedAsset.params.frequencyPenatly, format: .number)
                        }

                        GridRow {
                            Text("Present Penalty")
                                .gridColumnAlignment(.trailing)
                            Slider(value: .round(from: $selectedAsset.params.presentPenatly), in: 0 ... 2)
                            Text(selectedAsset.params.presentPenatly, format: .number)
                        }
                    }
                    .font(.callout)
                    .padding([.top, .bottom], 16)
                }

                HStack {
                    if selectedAsset.isInstalled {
                        Button("Uninstall") {
                            selectedAsset.unInstall()
                        }
                    }
                    if selectedAsset.category.useGpuOnThisSystem {
                        Spacer()
                    } else {
                        Text("This model won't fit in this system's video memory and will need to use the CPU. It will work but it will be **too slow for real-time chat**.")
                            .foregroundStyle(.black)
                            .frame(maxWidth: .infinity)
                    }
                    Button(showOverrides ? "Use Defaults" : "Customize…") {
                        if showOverrides {
                            let currentCategory = selectedAsset.category
                            selectedAsset = Asset(defaultFor: currentCategory)
                        } else {
                            showOverrides = true
                        }
                    }
                    if allowCancel {
                        Button("Cancel") {
                            dismiss()
                        }
                    }
                    Button(selectedAsset.isInstalled ? "Select" : "Install") {
                        selection()
                    }
                }
                .padding([.leading, .trailing], selectedAsset.category.useGpuOnThisSystem ? 0 : 16)
                .padding([.top, .bottom], 8)
                .background(selectedAsset.category.useGpuOnThisSystem ? .clear : .accent)
                .cornerRadius(8.0)
            }
            .padding([.leading, .trailing])
            .padding(.bottom, 10)
            .foregroundStyle(.white)
            .background(Image(.canvas).resizable())
            .navigationTitle("Select an ML model")
        }
    }
}
