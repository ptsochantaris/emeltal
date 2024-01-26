import Foundation
import SwiftUI

private struct IntRow: View {
    let title: String
    let range: ClosedRange<Float>
    @Binding var value: Int

    var body: some View {
        VStack {
            HStack {
                Text(title)
                Text(":")
                Text(value, format: .number)
                Spacer()
            }
            Slider(value: .convert(from: $value), in: range)
        }
    }
}

private struct FloatRow: View {
    let title: String
    let range: ClosedRange<Float>
    @Binding var value: Float

    var body: some View {
        VStack {
            HStack {
                Text(title)
                Text(":")
                Text(value, format: .number)
                Spacer()
            }
            Slider(value: .round(from: $value), in: range)
        }
    }
}

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

                ScrollViewReader { scrollReader in
                    ScrollView {
                        let item = GridItem(spacing: 14)
                        LazyVGrid(columns: [item, item, item], spacing: 14) {
                            let recommended = Asset.Category.sauerkrautSolar
                            ForEach(Asset.assetList) {
                                AssetCell(asset: $0, recommended: $0.category == recommended, selected: $selectedAsset)
                                    .aspectRatio(1.5, contentMode: .fit)
                                    .id($0.id)
                            }
                        }
                    }
                    .frame(idealHeight: 480)
                    .onAppear {
                        scrollReader.scrollTo(selectedAsset.id)
                    }
                }

                if showOverrides {
                    VStack(spacing: 24) {
                        if selectedAsset.category.format.acceptsSystemPrompt {
                            VStack {
                                HStack(alignment: .bottom) {
                                    Text("System Prompt")
                                        .padding(.top, 3)

                                    Spacer()

                                    Text("(applies when creating, or after resetting, a conversation)")
                                        .foregroundStyle(.secondary)
                                        .font(.caption2)
                                }
                                TextField("System Prompt", text: $selectedAsset.params.systemPrompt, axis: .vertical)
                                    .textFieldStyle(PlainTextFieldStyle())
                                    .padding([.top, .bottom], 4)
                                    .padding([.leading, .trailing], 7)
                                    .background {
                                        RoundedRectangle(cornerSize: CGSize(width: 8, height: 8), style: .continuous)
                                            .stroke(.secondary)
                                    }
                            }
                        }

                        HStack(alignment: .top, spacing: 30) {
                            VStack(spacing: 10) {
                                let samplingType = selectedAsset.params.samplingType

                                HStack(spacing: 10) {
                                    FloatRow(title: "Temperature", range: 0 ... 2, value: $selectedAsset.params.temperature)
                                        .opacity(samplingType == .temperature || samplingType == .entropy ? 1.0 : 0.5)

                                    FloatRow(title: "Range", range: 0 ... 1, value: $selectedAsset.params.temperatureRange)
                                        .opacity(samplingType == .entropy ? 1.0 : 0.5)

                                    FloatRow(title: "Exponent", range: 1 ... 2, value: $selectedAsset.params.temperatureExponent)
                                        .opacity(samplingType == .entropy ? 1.0 : 0.5)
                                }

                                FloatRow(title: "Top P", range: 0 ... 1, value: $selectedAsset.params.topP)
                                    .opacity(samplingType == .topP ? 1.0 : 0.5)

                                IntRow(title: "Top K", range: 1 ... 100, value: $selectedAsset.params.topK)
                                    .opacity(samplingType == .topK ? 1.0 : 0.5)
                            }

                            VStack(spacing: 10) {
                                FloatRow(title: "Repeat Penalty", range: 0 ... 2, value: $selectedAsset.params.repeatPenatly)
                                FloatRow(title: "Frequency Penalty", range: 0 ... 2, value: $selectedAsset.params.frequencyPenatly)
                                FloatRow(title: "Presence Penalty", range: 0 ... 2, value: $selectedAsset.params.presentPenatly)
                            }
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
                            selectedAsset.params = selectedAsset.category.defaultParams
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
                .buttonStyle(.borderedProminent)
                .padding([.leading, .trailing], selectedAsset.category.useGpuOnThisSystem ? 0 : 16)
                .padding([.top, .bottom], 8)
                .background(selectedAsset.category.useGpuOnThisSystem ? .clear : .accent)
                .cornerRadius(8.0)
            }
            .padding([.leading, .trailing])
            .padding(.bottom, 10)
            .foregroundStyle(.white)
            .background(ShimmerBackground())
            .navigationTitle("Select an ML model")
        }
    }
}
