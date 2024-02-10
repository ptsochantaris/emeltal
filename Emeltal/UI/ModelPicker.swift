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

    @State private var visible = false

    @State private var showOverrides = false
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                ScrollViewReader { verticalScrollReader in
                    ScrollView {
                        VStack(alignment: .leading, spacing: 10) {
                            Text("The model you select will be downloaded and installed locally on your system. You can change your selection from the menu later. Please ensure you have enough disk space for the model you select.")
                                .multilineTextAlignment(.center)
                                .font(.subheadline)
                                .fixedSize(horizontal: false, vertical: true)
                                .padding(.bottom, 8)
                                .padding([.leading, .trailing], 64)
                                .frame(maxWidth: .infinity)

                            ForEach(Asset.Section.allCases) { section in
                                let assetList = Asset.assetList(for: section)
                                if !assetList.isEmpty {
                                    SectionCarousel(section: section, assetList: assetList, selectedAsset: $selectedAsset)
                                }
                            }
                        }
                        .padding([.top, .bottom])
                    }
                    .scrollIndicators(.hidden)
                    .frame(minWidth: 0)
                    .frame(idealHeight: 480)
                    .onAppear {
                        if let section = selectedAsset.category.section {
                            verticalScrollReader.scrollTo(section.id)
                        }
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
                                let params = selectedAsset.params
                                HStack(spacing: 10) {
                                    let hasTemp = params.temperature > 0
                                    let hasRange = hasTemp && params.temperatureRange > 0
                                    FloatRow(title: "Temperature", range: 0 ... 2, value: $selectedAsset.params.temperature)
                                        .opacity(hasTemp ? 1.0 : 0.5)

                                    FloatRow(title: "Range", range: 0 ... 1, value: $selectedAsset.params.temperatureRange)
                                        .opacity(hasRange ? 1.0 : 0.5)

                                    FloatRow(title: "Exponent", range: 1 ... 2, value: $selectedAsset.params.temperatureExponent)
                                        .opacity(hasRange ? 1.0 : 0.5)
                                }

                                FloatRow(title: "Top P", range: 0 ... 1, value: $selectedAsset.params.topP)
                                    .opacity(params.topP < 1 ? 1.0 : 0.5)

                                IntRow(title: "Top K", range: 1 ... 100, value: $selectedAsset.params.topK)
                                    .opacity(params.topK < 100 ? 1.0 : 0.5)
                            }

                            VStack(spacing: 10) {
                                FloatRow(title: "Repeat Penalty", range: 0 ... 2, value: $selectedAsset.params.repeatPenatly)
                                FloatRow(title: "Frequency Penalty", range: 0 ... 2, value: $selectedAsset.params.frequencyPenatly)
                                FloatRow(title: "Presence Penalty", range: 0 ... 2, value: $selectedAsset.params.presentPenatly)
                            }
                        }
                    }
                    .font(.callout)
                    .padding(16)
                    .background(.white.opacity(0.2))
                }

                let gpuUsage = selectedAsset.category.usage

                HStack {
                    if selectedAsset.isInstalled {
                        Button("Uninstall") {
                            selectedAsset.unInstall()
                        }
                    }

                    Group {
                        switch gpuUsage {
                        case .none:
                            Text("⚠️ The app won't use the GPU at all. It will work but will extremely slow.")

                        case .asrOnly:
                            Text("⚠️ This model won't fit in the GPU at all. It will work but will be too slow for real-time chat.")

                        case let .low(allocated, total):
                            Text("⚠️ This model will fit **\(allocated) of \(total)** layers in the GPU. It will work but may be very slow for real-time chat.")

                        case let .partial(allocated, total):
                            Text("⚠️ This model will fit **\(allocated) of \(total)** layers in the GPU. It will work but may be slow for real-time chat.")

                        case let .full(total, kvOffload):
                            if kvOffload {
                                Spacer()
                            } else {
                                Text("⚠️ This model fit all **\(total)** layers on the GPU but will use the CPU for the KV cache.")
                            }
                        }
                    }
                    .foregroundStyle(.accent)
                    .frame(maxWidth: .infinity)

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
                .padding()
                .background(.white.opacity(0.2))
            }
            .foregroundStyle(.white)
            .background(ShimmerBackground(show: visible))
            .navigationTitle("Select an ML model")
            .onAppear { visible = true }
            .onDisappear { visible = false }
        }
        .onAppear {
            Asset.cleanupNonInstalledAssets()
        }
    }
}
