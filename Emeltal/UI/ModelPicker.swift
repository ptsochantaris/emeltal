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

                HStack {
                    if selectedAsset.isInstalled {
                        Button("Uninstall") {
                            selectedAsset.unInstall()
                        }
                    }

                    Spacer(minLength: 0)

                    let memoryUse = selectedAsset.category.usage

                    if memoryUse.cpuUsageEstimateBytes > 0 || memoryUse.gpuUsageEstimateBytes > 0 {
                        HStack {
                            Group {
                                if let warningMessage = memoryUse.warningMessage {
                                    Image(systemName: "exclamationmark.circle")
                                        .font(.title)

                                    Text(warningMessage)
                                        .foregroundColor(.white)
                                        .frame(width: 250)
                                } else {
                                    Text("Estimated Memory")
                                        .foregroundStyle(.secondary)
                                        .frame(width: 50)
                                }
                            }
                            .multilineTextAlignment(.leading)
                            .font(.caption2)

                            Group {
                                if memoryUse.gpuUsageEstimateBytes > 0 {
                                    VStack(spacing: 0) {
                                        Text("METAL")
                                            .foregroundColor(.accentColor)
                                        Text(memoryUse.gpuUsageEstimateBytes, format: .byteCount(style: .memory))
                                            .foregroundColor(.white)
                                    }
                                }

                                if memoryUse.cpuUsageEstimateBytes > 0 {
                                    VStack(spacing: 0) {
                                        Text("CPU")
                                            .foregroundColor(.accentColor)
                                        Text(memoryUse.cpuUsageEstimateBytes, format: .byteCount(style: .memory))
                                            .foregroundColor(.white)
                                    }
                                }

                                if memoryUse.excessBytes > 0 {
                                    VStack(spacing: 0) {
                                        Text("PAGED")
                                            .foregroundColor(.red)
                                        Text(memoryUse.excessBytes, format: .byteCount(style: .memory))
                                            .foregroundColor(.white)
                                    }
                                }
                            }
                            .fixedSize()
                            .padding(2)
                            .padding([.leading, .trailing], 2)
                            .font(.caption2)
                            .overlay {
                                RoundedRectangle(cornerRadius: 6, style: .circular)
                                    .foregroundStyle(.white.opacity(0.3))
                                    .blendMode(.softLight)
                            }
                        }
                        .padding(8)
                        .background {
                            RoundedRectangle(cornerRadius: 6, style: .circular)
                                .foregroundStyle(.white.opacity(0.3))
                                .blendMode(.softLight)
                        }
                    }

                    Spacer(minLength: 0)

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
                .padding([.top, .bottom], 8)
                .padding([.leading, .trailing])
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
