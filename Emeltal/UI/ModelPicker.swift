import Foundation
import SwiftUI

private struct DescriptorTitle: View {
    let descriptor: Model.Params.Descriptors.Descriptor
    let value: Float

    var body: some View {
        HStack {
            Text(descriptor.title)
                .opacity(0.8)
            if value == descriptor.disabled {
                Text("Disabled")
            } else {
                Text(value, format: .number)
            }
            Spacer()
        }
    }
}

private struct IntRow: View {
    let descriptor: Model.Params.Descriptors.Descriptor
    @Binding var value: Int

    var body: some View {
        VStack {
            DescriptorTitle(descriptor: descriptor, value: Float(value))
            Slider(value: .convert(from: $value), in: descriptor.min ... descriptor.max)
        }
    }
}

private struct FloatRow: View {
    let descriptor: Model.Params.Descriptors.Descriptor
    @Binding var value: Float

    var body: some View {
        VStack {
            DescriptorTitle(descriptor: descriptor, value: value)
            Slider(value: .round(from: $value), in: descriptor.min ... descriptor.max)
        }
    }
}

private struct SelectionGrid: View {
    let showingOverrides: Bool
    @Bindable var manager: ManagerViewModel

    var body: some View {
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

                    ForEach(Model.Category.allCases) { category in
                        if category.displayable {
                            let models = manager.models(for: category)
                            SectionCarousel(category: category, modelList: models, selected: $manager.selected)
                        }
                    }
                }
                .padding([.top, .bottom])
            }
            .scrollIndicators(.hidden)
            .onAppear {
                if let section = manager.category(for: manager.selected.variant) {
                    verticalScrollReader.scrollTo(section.id)
                }
            }
            .onChange(of: showingOverrides) { _, newValue in
                Task {
                    try? await Task.sleep(for: .seconds(0.3))
                    if newValue, let section = manager.category(for: manager.selected.variant) {
                        withAnimation {
                            verticalScrollReader.scrollTo(section.id)
                        }
                    }
                }
            }
        }
    }
}

private struct Overrides: View {
    @Bindable var manager: ManagerViewModel

    var body: some View {
        VStack(spacing: 24) {
            if manager.selected.variant.format.acceptsSystemPrompt {
                VStack {
                    HStack(alignment: .bottom) {
                        Text("System Prompt")
                            .padding(.top, 3)

                        Spacer()

                        Text("(applies when creating, or after resetting, a conversation)")
                            .foregroundStyle(.secondary)
                            .font(.caption2)
                    }
                    TextField("System Prompt", text: $manager.selected.params.systemPrompt, axis: .vertical)
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
                    let params = manager.selected.params
                    HStack(spacing: 10) {
                        let hasTemp = params.temperature != Model.Params.Descriptors.temperature.disabled
                        let hasRange = hasTemp && params.temperatureRange != Model.Params.Descriptors.temperatureRange.disabled
                        FloatRow(descriptor: Model.Params.Descriptors.temperature, value: $manager.selected.params.temperature)
                            .opacity(hasTemp ? 1.0 : 0.5)

                        FloatRow(descriptor: Model.Params.Descriptors.temperatureRange, value: $manager.selected.params.temperatureRange)
                            .opacity(hasRange ? 1.0 : 0.5)

                        FloatRow(descriptor: Model.Params.Descriptors.temperatureExponent, value: $manager.selected.params.temperatureExponent)
                            .opacity(hasRange ? 1.0 : 0.5)
                    }

                    FloatRow(descriptor: Model.Params.Descriptors.topP, value: $manager.selected.params.topP)
                        .opacity(params.topP != Model.Params.Descriptors.topP.disabled ? 1.0 : 0.5)

                    IntRow(descriptor: Model.Params.Descriptors.topK, value: $manager.selected.params.topK)
                        .opacity(params.topK != Int(Model.Params.Descriptors.topK.disabled) ? 1.0 : 0.5)
                }

                VStack(spacing: 10) {
                    FloatRow(descriptor: Model.Params.Descriptors.repeatPenatly, value: $manager.selected.params.repeatPenatly)
                    FloatRow(descriptor: Model.Params.Descriptors.frequencyPenatly, value: $manager.selected.params.frequencyPenatly)
                    FloatRow(descriptor: Model.Params.Descriptors.presentPenatly, value: $manager.selected.params.presentPenatly)
                }
            }
        }
        .font(.callout)
        .padding(16)
        .background(.white.opacity(0.2))
    }
}

private struct MemoryUse: View {
    let memoryUse: Model.GpuUsage

    var body: some View {
        if memoryUse.cpuUsageEstimateBytes > 0 || memoryUse.gpuUsageEstimateBytes > 0 {
            HStack {
                Group {
                    if let warningMessage = memoryUse.warningMessage {
                        Text(warningMessage)
                            .foregroundColor(.white)
                            .frame(width: 250, alignment: .trailing)
                    } else {
                        Text("Estimated Memory Use")
                            .foregroundStyle(.secondary)
                            .frame(width: 70, alignment: .trailing)
                    }
                }
                .multilineTextAlignment(.trailing)

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
                .overlay {
                    RoundedRectangle(cornerRadius: 6, style: .circular)
                        .foregroundStyle(.white.opacity(0.3))
                        .blendMode(.softLight)
                }
            }
            .font(.caption2)
            .padding([.top, .bottom], 8)
        }
    }
}

private struct Buttons: View {
    @Binding var showOverrides: Bool
    @Binding var appPhase: EmeltalApp.Phase
    let manager: ManagerViewModel

    var body: some View {
        HStack {
            let selected = manager.selected
            if selected.status == .installed {
                Button("Uninstall") {
                    selected.unInstall()
                }
                #if !os(visionOS)
                .foregroundStyle(.black)
                #endif
            }

            Spacer(minLength: 0)

            MemoryUse(memoryUse: selected.variant.usage)

            Button(showOverrides ? "Use Defaults" : "Customize…") {
                if showOverrides {
                    withAnimation {
                        selected.resetToDefaults()
                    }
                } else {
                    withAnimation {
                        showOverrides = true
                    }
                }
            }
            #if !os(visionOS)
            .foregroundStyle(.black)
            #endif

            switch selected.status {
            case .checking, .notReady:
                EmptyView()
            case .available, .recommended:
                Button("Install") {
                    go()
                }
                #if !os(visionOS)
                .foregroundStyle(.black)
                #endif
            case .installed:
                Button("Select") {
                    go()
                }
                #if !os(visionOS)
                .foregroundStyle(.black)
                #endif
            }
        }
        .buttonStyle(.borderedProminent)
        .padding([.top, .bottom], 8)
        .padding([.leading, .trailing])
        .background(.white.opacity(0.2))
    }

    private func go() {
        let llm = AssetFetcher(fetching: manager.selected)
        let state = ConversationState(llm: llm, whisper: manager.whisper)
        appPhase = .conversation(state)
    }
}

struct ModelPicker: View {
    @Binding var appPhase: EmeltalApp.Phase

    @State private var visible = false
    @State private var showOverrides = false

    private let manager = ManagerViewModel.shared

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            SelectionGrid(showingOverrides: showOverrides, manager: manager)

            if showOverrides {
                Overrides(manager: manager)
                    .transition(.move(edge: .bottom))
            }

            Buttons(showOverrides: $showOverrides, appPhase: $appPhase, manager: manager)
        }
        .background {
            ShimmerBackground(show: $visible)
                .ignoresSafeArea()
        }
        .navigationTitle("Select an ML model")
        .foregroundStyle(.white)
        .toolbarTitleDisplayMode(.inline)
        #if canImport(UIKit)
            .toolbarBackground(.material, for: .navigationBar)
        #endif
            .onAppear { visible = true }
            .onDisappear { visible = false }
            .onAppear {
                manager.cleanupNonInstalledAssets()
            }
    }
}
