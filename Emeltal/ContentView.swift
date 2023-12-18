import Carbon.HIToolbox.Events
import MarkdownUI
import SwiftUI

private struct Identifier: Identifiable, Hashable {
    let id: String
}

private let bottomId = Identifier(id: "bottomId")
private let startTime = Date()

private struct PushButton: NSViewRepresentable {
    private let view: MouseTracker

    init(handler: @escaping (Bool) -> Void) {
        view = MouseTracker(handler: handler)
    }

    func makeNSView(context _: Context) -> MouseTracker {
        view
    }

    func updateNSView(_: MouseTracker, context _: Context) {}

    final class MouseTracker: NSView {
        var handler: (Bool) -> Void

        private var pushed = false

        init(handler: @escaping (Bool) -> Void, pressed _: Bool = false) {
            self.handler = handler
            super.init(frame: .zero)
            NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .keyUp]) { [weak self] event in
                guard event.keyCode == kVK_DownArrow, let self else { return event }
                if event.isARepeat { return nil }
                switch event.type {
                case .keyUp:
                    self.handler(false)
                    return nil
                case .keyDown:
                    self.handler(true)
                    return nil
                default:
                    return event
                }
            }
        }

        @available(*, unavailable)
        required init?(coder _: NSCoder) {
            fatalError()
        }

        override func mouseDown(with event: NSEvent) {
            super.mouseDown(with: event)
            handler(true)

            while true {
                guard let theEvent = window?.nextEvent(matching: [.leftMouseUp]) else {
                    continue
                }
                switch theEvent.type {
                case .leftMouseUp:
                    handler(false)
                    return
                default:
                    break
                }
            }
        }
    }
}

private struct Assistant: View {
    let state: AppState

    var body: some View {
        ZStack {
            switch state.mode {
            case .booting, .loading, .startup, .warmup:
                ProgressView()
                    .colorScheme(.dark)

            case .waiting:
                Image(systemName: "circle")
                    .resizable()
                    .aspectRatio(contentMode: .fit)

                Text("Push to\nSpeak")
                    .fontWeight(.bold)
                    .multilineTextAlignment(.center)

            case let .listening(state):
                switch state {
                case .listening:
                    Image(systemName: "waveform.circle")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .symbolEffect(.variableColor.iterative)
                case .quiet:
                    Image(systemName: "waveform.circle")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .opacity(0.8)
                }

            case .thinking:
                Image(systemName: "ellipsis.circle")
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .symbolEffect(.variableColor)

            case .noting:
                Image(systemName: "circle")
                    .resizable()
                    .aspectRatio(contentMode: .fit)

                ProgressView()
                    .colorScheme(.dark)

            case .replying:
                Image(systemName: "waveform.circle.fill")
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .symbolEffect(.variableColor)
            }

            if state.mode != .loading(managers: []) {
                PushButton { down in
                    if down {
                        state.pushButtonDown()
                    } else {
                        state.pushButtonUp()
                    }
                }
            }
        }
        .contentTransition(.opacity)
        .fontWeight(.light)
    }
}

private struct Genie: View {
    let state: AppState

    var body: some View {
        TimelineView(.animation(paused: !state.mode.showGenie)) { timeline in
            let elapsedTime = startTime.distance(to: timeline.date)
            EllipticalGradient(colors: [.black.opacity(0.1), .clear], center: .center, startRadiusFraction: 0, endRadiusFraction: 0.5)
                .visualEffect { content, proxy in
                    content
                        .colorEffect(
                            ShaderLibrary.genie(
                                .float2(proxy.size),
                                .float(elapsedTime)
                            )
                        )
                }
        }
    }
}

@MainActor
private struct ManagerRow: View {
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

@MainActor
private struct ContentView: View {
    @Bindable var state: AppState

    @FocusState var focusEntryField

    var body: some View {
        VStack(spacing: 16) {
            HStack(alignment: .top, spacing: 14) {
                if !state.floatingMode {
                    VStack {
                        if case let .loading(managers) = state.mode {
                            let visibleFetchers = managers.filter(\.phase.shouldShowToUser)
                            if !visibleFetchers.isEmpty {
                                Text("Fetching ML models. This can take a while, but is only needed once.")
                                    .font(.headline)
                                ForEach(visibleFetchers) {
                                    ManagerRow(manager: $0)
                                }
                            }
                        } else if let message = state.statusMessage {
                            Text(message)
                                .foregroundStyle(.black)
                                .font(.headline)
                                .padding()
                                .background {
                                    Capsule(style: .continuous)
                                        .foregroundStyle(.accent)
                                }
                        }

                        ScrollViewReader { proxy in
                            ScrollView {
                                VStack(spacing: 0) {
                                    Markdown(MarkdownContent(state.messageLog))
                                        .textSelection(.enabled)
                                        .markdownTheme(.docC)
                                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                                    Spacer()
                                        .id(bottomId)
                                }
                            }
                            .onChange(of: state.messageLog) { _, _ in
                                proxy.scrollTo(bottomId)
                            }
                        }

                        TextField("Hold \"↓\" to speak, or enter your message here", text: $state.multiLineText)
                            .textFieldStyle(.roundedBorder)
                            .focused($focusEntryField)
                            .onSubmit {
                                if state.mode == .waiting {
                                    state.send()
                                }
                            }
                    }
                    .toolbar {
                        Button {
                            state.textOnly.toggle()
                        } label: {
                            HStack(spacing: 0) {
                                Image(systemName: state.textOnly ? "text.bubble" : "speaker.wave.2.bubble")
                                Text(state.textOnly ? "Text-Only Replies" : "Spoken Replies")
                                    .padding([.leading, .trailing], 4)
                                    .font(.caption)
                            }
                        }

                        Button {
                            Task {
                                if state.mode == .waiting {
                                    try? await state.reset()
                                }
                            }
                        } label: {
                            HStack(spacing: 0) {
                                Image(systemName: "clear")
                                Text("Reset")
                                    .padding([.leading, .trailing], 4)
                                    .font(.caption)
                            }
                        }
                        .keyboardShortcut(KeyEquivalent("k"), modifiers: .command)
                    }
                }

                VStack(spacing: 0) {
                    Button {
                        state.floatingMode.toggle()
                        if !state.floatingMode {
                            focusEntryField = true
                        }
                    } label: {
                        Image(systemName: state.floatingMode ? "chevron.compact.up" : "chevron.compact.down")
                            .font(.largeTitle)
                            .padding()
                    }
                    .buttonStyle(.borderless)

                    if state.mode.showGenie {
                        Genie(state: state)
                            .padding(.top, -4)
                            .padding(.bottom, -1)
                            .gesture(DragGesture().onChanged { value in
                                dragged(by: value)
                            }.onEnded { _ in
                                dragEnd()
                            })

                    } else {
                        let va = state.listenState == .voiceActivated
                        if state.mode.showAlwaysOn {
                            Button("ALWAYS ON") {
                                if va {
                                    state.switchToPushButton()
                                } else {
                                    state.switchToVoiceActivated()
                                }
                            }
                            .font(.caption2.bold())
                            .buttonStyle(PlainButtonStyle())
                            .foregroundStyle(va ? .accent : .widgetForeground)
                            .padding(.vertical, 4)
                            .padding(.horizontal, 8)
                            .background {
                                Capsule(style: .continuous)
                                    .stroke(va ? .accent : .widgetForeground, lineWidth: 1)
                            }
                            .padding(1)
                        }

                        Color(.clear)
                            .contentShape(Rectangle())
                            .gesture(DragGesture().onChanged { value in
                                dragged(by: value)
                            }.onEnded { _ in
                                dragEnd()
                            })
                    }

                    Assistant(state: state)
                        .padding(.bottom, 4)
                        .padding([.leading, .trailing])
                        .foregroundColor(.widgetForeground.opacity(state.mode == .waiting ? 0.7 : 0.8))
                        .frame(height: assistantWidth) // make it square
                }
                .foregroundColor(.widgetForeground)
                .background(Image(.canvas).resizable().aspectRatio(contentMode: .fill))
                .contentShape(Rectangle())
                .frame(width: assistantWidth)
                .cornerRadius(44)
                .shadow(radius: state.floatingMode ? 0 : 6)
            }
        }
        .padding()
        .navigationTitle("Emeltal — \(state.displayName)")
        .task {
            do {
                try await state.boot()
            } catch {
                fatalError(error.localizedDescription)
            }
        }
    }

    @State private var originalPos: CGPoint? = nil

    private func dragged(by value: DragGesture.Value) {
        guard let window = NSApplication.shared.keyWindow else {
            return
        }
        var frame = window.frame
        let p = window.frame.origin
        frame.origin = CGPoint(x: p.x + value.translation.width, y: p.y - value.translation.height)
        originalPos = frame.origin
        window.setFrame(frame, display: false)
    }

    private func dragEnd() {
        originalPos = nil
    }
}

private struct AssetSelection: View {
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
                Text(asset.displayName)
                    .font(.title2)
                    .lineLimit(1)

                Text(asset.aboutText)

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

                    Text(asset.sizeDescription)
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

private struct ModelPicker: View {
    let allowCancel: Bool
    @State var selectedAsset: Asset
    let selection: (Asset) -> Void

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                Text("The model you select will be downloaded and installed locally on your system. You can change your selection from the menu later. Please ensure you have enough disk space for the model you select.")
                    .multilineTextAlignment(.center)
                    .font(.subheadline)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding()
                    .padding([.leading, .trailing], 64)

                LazyVGrid(columns: [GridItem(), GridItem()]) {
                    AssetSelection(asset: .solar, recommended: true, selected: $selectedAsset)
                    AssetSelection(asset: .dolphinMixtral, recommended: false, selected: $selectedAsset)
                    AssetSelection(asset: .mythoMax, recommended: false, selected: $selectedAsset)
                    AssetSelection(asset: .deepSeekCoder, recommended: false, selected: $selectedAsset)
                }

                HStack {
                    if selectedAsset.isInstalled {
                        Button("Uninstall") {
                            let a = selectedAsset
                            a.unInstall()
                            selectedAsset = .mythoMax // force an update
                            selectedAsset = .solar // force an update
                            selectedAsset = a
                        }
                        Spacer()
                    }
                    if !selectedAsset.useGpuOnThisSystem {
                        Text("This model won't fit in this system's video memory and will need to use the CPU, which will make it **too slow for real-time chat**.")
                            .foregroundStyle(.accent)
                    }
                    Spacer()
                    if allowCancel {
                        Button("Cancel") {
                            dismiss()
                        }
                    }
                    Button(selectedAsset.isInstalled ? "Select" : "Install") {
                        selection(selectedAsset)
                    }
                }
            }
            .foregroundStyle(.white)
            .padding()
            .background(Image(.canvas).resizable())
            .navigationTitle("Please select an ML model")
        }
    }
}

#Preview {
    ModelPicker(allowCancel: true, selectedAsset: .dolphinMixtral) { asset in
        print("selected: \(asset.id)")
    }
    .frame(height: 600)
    .preferredColorScheme(.dark)
}

@main
@MainActor
struct EmeltalApp: App {
    @State private var state: AppState? = nil // TODO: AppState(asset: Persisted.selectedAsset) - for when we have the option to select in menu

    var body: some Scene {
        Window("Emeltal", id: "Emeltal") {
            if let state {
                ContentView(state: state)
            } else {
                ModelPicker(allowCancel: false, selectedAsset: Persisted.selectedAsset ?? .solar) { asset in
                    Persisted.selectedAsset = asset
                    state = AppState(asset: asset)
                }
                .frame(width: 800, height: 400)
                .fixedSize()
            }
        }
        .windowResizability(state != nil ? .automatic : .contentSize)
    }
}
