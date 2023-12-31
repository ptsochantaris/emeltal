import Foundation
import SwiftUI

@MainActor
protocol ModeProvider {
    var mode: AppMode { get }
}

struct ModeView: View {
    let modeProvider: any ModeProvider

    init(modeProvider: any ModeProvider) {
        self.modeProvider = modeProvider
    }

    var body: some View {
        ZStack {
            switch modeProvider.mode {
            case .booting, .loading, .startup, .warmup:
                ProgressView()
                    .colorScheme(.dark)

            case .waiting:
                Image(systemName: "circle")
                    .resizable()
                    .aspectRatio(contentMode: .fit)

                Text(canUseMic ? "Push to\nSpeak" : "Need Mic\nPermission")
                    .fontWeight(.bold)
                    .multilineTextAlignment(.center)

            case let .listening(state):
                switch state {
                case let .talking(voiceDetected, _):
                    Image(systemName: "waveform.circle")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .symbolEffect(.variableColor.iterative)
                        .opacity(voiceDetected ? 1.0 : 0.6)
                case .quiet:
                    Image(systemName: "waveform.circle")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .opacity(0.4)
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
        }
        .contentTransition(.opacity)
        .fontWeight(.light)
    }
}
