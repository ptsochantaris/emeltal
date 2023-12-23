import Foundation
import SwiftUI

struct ModeView: View {
    let mode: AppMode

    var body: some View {
        ZStack {
            switch mode {
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

            case let .alwaysOn(state):
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
        }
        .contentTransition(.opacity)
        .fontWeight(.light)
    }
}
