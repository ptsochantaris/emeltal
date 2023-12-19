import Foundation
import SwiftUI

struct Assistant: View {
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
