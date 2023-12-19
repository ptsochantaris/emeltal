import Foundation
import SwiftUI

private let startTime = Date()

struct Genie: View {
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
