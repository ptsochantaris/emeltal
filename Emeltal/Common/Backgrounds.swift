import Foundation
import SwiftUI

private let startTime = Date()

struct Genie: View {
    let show: Bool

    var body: some View {
        TimelineView(.animation(paused: !show)) { timeline in
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

struct PickerEntryBackground: View {
    var body: some View {
        RoundedRectangle(cornerSize: CGSize(width: 20, height: 20), style: .continuous)
            .foregroundStyle(.white.opacity(0.2))
            .blendMode(.softLight)
    }
}

struct ShimmerBackground: View {
    var body: some View {
        TimelineView(.animation) { timeline in
            let elapsedTime = startTime.distance(to: timeline.date)
            Image(.canvas)
                .resizable()
                .visualEffect { content, proxy in
                    content
                        .colorEffect(
                            ShaderLibrary.modelBackground(
                                .float2(proxy.size),
                                .float(elapsedTime)
                            )
                        )
                }
        }
    }
}
