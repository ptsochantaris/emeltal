import Foundation
import SwiftUI

private let startTime = Date.now.addingTimeInterval(Double.random(in: -10 ..< 0))

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
            .foregroundStyle(.white.opacity(0.3))
            .blendMode(.softLight)
    }
}

struct ShimmerBackground: View {
    let show: Bool

    var body: some View {
        TimelineView(.animation(paused: !show)) { timeline in
            let elapsedTime = startTime.distance(to: timeline.date)
            Rectangle()
                .frame(width: 256, height: 256)
                .visualEffect { content, proxy in
                    content
                        .colorEffect(
                            ShaderLibrary.pickerBackground(
                                .float2(proxy.size),
                                .float(elapsedTime)
                            )
                        )
                        .scaleEffect(CGSize(width: 3.95, height: 3.75))
                }
        }
    }
}
