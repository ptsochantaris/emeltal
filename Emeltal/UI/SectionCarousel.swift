import Foundation
import SwiftUI

struct SectionCarousel: View {
    let category: Model.Category
    let modelList: [Model]
    @Binding var selected: Model

    var body: some View {
        ScrollViewReader { horizontalScrollReader in
            ScrollView(.horizontal) {
                HStack(spacing: 14) {
                    SectionCell(category: category)
                        .frame(width: 200)

                    ForEach(modelList) {
                        AssetCell(model: $0, selected: $selected)
                    }
                    .aspectRatio(1.4, contentMode: .fit)
                }
                .frame(height: 200)
                .scrollIndicators(.hidden)
                .padding([.trailing, .top, .bottom])
            }
            .background(.white.opacity(0.3).blendMode(.softLight))
            .onAppear {
                horizontalScrollReader.scrollTo(selected.variant.id)
            }
        }
    }
}
