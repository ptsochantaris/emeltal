import Foundation
import SwiftUI

struct SectionCarousel: View {
    let section: Asset.Section
    let assetList: [Asset]
    @Binding var selectedAsset: Asset

    var body: some View {
        ScrollViewReader { horizontalScrollReader in
            ScrollView(.horizontal) {
                HStack(spacing: 14) {
                    SectionCell(section: section)
                        .frame(width: 200)

                    ForEach(assetList) {
                        AssetCell(asset: $0, selected: $selectedAsset)
                    }
                    .aspectRatio(1.4, contentMode: .fit)
                }
                .frame(height: 200)
                .scrollIndicators(.hidden)
                .padding([.trailing, .top, .bottom])
            }
            .background(.white.opacity(0.3).blendMode(.softLight))
            .onAppear {
                horizontalScrollReader.scrollTo(selectedAsset.id)
            }
        }
    }
}
