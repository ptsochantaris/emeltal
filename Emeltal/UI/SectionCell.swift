import Foundation
import SwiftUI

struct SectionCell: View {
    let section: Asset.Section

    var body: some View {
        HStack(spacing: 16) {
            Rectangle()
                .frame(width: 1)

            VStack(alignment: .leading, spacing: 8) {
                Text(section.title)
                    .font(.title)

                Text(section.description)
                    .font(.caption2)

                Spacer(minLength: 0)
            }
            .multilineTextAlignment(.leading)

            Spacer(minLength: 0)
        }
        .padding(.leading, 16)
    }
}
