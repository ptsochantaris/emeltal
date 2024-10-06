import Foundation
import SwiftUI

extension View {
    func pushButton(_ pushed: Binding<Bool>) -> some View {
        gesture(DragGesture(minimumDistance: 0, coordinateSpace: .local)
            .onChanged { _ in
                if pushed.wrappedValue == false {
                    pushed.wrappedValue = true
                }
            }
            .onEnded { _ in
                if pushed.wrappedValue == true {
                    pushed.wrappedValue = false
                }
            })
    }
}
