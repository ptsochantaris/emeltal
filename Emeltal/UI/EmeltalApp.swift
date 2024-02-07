import Foundation
import SwiftUI

@main
@MainActor
struct EmeltalApp: App {
    @State private var state: AppState? = nil
    @State private var asset = Persisted.selectedAsset

    var body: some Scene {
        Window("Emeltal", id: "Emeltal") {
            if let state {
                ContentView(state: state)
            } else {
                ModelPicker(selectedAsset: $asset, allowCancel: false) {
                    Persisted.selectedAssetId = asset.category.id
                    state = AppState(asset: asset)
                }
                .frame(width: 1000, height: 950)
                .fixedSize()
            }
        }
        .windowResizability(state != nil ? .automatic : .contentSize)
    }
}
