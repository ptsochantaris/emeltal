import Foundation
import SwiftUI

@main
@MainActor
struct EmeltalApp: App {
    @State private var state: AppState? = nil
    @State private var asset = Persisted.selectedAsset

    var body: some Scene {
        Window("Emeltal", id: "Emeltal") {
            if let currentState = state {
                ContentView(state: currentState) {
                    Task { @MainActor in
                        await currentState.shutdown()
                        state = nil
                    }
                }
            } else {
                ModelPicker(selectedAsset: $asset, allowCancel: false) {
                    Persisted.selectedAssetId = asset.category.id
                    state = AppState(asset: asset)
                }
                .frame(idealWidth: 1000, idealHeight: 950)
            }
        }
        .windowResizability(state != nil ? .automatic : .contentSize)
    }
}
