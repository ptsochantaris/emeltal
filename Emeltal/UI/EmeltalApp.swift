import Foundation
import SwiftUI

@main
@MainActor
struct EmeltalApp: App {
    @State private var state: AppState? = nil // TODO: AppState(asset: Persisted.selectedAsset) - for when we have the option to select in menu

    var body: some Scene {
        Window("Emeltal", id: "Emeltal") {
            if let state {
                ContentView(state: state)
            } else {
                ModelPicker(allowCancel: false, selectedAsset: Persisted.selectedAsset ?? .solar) { asset in
                    Persisted.selectedAsset = asset
                    state = AppState(asset: asset)
                }
                .frame(width: 600)
                .fixedSize()
            }
        }
        .windowResizability(state != nil ? .automatic : .contentSize)
    }
}
