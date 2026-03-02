import Foundation
import SwiftUI

let appDocumentsUrl: URL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!

let performanceCpuCount = {
    var performanceCpuCount = Int32()
    var size = MemoryLayout.size(ofValue: performanceCpuCount)
    sysctlbyname("hw.perflevel0.physicalcpu", &performanceCpuCount, &size, nil, 0)
    return Int(performanceCpuCount)
}()

struct ParamsHolder: Codable {
    let modelId: String
    let params: Model.Params
}

@MainActor
enum Persisted {
    @AppStorage("_textOnly") static var textOnly = false
    @AppStorage("_floatingMode") static var floatingMode = false
    @AppStorage("_assetSettings") static var selectedAssetId: Model.Variant.ID?
    @AppStorage("_modelParams") static var modelParams: Data?
}

extension Binding {
    // thanks to https://stackoverflow.com/questions/65736518/how-do-i-create-a-slider-in-swiftui-for-an-int-type-property

    static func convert<TInt: BinaryInteger & Sendable, TFloat: BinaryFloatingPoint & Sendable>(from intBinding: Binding<TInt>) -> Binding<TFloat> {
        Binding<TFloat>(
            get: { TFloat(intBinding.wrappedValue) },
            set: { intBinding.wrappedValue = TInt($0) }
        )
    }

    static func convert<TFloat: BinaryFloatingPoint & Sendable, TInt: BinaryInteger & Sendable>(from floatBinding: Binding<TFloat>) -> Binding<TInt> {
        Binding<TInt>(
            get: { TInt(floatBinding.wrappedValue) },
            set: { floatBinding.wrappedValue = TFloat($0) }
        )
    }

    static func round<TFloat: BinaryFloatingPoint & Sendable>(from floatBinding: Binding<TFloat>) -> Binding<TFloat> {
        Binding<TFloat>(
            get: { floatBinding.wrappedValue },
            set: { floatBinding.wrappedValue = ($0 * 100.0).rounded() / 100.0 }
        )
    }
}
