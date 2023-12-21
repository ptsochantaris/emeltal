import Foundation
import SwiftUI

extension String: Error {}

let appDocumentsUrl: URL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!

let assistantWidth: CGFloat = 140
let assistantHeight: CGFloat = 380

let performanceCpuCount = {
    var performanceCpuCount = Int32()
    var size = MemoryLayout.size(ofValue: performanceCpuCount)
    sysctlbyname("hw.perflevel0.physicalcpu", &performanceCpuCount, &size, nil, 0)
    return Int(performanceCpuCount)
}()

enum Persisted {
    @AppStorage("_textOnly") static var _textOnly = false
    @AppStorage("_floatingMode") static var _floatingMode = false
    @AppStorage("_assetSettings") static var selectedAssetId: Asset.Category.ID?
    @AppStorage("_assetList") static var assetList: [Asset]?

    static var selectedAsset: Asset {
        let aid = selectedAssetId ?? Asset.Category.solar.id
        return Asset.assetList.first(where: { $0.id == aid })!
    }
}

extension NSSound: @unchecked Sendable {}

let sizeFormatter: ByteCountFormatter = {
    let b = ByteCountFormatter()
    b.countStyle = .file
    b.formattingContext = .standalone
    return b
}()

extension Binding {
    // thanks to https://stackoverflow.com/questions/65736518/how-do-i-create-a-slider-in-swiftui-for-an-int-type-property

    static func convert<TInt, TFloat>(from intBinding: Binding<TInt>) -> Binding<TFloat>
        where TInt: BinaryInteger,
        TFloat: BinaryFloatingPoint {
        Binding<TFloat>(
            get: { TFloat(intBinding.wrappedValue) },
            set: { intBinding.wrappedValue = TInt($0) }
        )
    }

    static func convert<TFloat, TInt>(from floatBinding: Binding<TFloat>) -> Binding<TInt>
        where TFloat: BinaryFloatingPoint,
        TInt: BinaryInteger {
        Binding<TInt>(
            get: { TInt(floatBinding.wrappedValue) },
            set: { floatBinding.wrappedValue = TFloat($0) }
        )
    }

    static func round<TFloat>(from floatBinding: Binding<TFloat>) -> Binding<TFloat>
        where TFloat: BinaryFloatingPoint {
        Binding<TFloat>(
            get: { floatBinding.wrappedValue },
            set: { floatBinding.wrappedValue = ($0 * 100.0).rounded() / 100.0 }
        )
    }
}
