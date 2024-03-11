import Foundation
import SwiftUI

let appDocumentsUrl: URL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!

let performanceCpuCount = {
    var performanceCpuCount = Int32()
    var size = MemoryLayout.size(ofValue: performanceCpuCount)
    sysctlbyname("hw.perflevel0.physicalcpu", &performanceCpuCount, &size, nil, 0)
    return Int(performanceCpuCount)
}()

@MainActor
enum Persisted {
    @AppStorage("_textOnly") static var _textOnly = false
    @AppStorage("_floatingMode") static var _floatingMode = false
    @AppStorage("_assetSettings") static var selectedAssetId: Asset.Category.ID?
    @AppStorage("_assetListData") static var assetListData: Data?

    private static var _cachedAssetList: [Asset]?
    static var assetList: [Asset] {
        get {
            let returned: [Asset] = if let _cachedAssetList {
                _cachedAssetList
            } else if let assetListData, let list = try? JSONDecoder().decode([Asset].self, from: assetListData) {
                list
            } else {
                [Asset]()
            }
            _cachedAssetList = returned
            return returned
        }
        set {
            _cachedAssetList = newValue
            assetListData = try? JSONEncoder().encode(newValue)
        }
    }

    static var selectedAsset: Asset {
        let list = Asset.assetList()
        if let selectedAssetId, let existingAsset = list.first(where: { $0.id == selectedAssetId }) {
            return existingAsset
        }
        return list.first!
    }

    static func update(asset: Asset) {
        var list = Asset.assetList()
        if let index = list.firstIndex(where: { $0.id == asset.id }) {
            list[index] = asset
        } else {
            list.append(asset)
        }
        assetList = list
    }
}

let sizeFormatter: ByteCountFormatter = {
    let b = ByteCountFormatter()
    b.countStyle = .file
    b.formattingContext = .standalone
    return b
}()

let memoryFormatter: ByteCountFormatter = {
    let b = ByteCountFormatter()
    b.countStyle = .memory
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
