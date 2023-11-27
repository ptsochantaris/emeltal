import Foundation
import SwiftUI

extension String: Error {}

let appDocumentsUrl: URL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!

let assistantWidth: CGFloat = 140
let assistantHeight: CGFloat = 380

@globalActor actor GGMLActor {
    static var shared = GGMLActor()
}

let performanceCpuCount = {
    var performanceCpuCount = Int32()
    var size = MemoryLayout.size(ofValue: performanceCpuCount)
    sysctlbyname("hw.perflevel0.physicalcpu", &performanceCpuCount, &size, nil, 0)
    return Int(performanceCpuCount)
}()

enum Persisted {
    @AppStorage("_textOnly") static var _textOnly = false
    @AppStorage("_floatingMode") static var _floatingMode = false
}

extension NSSound: @unchecked Sendable {}

let sizeFormatter: ByteCountFormatter = {
    let b = ByteCountFormatter()
    b.countStyle = .file
    b.formattingContext = .standalone
    return b
}()
