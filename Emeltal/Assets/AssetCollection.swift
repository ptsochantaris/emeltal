import Foundation

extension [Asset]: RawRepresentable {
    public init?(rawValue: String) {
        guard let data = rawValue.data(using: .utf8),
              let instance = try? JSONDecoder().decode([Asset].self, from: data) else {
            return nil
        }
        self = instance
    }

    public var rawValue: String {
        let data = (try? JSONEncoder().encode(self)) ?? Data()
        return String(data: data, encoding: .utf8) ?? ""
    }
}
