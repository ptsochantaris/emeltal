import Foundation

extension Model {
    struct MemoryEstimate {
        let layersOffloaded: Int64
        let layersTotal: Int64
        let offloadAsr: Bool
        let offloadKvCache: Bool
        let cpuUsageEstimateBytes: Int64
        let gpuUsageEstimateBytes: Int64
        let excessBytes: Int64

        init(layersOffloaded: Int64, layersTotal: Int64, offloadAsr: Bool, offloadKvCache: Bool, nonOffloadedEstimateBytes: Int64, gpuUsageEstimateBytes: Int64, totalSystemBytes: UInt64) {
            self.layersOffloaded = layersOffloaded
            self.layersTotal = layersTotal
            self.offloadAsr = offloadAsr
            self.offloadKvCache = offloadKvCache
            self.gpuUsageEstimateBytes = gpuUsageEstimateBytes

            let totalSystemBytes = Int64(totalSystemBytes)
            let available = totalSystemBytes - gpuUsageEstimateBytes
            if nonOffloadedEstimateBytes < available {
                cpuUsageEstimateBytes = nonOffloadedEstimateBytes
                excessBytes = 0
            } else {
                cpuUsageEstimateBytes = available
                excessBytes = nonOffloadedEstimateBytes - available
            }
        }

        var warningMessage: String? {
            if excessBytes > 0 {
                return "This model will not fit into memory. It will run but extremely slowly, as data will need paging"
            }

            if offloadKvCache {
                return nil
            }

            if layersOffloaded > 0, layersTotal > 0 {
                if layersOffloaded == layersTotal - 1 {
                    return "This model fit all \(layersTotal) layers in Metal but will use the CPU for the KV cache and output layer"
                }
                let ratio = Float(layersOffloaded) / Float(layersTotal)
                if ratio == 1 {
                    return "This model fit all \(layersTotal) layers in Metal but will use the CPU for the KV cache"
                } else if ratio > 0.8 {
                    return "This model will fit \(layersOffloaded) of \(layersTotal) layers in Metal. It will work but may be slow for real-time chat"
                } else {
                    return "This model will fit \(layersOffloaded) of \(layersTotal) layers in Metal. It will work but may be very slow for real-time chat"
                }
            }

            if offloadAsr {
                return "This model won't fit in Metal at all. It will work but will be too slow for real-time chat"
            }

            return "Emeltal won't use Metal at all. It will work but will probably be slow"
        }

        var warningBeforeStart: String? {
            if excessBytes > 0 {
                return "This model will not fit into your device's memory. You can try to run it, but most likely it will crash or run extremely slowly!"
            }

            return nil
        }
    }
}
