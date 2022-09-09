import Foundation
import simd

class AccumulatedPointCloud {
    
    var points = ContiguousArray<simd_float3>()
    var colors = ContiguousArray<simd_uint3>()
    var identifiedIndices = [UInt64: Int]()
    var count: Int {
        return self.points.count
    }
    
    init() {
        let baseCapacity = 20000
        self.points.reserveCapacity(baseCapacity)
        self.colors.reserveCapacity(baseCapacity)
        self.identifiedIndices.reserveCapacity(baseCapacity)
    }
    
    func resetPointCloud() {
        self.points.removeAll()
    }
    
    func appendPointCloud(_ point: vector_float3, _ identifier: UInt64, _ color: vector_uint3) {
        if let existingIndex = self.identifiedIndices[identifier] {
            self.points[existingIndex] = point
            self.colors[existingIndex] = color
        } else {
            self.identifiedIndices[identifier] = self.points.endIndex
            self.points.append(point)
            self.colors.append(color)
        }
    }
}
