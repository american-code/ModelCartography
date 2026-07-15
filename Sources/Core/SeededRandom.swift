//
//  SeededRandom.swift
//  A deterministic RNG so the demo model and corpus are identical on every launch
//  and across all three platforms.
//

import Foundation

public struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64
    public init(seed: UInt64) { self.state = seed }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform double in 0..<1.
    public mutating func unit() -> Double {
        Double(next() >> 11) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Approximately-normal sample via central limit (sum of 6 uniforms), mean 0.
    public mutating func gaussian(_ sigma: Double) -> Double {
        var s = 0.0
        for _ in 0..<6 { s += unit() }
        return (s - 3.0) / 1.732_050_8 * sigma * 1.732_050_8
    }
}
