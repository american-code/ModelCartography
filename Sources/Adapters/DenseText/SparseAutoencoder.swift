//
//  SparseAutoencoder.swift
//  Map B, at toy scale. Trained on a layer's activations to pull out a sparse, overcomplete
//  dictionary of directions — the interpretable "features" the design calls the real domains.
//
//      f    = ReLU(We · (x − b_dec) + be)     encode (sparse code)
//      x̂    = b_dec + Wd · f                   decode (reconstruction)
//      loss = ‖x̂ − x‖² + λ · Σ|f|             reconstruct sparsely
//

import Foundation

public final class SparseAutoencoder {
    public let dim: Int          // D — width of the residual activations
    public let count: Int        // M — number of features (overcomplete: M ≥ D)
    let lambda: Double

    var we: [[Double]]           // M × D
    var be: [Double]             // M
    var wd: [[Double]]           // D × M  (columns are feature directions)
    var bDec: [Double]           // D

    public init(dim: Int, count: Int, lambda: Double = 0.03, seed: UInt64) {
        self.dim = dim; self.count = count; self.lambda = lambda
        var rng = SplitMix64(seed: seed)
        we = (0..<count).map { _ in (0..<dim).map { _ in rng.gaussian(0.3) } }
        be = .init(repeating: 0, count: count)
        wd = (0..<dim).map { _ in (0..<count).map { _ in rng.gaussian(0.3) } }
        bDec = .init(repeating: 0, count: dim)
        normalizeDictionary()
    }

    /// Sparse code for one activation vector.
    public func encode(_ x: [Double]) -> [Double] {
        var f = [Double](repeating: 0, count: count)
        for m in 0..<count {
            var s = be[m]
            let row = we[m]
            for j in 0..<dim { s += row[j] * (x[j] - bDec[j]) }
            f[m] = max(0, s)
        }
        return f
    }

    /// Direction of feature `m` in activation space (its decoder column).
    public func direction(_ m: Int) -> [Double] { (0..<dim).map { wd[$0][m] } }

    private func normalizeDictionary() {
        for m in 0..<count {
            var n = 0.0
            for i in 0..<dim { n += wd[i][m] * wd[i][m] }
            n = n.squareRoot()
            if n > 1e-8 { for i in 0..<dim { wd[i][m] /= n } }
        }
    }

    public func train(on data: [[Double]], epochs: Int = 160, lr: Double = 0.05, seed: UInt64 = 5) {
        var rng = SplitMix64(seed: seed)
        var order = Array(data.indices)
        for _ in 0..<epochs {
            for i in stride(from: order.count - 1, through: 1, by: -1) {
                let j = Int(rng.unit() * Double(i + 1)); order.swapAt(i, j)
            }
            for si in order { step(data[si], lr: lr) }
            normalizeDictionary()
        }
    }

    private func step(_ x: [Double], lr: Double) {
        // Forward
        var pre = be
        for m in 0..<count {
            var s = be[m]
            for j in 0..<dim { s += we[m][j] * (x[j] - bDec[j]) }
            pre[m] = s
        }
        let f = pre.map { max(0, $0) }
        var xhat = bDec
        for i in 0..<dim {
            var s = bDec[i]
            for m in 0..<count { s += wd[i][m] * f[m] }
            xhat[i] = s
        }
        let err = (0..<dim).map { xhat[$0] - x[$0] }        // ∂½‖·‖² folded into lr

        // Feature gradient: reconstruction + L1
        var df = [Double](repeating: 0, count: count)
        for m in 0..<count {
            var g = 0.0
            for i in 0..<dim { g += wd[i][m] * err[i] }
            if f[m] > 0 { g += lambda }
            df[m] = g
        }
        let dpre = (0..<count).map { (pre[$0] > 0 ? 1.0 : 0.0) * df[$0] }

        // Decoder + decode-side of b_dec
        for i in 0..<dim {
            for m in 0..<count { wd[i][m] -= lr * err[i] * f[m] }
            bDec[i] -= lr * err[i]
        }
        // Encoder + encode-side of b_dec
        for m in 0..<count {
            let g = dpre[m]
            if g != 0 {
                for j in 0..<dim {
                    we[m][j] -= lr * g * (x[j] - bDec[j])
                    bDec[j] -= lr * g * (-we[m][j])   // ∂(x−b_dec)/∂b_dec = −1
                }
                be[m] -= lr * g
            }
        }
    }
}
