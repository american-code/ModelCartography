//
//  DenseTextTrainer.swift
//  Backprop + SGD through the residual blocks. Small enough to run at launch.
//

import Foundation

public struct DenseTextTrainer {
    public static func train(_ net: DenseTextModel,
                             corpus: [CartographyInput],
                             epochs: Int = 120,
                             lr: Double = 0.2,
                             seed: UInt64 = 11) {
        var rng = SplitMix64(seed: seed)
        let classIndex = Dictionary(uniqueKeysWithValues: net.classes.enumerated().map { ($1, $0) })
        var order = Array(corpus.indices)
        for _ in 0..<epochs {
            for i in stride(from: order.count - 1, through: 1, by: -1) {
                let j = Int(rng.unit() * Double(i + 1)); order.swapAt(i, j)
            }
            for si in order {
                guard let t = corpus[si].truth, let target = classIndex[t] else { continue }
                step(net, x: corpus[si].flattened, target: target, lr: lr)
            }
        }
    }

    private static func step(_ net: DenseTextModel, x: [Double], target: Int, lr: Double) {
        let f = net.forward(x)
        let V = net.vocab, H = net.hidden, C = net.classes.count

        var dLogits = f.probs; dLogits[target] -= 1

        // Head + dr2
        var dr2 = [Double](repeating: 0, count: V)
        for c in 0..<C {
            let g = dLogits[c]
            for j in 0..<V { dr2[j] += net.wh[c][j] * g; net.wh[c][j] -= lr * g * f.r2[j] }
            net.bh[c] -= lr * g
        }

        // Block 2 (residual: dr1 starts from the skip connection)
        var dr1 = dr2
        var dA2 = [Double](repeating: 0, count: H)
        for i in 0..<V {
            let g = dr2[i]
            for k in 0..<H { dA2[k] += net.wo2[i][k] * g; net.wo2[i][k] -= lr * g * f.a2[k] }
        }
        var dz2 = [Double](repeating: 0, count: H)
        for k in 0..<H { dz2[k] = (f.z2[k] > 0 ? 1 : 0) * dA2[k] }
        for k in 0..<H {
            let g = dz2[k]
            if g != 0 { for j in 0..<V { dr1[j] += net.w2[k][j] * g; net.w2[k][j] -= lr * g * f.r1[j] } }
            net.b2[k] -= lr * g
        }

        // Block 1
        var dA1 = [Double](repeating: 0, count: H)
        for i in 0..<V {
            let g = dr1[i]
            for k in 0..<H { dA1[k] += net.wo1[i][k] * g; net.wo1[i][k] -= lr * g * f.a1[k] }
        }
        var dz1 = [Double](repeating: 0, count: H)
        for k in 0..<H { dz1[k] = (f.z1[k] > 0 ? 1 : 0) * dA1[k] }
        for k in 0..<H {
            let g = dz1[k]
            if g != 0 { for j in 0..<V { net.w1[k][j] -= lr * g * f.r0[j] } }
            net.b1[k] -= lr * g
        }
    }
}
