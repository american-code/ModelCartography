//
//  PatternDataset.swift
//  Synthetic, deterministic training data for the demo model.
//
//  Three visually distinct classes on an 8x8 grid so the resulting neurons develop
//  genuine, nameable selectivity — which is what makes the cortex map worth looking at.
//

import Foundation

public enum PatternClass: Int, CaseIterable {
    case horizontal, vertical, diagonal
    public var name: String {
        switch self {
        case .horizontal: return "horizontal"
        case .vertical:   return "vertical"
        case .diagonal:   return "diagonal"
        }
    }
}

public struct PatternDataset {
    public static let side = 8
    public static let labels = PatternClass.allCases.map(\.name)

    /// Build one labeled grid sample for a class.
    public static func sample(_ cls: PatternClass, rng: inout SplitMix64) -> [[Double]] {
        let n = side
        var g = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
        // Faint background texture.
        for r in 0..<n { for c in 0..<n { g[r][c] = rng.unit() * 0.12 } }

        let thickness = 1 + Int(rng.unit() * 1.5)   // 1..2 lines
        switch cls {
        case .horizontal:
            for _ in 0...thickness {
                let r = Int(rng.unit() * Double(n))
                for c in 0..<n { g[min(r, n - 1)][c] = 0.85 + rng.unit() * 0.15 }
            }
        case .vertical:
            for _ in 0...thickness {
                let c = Int(rng.unit() * Double(n))
                for r in 0..<n { g[r][min(c, n - 1)] = 0.85 + rng.unit() * 0.15 }
            }
        case .diagonal:
            let anti = rng.unit() > 0.5
            for i in 0..<n {
                let c = anti ? (n - 1 - i) : i
                g[i][c] = 0.85 + rng.unit() * 0.15
                if i + 1 < n { g[i][min(c + 1, n - 1)] = 0.6 + rng.unit() * 0.2 }
            }
        }
        return g
    }

    /// A reproducible corpus: `perClass` inputs of each class.
    public static func corpus(perClass: Int, seed: UInt64) -> [CartographyInput] {
        var rng = SplitMix64(seed: seed)
        var out: [CartographyInput] = []
        var idx = 0
        for cls in PatternClass.allCases {
            for _ in 0..<perClass {
                let g = sample(cls, rng: &rng)
                out.append(CartographyInput(id: "s\(idx)", truth: cls.name, payload: .grid(g)))
                idx += 1
            }
        }
        return out
    }
}

/// Minimal SGD trainer — enough for the demo net to genuinely separate the classes.
public struct MockNetworkTrainer {
    public static func train(_ net: MockNetwork,
                             corpus: [CartographyInput],
                             epochs: Int = 60,
                             lr: Double = 0.15,
                             seed: UInt64 = 7) {
        var rng = SplitMix64(seed: seed)
        let labelIndex = Dictionary(uniqueKeysWithValues: net.classes.enumerated().map { ($1, $0) })
        var order = Array(corpus.indices)

        for _ in 0..<epochs {
            // Shuffle deterministically.
            for i in stride(from: order.count - 1, through: 1, by: -1) {
                let j = Int(rng.unit() * Double(i + 1))
                order.swapAt(i, j)
            }
            for si in order {
                let x = corpus[si].flattened
                guard let t = corpus[si].truth, let target = labelIndex[t] else { continue }
                sgdStep(net, x: x, target: target, lr: lr)
            }
        }
    }

    private static func sgdStep(_ net: MockNetwork, x: [Double], target: Int, lr: Double) {
        let a = net.activations(x)
        let C = net.classes.count

        // dLogits = probs - onehot
        var dLogits = a.probs
        dLogits[target] -= 1

        // Layer 3
        var dH2 = [Double](repeating: 0, count: net.h2Size)
        for i in 0..<C {
            let g = dLogits[i]
            for j in 0..<net.h2Size {
                dH2[j] += net.w3[i][j] * g
                net.w3[i][j] -= lr * g * a.h2[j]
            }
            net.b3[i] -= lr * g
        }
        // Layer 2
        var dH1 = [Double](repeating: 0, count: net.h1Size)
        for i in 0..<net.h2Size {
            let gz = (a.z2[i] > 0 ? 1.0 : 0.0) * net.maskH2[i] * dH2[i]
            if gz != 0 {
                for j in 0..<net.h1Size {
                    dH1[j] += net.w2[i][j] * gz
                    net.w2[i][j] -= lr * gz * a.h1[j]
                }
                net.b2[i] -= lr * gz
            }
        }
        // Layer 1
        for i in 0..<net.h1Size {
            let gz = (a.z1[i] > 0 ? 1.0 : 0.0) * net.maskH1[i] * dH1[i]
            if gz != 0 {
                for j in 0..<net.inputSize {
                    net.w1[i][j] -= lr * gz * x[j]
                }
                net.b1[i] -= lr * gz
            }
        }
    }
}
