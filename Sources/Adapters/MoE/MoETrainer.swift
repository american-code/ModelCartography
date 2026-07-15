//
//  MoETrainer.swift
//  Backprop through the router (softmax gating) and every expert, with residual skips.
//  Dense gating keeps it differentiable; per-element read-before-write keeps the residual
//  gradient correct while updating weights in place.
//

import Foundation

public struct MoETrainer {
    public static func train(_ m: MoEModel,
                             corpus: [CartographyInput],
                             epochs: Int = 70,
                             lr: Double = 0.15,
                             seed: UInt64 = 13) {
        var rng = SplitMix64(seed: seed)
        let classIndex = Dictionary(uniqueKeysWithValues: m.classes.enumerated().map { ($1, $0) })
        var order = Array(corpus.indices)
        for _ in 0..<epochs {
            for i in stride(from: order.count - 1, through: 1, by: -1) {
                let j = Int(rng.unit() * Double(i + 1)); order.swapAt(i, j)
            }
            for si in order {
                guard let t = corpus[si].truth, let target = classIndex[t] else { continue }
                step(m, x: corpus[si].flattened, target: target, lr: lr)
            }
        }
    }

    private static func step(_ m: MoEModel, x: [Double], target: Int, lr: Double) {
        let f = m.forward(x)
        let V = m.vocab, H = m.hidden, E = m.expertCount, C = m.classes.count
        let last = f.r[m.layerCount]

        var dLogits = f.probs; dLogits[target] -= 1

        // Head + dr (d of last residual)
        var dr = [Double](repeating: 0, count: V)
        for c in 0..<C {
            let dl = dLogits[c]
            for j in 0..<V { dr[j] += m.wh[c][j] * dl; m.wh[c][j] -= lr * dl * last[j] }
            m.bh[c] -= lr * dl
        }

        for l in stride(from: m.layerCount - 1, through: 0, by: -1) {
            let rIn = f.r[l]
            let g = f.gates[l]
            let dCombine = dr
            var drPrev = dr                       // residual skip contribution
            var dg = [Double](repeating: 0, count: E)

            for e in 0..<E {
                let yE = f.y[l][e], aE = f.a1[l][e], zE = f.z1[l][e]
                var dgE = 0.0
                for i in 0..<V { dgE += dCombine[i] * yE[i] }
                dg[e] = dgE
                let gE = g[e]

                // dY = gE * dCombine ; expert backward
                var dA = [Double](repeating: 0, count: H)
                for i in 0..<V {
                    let dyi = gE * dCombine[i]
                    for k in 0..<H {
                        dA[k] += m.w2[l][e][i][k] * dyi
                        m.w2[l][e][i][k] -= lr * dyi * aE[k]
                    }
                    m.b2[l][e][i] -= lr * dyi
                }
                for k in 0..<H {
                    let dzk = (zE[k] > 0 ? 1.0 : 0.0) * dA[k]
                    if dzk != 0 {
                        for j in 0..<V {
                            drPrev[j] += m.w1[l][e][k][j] * dzk
                            m.w1[l][e][k][j] -= lr * dzk * rIn[j]
                        }
                    }
                    m.b1[l][e][k] -= lr * dzk
                }
            }

            // Router: softmax backward, then update Wr / br
            var gdot = 0.0
            for e in 0..<E { gdot += g[e] * dg[e] }
            for e in 0..<E {
                let ds = g[e] * (dg[e] - gdot)
                if ds != 0 {
                    for j in 0..<V {
                        drPrev[j] += m.wr[l][e][j] * ds
                        m.wr[l][e][j] -= lr * ds * rIn[j]
                    }
                }
                m.br[l][e] -= lr * ds
            }
            dr = drPrev
        }
    }
}
