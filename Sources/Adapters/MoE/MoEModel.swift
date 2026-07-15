//
//  MoEModel.swift
//  Our own small Mixture-of-Experts model — Colibrì's *method*, not its code.
//
//  Each MoE layer has a router that scores every expert for the current input and a set
//  of independent expert sub-networks; the layer's output is the gate-weighted mix of the
//  experts, added back as a residual. This is where the "route input -> the right
//  sub-network -> output" behavior is literally the architecture: the sequence of
//  top experts across layers is the realized pathway, and per-expert usage over a corpus
//  is the signal for pruning cold experts and pinning hot ones.
//
//  Trained with dense (soft) gating so it is differentiable; sparsity and top-k routing
//  are read off the learned gate weights at inference.
//

import Foundation

public final class MoEModel {
    public let vocab: Int          // V — residual width
    public let hidden: Int         // H — expert inner width
    public let expertCount: Int    // E
    public let layerCount: Int     // number of MoE layers
    public let classes: [String]

    // Per layer: router (E×V) + bias (E); experts each an MLP V->H->V.
    var wr: [[[Double]]]           // [layer][E][V]
    var br: [[Double]]             // [layer][E]
    var w1: [[[[Double]]]]         // [layer][E][H][V]
    var b1: [[[Double]]]           // [layer][E][H]
    var w2: [[[[Double]]]]         // [layer][E][V][H]
    var b2: [[[Double]]]           // [layer][E][V]
    var wh: [[Double]]             // head (C×V)
    var bh: [Double]

    public struct Forward {
        public var r: [[Double]]              // residual points: r[0]=x0 … r[layers]
        public var gates: [[Double]]          // [layer][E] router softmax (post-mask)
        var z1: [[[Double]]]                  // [layer][E][H] expert pre-activations
        var a1: [[[Double]]]                  // [layer][E][H] relu
        var y: [[[Double]]]                   // [layer][E][V] expert outputs
        public var logits: [Double]
        public var probs: [Double]
    }

    public init(vocab: Int, hidden: Int, experts: Int, layers: Int, classes: [String], seed: UInt64) {
        self.vocab = vocab; self.hidden = hidden; self.expertCount = experts
        self.layerCount = layers; self.classes = classes
        var rng = SplitMix64(seed: seed)
        func vec(_ n: Int, _ s: Double) -> [Double] { (0..<n).map { _ in rng.gaussian(s) } }
        func mat(_ r: Int, _ c: Int, _ s: Double) -> [[Double]] { (0..<r).map { _ in vec(c, s) } }

        let sv = (2.0 / Double(vocab)).squareRoot()
        let sh = (2.0 / Double(hidden)).squareRoot()
        wr = (0..<layers).map { _ in mat(experts, vocab, 0.2) }
        br = (0..<layers).map { _ in [Double](repeating: 0, count: experts) }
        w1 = (0..<layers).map { _ in (0..<experts).map { _ in mat(hidden, vocab, sv) } }
        b1 = (0..<layers).map { _ in (0..<experts).map { _ in [Double](repeating: 0, count: hidden) } }
        w2 = (0..<layers).map { _ in (0..<experts).map { _ in mat(vocab, hidden, sh) } }
        b2 = (0..<layers).map { _ in (0..<experts).map { _ in [Double](repeating: 0, count: vocab) } }
        wh = mat(classes.count, vocab, sv)
        bh = [Double](repeating: 0, count: classes.count)
    }

    static func softmax(_ v: [Double]) -> [Double] {
        let m = v.max() ?? 0
        let e = v.map { Foundation.exp($0 - m) }
        let s = e.reduce(0, +)
        return s > 0 ? e.map { $0 / s } : e
    }

    /// `mask[layer][e]` = 0 prunes expert e in that layer (its gate is removed, the rest
    /// renormalized). Pass nil for the intact model.
    public func forward(_ x: [Double], mask: [[Double]]? = nil) -> Forward {
        var r = x
        var rPoints: [[Double]] = [x]
        var gatesAll: [[Double]] = []
        var z1All: [[[Double]]] = [], a1All: [[[Double]]] = [], yAll: [[[Double]]] = []

        for l in 0..<layerCount {
            var s = br[l]
            for e in 0..<expertCount {
                var acc = br[l][e]
                let row = wr[l][e]
                for j in 0..<vocab { acc += row[j] * r[j] }
                s[e] = acc
            }
            var g = MoEModel.softmax(s)
            if let mask {
                for e in 0..<expertCount { g[e] *= mask[l][e] }
                let sum = g.reduce(0, +)
                if sum > 1e-12 { for e in 0..<expertCount { g[e] /= sum } }
            }

            var combine = [Double](repeating: 0, count: vocab)
            var z1L: [[Double]] = [], a1L: [[Double]] = [], yL: [[Double]] = []
            for e in 0..<expertCount {
                var z = b1[l][e]
                let W1 = w1[l][e]
                for k in 0..<hidden {
                    var acc = b1[l][e][k]
                    let row = W1[k]
                    for j in 0..<vocab { acc += row[j] * r[j] }
                    z[k] = acc
                }
                let a = z.map { max(0, $0) }
                var y = b2[l][e]
                let W2 = w2[l][e]
                for i in 0..<vocab {
                    var acc = b2[l][e][i]
                    let row = W2[i]
                    for k in 0..<hidden { acc += row[k] * a[k] }
                    y[i] = acc
                }
                let ge = g[e]
                if ge != 0 { for i in 0..<vocab { combine[i] += ge * y[i] } }
                z1L.append(z); a1L.append(a); yL.append(y)
            }
            for i in 0..<vocab { r[i] += combine[i] }   // residual
            rPoints.append(r)
            gatesAll.append(g); z1All.append(z1L); a1All.append(a1L); yAll.append(yL)
        }

        var logits = bh
        for c in 0..<classes.count {
            var acc = bh[c]
            let row = wh[c]
            for j in 0..<vocab { acc += row[j] * r[j] }
            logits[c] = acc
        }
        return Forward(r: rPoints, gates: gatesAll, z1: z1All, a1: a1All, y: yAll,
                       logits: logits, probs: MoEModel.softmax(logits))
    }

    /// Head applied to any residual point — logit lens across MoE layers.
    public func readout(_ residual: [Double]) -> [Double] {
        var logits = bh
        for c in 0..<classes.count {
            var acc = bh[c]
            for j in 0..<vocab { acc += wh[c][j] * residual[j] }
            logits[c] = acc
        }
        return MoEModel.softmax(logits)
    }
}
