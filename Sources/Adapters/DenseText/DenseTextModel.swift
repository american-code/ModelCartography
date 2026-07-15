//
//  DenseTextModel.swift
//  A small dense classifier with a genuine residual stream — the "dense LLM" stand-in.
//
//  Shape:  x0 (bag-of-words, dim V)
//          r1 = r0 + Wo1 · ReLU(W1 · r0 + b1)     residual block 1
//          r2 = r1 + Wo2 · ReLU(W2 · r1 + b2)     residual block 2
//          logits = Wh · r2 + bh                   classifier head
//
//  Because every residual point r0/r1/r2 shares dimension V, the SAME head can be
//  applied to each — that is the logit lens. Steering adds a direction to r2 before
//  the head.
//

import Foundation

public final class DenseTextModel {
    public let vocab: Int          // V, residual-stream width
    public let hidden: Int         // MLP inner width
    public let classes: [String]

    var w1: [[Double]]; var b1: [Double]; var wo1: [[Double]]   // block 1
    var w2: [[Double]]; var b2: [Double]; var wo2: [[Double]]   // block 2
    var wh: [[Double]]; var bh: [Double]                        // head

    public struct Forward {
        public var r0: [Double]
        public var r1: [Double]
        public var r2: [Double]
        public var logits: [Double]
        public var probs: [Double]
        // pre-activations retained for backprop
        var z1: [Double]; var a1: [Double]
        var z2: [Double]; var a2: [Double]
    }

    public init(vocab: Int, hidden: Int, classes: [String], seed: UInt64) {
        self.vocab = vocab; self.hidden = hidden; self.classes = classes
        var rng = SplitMix64(seed: seed)
        func mat(_ r: Int, _ c: Int, _ s: Double) -> [[Double]] {
            (0..<r).map { _ in (0..<c).map { _ in rng.gaussian(s) } }
        }
        let sv = (2.0 / Double(vocab)).squareRoot()
        let sh = (2.0 / Double(hidden)).squareRoot()
        w1 = mat(hidden, vocab, sv); b1 = .init(repeating: 0, count: hidden); wo1 = mat(vocab, hidden, sh)
        w2 = mat(hidden, vocab, sv); b2 = .init(repeating: 0, count: hidden); wo2 = mat(vocab, hidden, sh)
        wh = mat(classes.count, vocab, sv); bh = .init(repeating: 0, count: classes.count)
    }

    // MARK: Linear algebra helpers

    static func matVec(_ m: [[Double]], _ v: [Double]) -> [Double] {
        m.map { row in var s = 0.0; for j in 0..<v.count { s += row[j] * v[j] }; return s }
    }
    static func add(_ a: [Double], _ b: [Double]) -> [Double] {
        var o = a; for i in 0..<b.count { o[i] += b[i] }; return o
    }
    static func relu(_ v: [Double]) -> [Double] { v.map { max(0, $0) } }

    public static func softmax(_ logits: [Double]) -> [Double] {
        let m = logits.max() ?? 0
        let e = logits.map { Foundation.exp($0 - m) }
        let s = e.reduce(0, +)
        return s > 0 ? e.map { $0 / s } : e
    }

    // MARK: Forward

    /// Optional `inject` is added to r2 before the head — the steering hook.
    public func forward(_ x: [Double], inject: [Double]? = nil) -> Forward {
        let r0 = x
        let z1 = DenseTextModel.add(DenseTextModel.matVec(w1, r0), b1)
        let a1 = DenseTextModel.relu(z1)
        let r1 = DenseTextModel.add(r0, DenseTextModel.matVec(wo1, a1))

        let z2 = DenseTextModel.add(DenseTextModel.matVec(w2, r1), b2)
        let a2 = DenseTextModel.relu(z2)
        var r2 = DenseTextModel.add(r1, DenseTextModel.matVec(wo2, a2))
        if let inject { for i in 0..<r2.count { r2[i] += inject[i] } }

        let logits = DenseTextModel.add(DenseTextModel.matVec(wh, r2), bh)
        return Forward(r0: r0, r1: r1, r2: r2, logits: logits,
                       probs: DenseTextModel.softmax(logits),
                       z1: z1, a1: a1, z2: z2, a2: a2)
    }

    /// Apply the classifier head to an arbitrary residual vector — the logit lens.
    public func readout(_ residual: [Double]) -> [Double] {
        DenseTextModel.softmax(DenseTextModel.add(DenseTextModel.matVec(wh, residual), bh))
    }
}
