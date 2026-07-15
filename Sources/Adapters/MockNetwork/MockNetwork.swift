//
//  MockNetwork.swift
//  A small, fully-observable feed-forward classifier implemented in pure Swift.
//
//  Why hand-roll a net instead of only wrapping Core ML? Because Core ML is a black
//  box for *internal* activations and gradients. To prove the full five-stage loop —
//  capture internal regions, attribute them, ablate a specific neuron, re-verify —
//  we need a model we can see all the way through. This is that model: the literal
//  "smallest real model" of Phase 1.
//
//  Architecture:  input(rows*cols) -> h1 (ReLU) -> h2 (ReLU) -> logits -> softmax
//

import Foundation

public final class MockNetwork {
    public let inputSize: Int
    public let h1Size: Int
    public let h2Size: Int
    public let classes: [String]

    // Weights: wN[out][in]. Biases: bN[out].
    public var w1: [[Double]]
    public var b1: [Double]
    public var w2: [[Double]]
    public var b2: [Double]
    public var w3: [[Double]]
    public var b3: [Double]

    // Per-neuron ablation masks (1 = active, 0 = removed). Ablation flips these.
    public var maskH1: [Double]
    public var maskH2: [Double]

    /// One forward pass, with everything needed for backprop retained.
    public struct Activations {
        public var input: [Double]
        public var z1: [Double]     // pre-activation h1
        public var h1: [Double]     // post-ReLU, post-mask
        public var z2: [Double]
        public var h2: [Double]
        public var logits: [Double]
        public var probs: [Double]
    }

    public init(inputSize: Int, h1Size: Int, h2Size: Int, classes: [String], seed: UInt64) {
        self.inputSize = inputSize
        self.h1Size = h1Size
        self.h2Size = h2Size
        self.classes = classes

        var rng = SplitMix64(seed: seed)
        func matrix(_ rows: Int, _ cols: Int, scale: Double) -> [[Double]] {
            (0..<rows).map { _ in (0..<cols).map { _ in rng.gaussian(scale) } }
        }
        // He-ish initialization.
        w1 = matrix(h1Size, inputSize, scale: (2.0 / Double(inputSize)).squareRoot())
        b1 = [Double](repeating: 0, count: h1Size)
        w2 = matrix(h2Size, h1Size, scale: (2.0 / Double(h1Size)).squareRoot())
        b2 = [Double](repeating: 0, count: h2Size)
        w3 = matrix(classes.count, h2Size, scale: (2.0 / Double(h2Size)).squareRoot())
        b3 = [Double](repeating: 0, count: classes.count)

        maskH1 = [Double](repeating: 1, count: h1Size)
        maskH2 = [Double](repeating: 1, count: h2Size)
    }

    /// Deep copy — used so ablation never mutates the original model.
    public func copy() -> MockNetwork {
        let n = MockNetwork(inputSize: inputSize, h1Size: h1Size, h2Size: h2Size,
                            classes: classes, seed: 0)
        n.w1 = w1; n.b1 = b1; n.w2 = w2; n.b2 = b2; n.w3 = w3; n.b3 = b3
        n.maskH1 = maskH1; n.maskH2 = maskH2
        return n
    }

    // MARK: Forward

    public func activations(_ x: [Double]) -> Activations {
        var z1 = b1, h1 = [Double](repeating: 0, count: h1Size)
        for i in 0..<h1Size {
            var s = b1[i]
            let row = w1[i]
            for j in 0..<inputSize { s += row[j] * x[j] }
            z1[i] = s
            h1[i] = max(0, s) * maskH1[i]
        }
        var z2 = b2, h2 = [Double](repeating: 0, count: h2Size)
        for i in 0..<h2Size {
            var s = b2[i]
            let row = w2[i]
            for j in 0..<h1Size { s += row[j] * h1[j] }
            z2[i] = s
            h2[i] = max(0, s) * maskH2[i]
        }
        var logits = b3
        for i in 0..<classes.count {
            var s = b3[i]
            let row = w3[i]
            for j in 0..<h2Size { s += row[j] * h2[j] }
            logits[i] = s
        }
        return Activations(input: x, z1: z1, h1: h1, z2: z2, h2: h2,
                           logits: logits, probs: MockNetwork.softmax(logits))
    }

    public static func softmax(_ logits: [Double]) -> [Double] {
        let m = logits.max() ?? 0
        let exps = logits.map { Foundation.exp($0 - m) }
        let sum = exps.reduce(0, +)
        return sum > 0 ? exps.map { $0 / sum } : exps
    }

    // MARK: Input-gradient saliency (true gradients — we own the graph)

    /// d(predicted logit)/d(input), absolute value per input dimension.
    public func inputGradientMagnitude(_ x: [Double]) -> [Double] {
        let a = activations(x)
        let target = a.probs.indices.max(by: { a.probs[$0] < a.probs[$1] }) ?? 0

        // Seed gradient at the target logit only.
        var dLogits = [Double](repeating: 0, count: classes.count)
        dLogits[target] = 1

        var dH2 = [Double](repeating: 0, count: h2Size)
        for i in 0..<classes.count {
            let g = dLogits[i]
            if g == 0 { continue }
            for j in 0..<h2Size { dH2[j] += w3[i][j] * g }
        }
        var dZ2 = [Double](repeating: 0, count: h2Size)
        for j in 0..<h2Size { dZ2[j] = (a.z2[j] > 0 ? 1 : 0) * maskH2[j] * dH2[j] }

        var dH1 = [Double](repeating: 0, count: h1Size)
        for i in 0..<h2Size {
            let g = dZ2[i]
            if g == 0 { continue }
            for j in 0..<h1Size { dH1[j] += w2[i][j] * g }
        }
        var dZ1 = [Double](repeating: 0, count: h1Size)
        for j in 0..<h1Size { dZ1[j] = (a.z1[j] > 0 ? 1 : 0) * maskH1[j] * dH1[j] }

        var dX = [Double](repeating: 0, count: inputSize)
        for i in 0..<h1Size {
            let g = dZ1[i]
            if g == 0 { continue }
            for j in 0..<inputSize { dX[j] += w1[i][j] * g }
        }
        return dX.map { Swift.abs($0) }
    }
}
