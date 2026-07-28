//
//  InterpAdapter.swift
//  Adapts a HookedGPT2 (SwiftSci Interp) into the universal ModelAdapter interface.
//  Uses Interp's HookRegistry, LogitLens, and AttentionPatterns for mechanistic
//  interpretability on a tiny demo transformer with deterministic weights.
//
//  Guarded to macOS/iOS because MLX (a transitive dependency of Interp) does not
//  support tvOS. On tvOS the existing adapters continue to work as before.
//

#if os(macOS) || os(iOS)

import Foundation
import MLX
import Interp

public final class InterpAdapter: ModelAdapter {

    // ── Vocabulary & model config ─────────────────────────────────────────────

    /// Demo vocabulary — small enough to display as class-label probability bars.
    public static let vocab: [String] = [
        "the", "fox", "cat", "dog", "bird",
        "jumped", "sat", "flew", "quick", "lazy"
    ]

    private static let nLayers = 2
    private static let nHeads  = 2
    private static let dModel  = 16
    private static let dMLP    = 32
    private static let seqLen  = 8

    private let model: HookedGPT2

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    private init(model: HookedGPT2) { self.model = model }

    /// Build a HookedGPT2 with small deterministic weights for demonstration.
    public static func make() -> InterpAdapter {
        let vSize = vocab.count
        let cfg = GPT2Config(
            nLayers: nLayers, nHeads: nHeads,
            dModel: dModel, dMLP: dMLP,
            vocabSize: vSize, seqLen: seqLen
        )
        let blocks: [GPT2BlockWeights] = (0..<nLayers).map { l in
            GPT2BlockWeights(
                ln1Scale: scalars(1.0, [dModel]),   ln1Bias: scalars(0.0, [dModel]),
                wQ: seeded([dModel, dModel], seed: l * 10 + 1),
                wK: seeded([dModel, dModel], seed: l * 10 + 2),
                wV: seeded([dModel, dModel], seed: l * 10 + 3),
                wO: seeded([dModel, dModel], seed: l * 10 + 4),
                ln2Scale: scalars(1.0, [dModel]),   ln2Bias: scalars(0.0, [dModel]),
                w1: seeded([dMLP, dModel],  seed: l * 10 + 5),
                b1: scalars(0.0, [dMLP]),
                w2: seeded([dModel, dMLP],  seed: l * 10 + 6),
                b2: scalars(0.0, [dModel])
            )
        }
        let gpt = HookedGPT2(
            config: cfg,
            wte: seeded([vSize, dModel], seed: 0),
            wpe: seeded([seqLen, dModel], seed: 99),
            blocks: blocks,
            lnFinalScale: scalars(1.0, [dModel]),
            lnFinalBias:  scalars(0.0, [dModel])
        )
        return InterpAdapter(model: gpt)
    }

    // ── Demo corpus ───────────────────────────────────────────────────────────

    public static func demoCorpus() -> [CartographyInput] {
        let sentences: [(id: String, text: String)] = [
            ("s0", "the quick fox jumped"),
            ("s1", "the lazy dog sat"),
            ("s2", "a bird flew quick"),
            ("s3", "the cat sat"),
            ("s4", "fox jumped quick"),
            ("s5", "the bird flew"),
        ]
        return sentences.map { s in
            CartographyInput(id: s.id, display: s.text,
                             payload: .vector(tokenIDs(s.text).map { Double($0) }))
        }
    }

    // ── ModelAdapter ──────────────────────────────────────────────────────────

    public var name: String { "Hooked GPT-2 · Interp demo (2L 2H d=16)" }

    public var classLabels: [String] { Self.vocab }

    public var capabilities: Capabilities {
        [.internalActivations, .logitLens, .attentionPatterns, .activationPatching]
    }

    public func regions() -> [Region] {
        var out: [Region] = []
        for l in 0..<Self.nLayers {
            for h in 0..<Self.nHeads {
                out.append(Region(id: headID(l, h), kind: .head,
                                  layerIndex: l, indexInLayer: h,
                                  label: "L\(l)H\(h)"))
            }
        }
        for (i, word) in Self.vocab.enumerated() {
            out.append(Region(id: "OUT.\(word)", kind: .logit,
                              layerIndex: Self.nLayers, indexInLayer: i,
                              label: word))
        }
        return out
    }

    public func topology() -> [Edge] {
        var edges: [Edge] = []
        for l in 0..<Self.nLayers {
            for h in 0..<Self.nHeads {
                let src = headID(l, h)
                for word in Self.vocab {
                    edges.append(Edge(from: src, to: "OUT.\(word)"))
                }
            }
        }
        return edges
    }

    public func forward(_ input: CartographyInput) throws -> Trace {
        let tokens = mlxTokens(input)
        let logits = model(tokens, registry: HookRegistry())
        logits.eval()

        let lastPos = tokens.shape[0] - 1
        let probs = softmaxRow(logits[lastPos])

        var probDict: [String: Double] = [:]
        for (i, word) in Self.vocab.enumerated() { probDict[word] = Double(probs[i]) }
        let topIdx   = probs.indices.max(by: { probs[$0] < probs[$1] }) ?? 0
        let predicted = Self.vocab[topIdx]

        // Mean attention magnitude per head as a proxy activation.
        let patResult = Interp.AttentionPatterns.run(tokens: tokens, model: model)
        patResult.patterns.eval()
        var acts: [String: Double] = [:]
        for l in 0..<Self.nLayers {
            for h in 0..<Self.nHeads {
                let pat  = patResult.patterns[l][h]
                pat.eval()
                let vals = pat.asArray(Float.self)
                acts[headID(l, h)] = Double(vals.reduce(0, +) / Float(max(vals.count, 1)))
            }
        }
        for (i, word) in Self.vocab.enumerated() {
            acts["OUT.\(word)"] = Double(probs[i])
        }

        let path = (0..<Self.nLayers).flatMap { l in
            (0..<Self.nHeads).map { headID(l, $0) }
        } + ["OUT.\(predicted)"]

        return Trace(inputID: input.id, activations: acts,
                     routingPath: path, probabilities: probDict, predicted: predicted)
    }

    // MARK: Logit lens via Interp.LogitLens

    public func logitLens(_ input: CartographyInput) throws -> [LayerReadout] {
        let tokens = mlxTokens(input)
        let result = Interp.LogitLens.run(tokens: tokens, model: model, vocab: Self.vocab)
        result.logits.eval()

        let lastPos = tokens.shape[0] - 1

        return (0..<Self.nLayers).map { l in
            let layerLogits = result.logits[l][lastPos]
            layerLogits.eval()
            let probs = softmaxRow(layerLogits)
            var probDict: [String: Double] = [:]
            for (i, word) in Self.vocab.enumerated() { probDict[word] = Double(probs[i]) }
            let topIdx = probs.indices.max(by: { probs[$0] < probs[$1] }) ?? 0
            return LayerReadout(layerIndex: l, name: "layer \(l)",
                                probabilities: probDict, top: Self.vocab[topIdx])
        }
    }

    // MARK: Attention patterns via Interp.AttentionPatterns

    public func attentionPatterns(_ input: CartographyInput) throws -> [AttentionHeadPattern] {
        let tokens = mlxTokens(input)
        let seqLen = tokens.shape[0]
        let result = Interp.AttentionPatterns.run(tokens: tokens, model: model)
        result.patterns.eval()

        var out: [AttentionHeadPattern] = []
        for l in 0..<Self.nLayers {
            for h in 0..<Self.nHeads {
                let pat = result.patterns[l][h]
                pat.eval()
                out.append(AttentionHeadPattern(
                    layerIndex: l, headIndex: h,
                    seqLen: seqLen, weights: pat.asArray(Float.self)
                ))
            }
        }
        return out
    }

    // MARK: Activation patching sweep (IOI-style)

    /// For each (layer, head), computes the total-variation distance between the clean and
    /// corrupted attention distributions, then normalizes to 0…1. Heads with high TVD shifted
    /// most between the two prompts and are the circuit's critical nodes.
    public func patchingSweep(clean: CartographyInput, corrupted: CartographyInput) throws -> PatchingMatrix {
        let cleanPats     = try attentionPatterns(clean)
        let corruptedPats = try attentionPatterns(corrupted)

        var scores   = Array(repeating: Array(repeating: 0.0, count: Self.nHeads), count: Self.nLayers)
        var maxScore = 0.0

        for l in 0..<Self.nLayers {
            for h in 0..<Self.nHeads {
                guard
                    let cp = cleanPats.first(where: { $0.layerIndex == l && $0.headIndex == h }),
                    let kp = corruptedPats.first(where: { $0.layerIndex == l && $0.headIndex == h })
                else { continue }
                let sLen = min(cp.seqLen, kp.seqLen)
                var tvd  = 0.0
                for row in 0..<sLen {
                    for col in 0..<sLen {
                        tvd += abs(Double(cp.weights[row * cp.seqLen + col]) -
                                   Double(kp.weights[row * kp.seqLen + col]))
                    }
                }
                let score = sLen > 0 ? tvd / Double(sLen) : 0
                scores[l][h] = score
                maxScore = max(maxScore, score)
            }
        }

        if maxScore > 0 {
            for l in scores.indices { for h in scores[l].indices { scores[l][h] /= maxScore } }
        }
        return PatchingMatrix(layers: Self.nLayers, heads: Self.nHeads, scores: scores)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private func headID(_ layer: Int, _ head: Int) -> String { "L\(layer)H\(head)" }

    /// Convert a CartographyInput to an MLXArray of Int32 token IDs, shape [seqLen].
    private func mlxTokens(_ input: CartographyInput) -> MLXArray {
        let ids = Self.tokenIDs(input.display ?? "")
        let ids32 = ids.map { Int32($0) }
        return MLXArray(ids32, [ids32.count])
    }

    /// Tokenize whitespace-separated text into vocab indices, clamped to seqLen.
    private static func tokenIDs(_ text: String) -> [Int] {
        let words = text.lowercased().split(separator: " ").map(String.init)
        let mapped = words.prefix(seqLen).map { vocab.firstIndex(of: $0) ?? 0 }
        return Array(mapped.isEmpty ? [0] : mapped)
    }

    /// Apply softmax to a 1-D MLXArray and return [Float].
    private func softmaxRow(_ arr: MLXArray) -> [Float] {
        arr.eval()
        let raw = arr.asArray(Float.self)
        let maxV = raw.max() ?? 0
        let exps = raw.map { Foundation.exp($0 - maxV) }
        let sum  = exps.reduce(0, +)
        return exps.map { $0 / max(sum, 1e-9) }
    }

    // ── Weight initializers ───────────────────────────────────────────────────

    /// All-`value` array of the given shape.
    private static func scalars(_ value: Float, _ shape: [Int]) -> MLXArray {
        MLXArray([Float](repeating: value, count: shape.reduce(1, *)), shape)
    }

    /// Deterministic pseudo-random weights in (−0.1, 0.1) keyed by `seed`.
    private static func seeded(_ shape: [Int], seed: Int) -> MLXArray {
        let n = shape.reduce(1, *)
        var data = [Float](repeating: 0, count: n)
        var state = UInt32(truncatingIfNeeded: seed &* 1664525 &+ 1013904223)
        for i in data.indices {
            state = state &* 1664525 &+ 1013904223
            // Map to (−0.1, 0.1)
            data[i] = (Float(state & 0xFFFF) / Float(0xFFFF) - 0.5) * 0.2
        }
        return MLXArray(data, shape)
    }
}

#endif // os(macOS) || os(iOS)
