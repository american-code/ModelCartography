//
//  InterpAdapter.swift
//  Adapts a HookedGPT2 (SwiftSci Interp) into the universal ModelAdapter interface.
//  Uses Interp's HookRegistry, LogitLens, and AttentionPatterns for mechanistic
//  interpretability on a tiny transformer trained on a real word corpus.
//
//  Weight lifecycle
//  ────────────────
//  1. On first launch: ToyTransformerTrainer.trainAndSave() runs ~300 SGD steps
//     over the Shakespeare-like word corpus and writes the weights to
//     Application Support as a safetensors file.
//  2. On subsequent launches: ToyTransformerTrainer.loadTrained() reads the
//     file via MLX loadArrays(url:) — the same call LocalModelLocator (Interp)
//     uses for real HuggingFace checkpoints.
//
//  Activation streaming
//  ────────────────────
//  An ActivationLogger (ActivationStreamReceiver + ActivationStreamSender pair)
//  is created alongside the model.  Every forward() call passes the logger's
//  HookRegistry to HookedGPT2.callAsFunction, so the sender's residPost hooks
//  stream live float16 activations to the receiver, which writes .bin + .json
//  files to Application Support/activations/.
//
//  Guarded to macOS/iOS because MLX (a transitive dependency of Interp) does
//  not support tvOS.  On tvOS the existing adapters continue to work as before.
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

    private let model:  HookedGPT2
    private let logger: ActivationLogger

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    private init(model: HookedGPT2, logger: ActivationLogger) {
        self.model  = model
        self.logger = logger
    }

    /// Builds a trained HookedGPT2.
    ///
    /// - If a saved checkpoint exists (from a previous run) it is loaded via
    ///   MLX `loadArrays(url:)` — the SafetensorsLoader pattern from Interp.
    /// - Otherwise the model is trained for 300 SGD steps on the Shakespeare
    ///   word corpus and the resulting weights are saved to disk.
    ///
    /// An ActivationLogger is started so that every subsequent forward() call
    /// streams residPost activations to disk via ActivationStreamReceiver.
    public static func make() -> InterpAdapter {
        let cfg = GPT2Config(
            nLayers:   nLayers,
            nHeads:    nHeads,
            dModel:    dModel,
            dMLP:      dMLP,
            vocabSize: vocab.count,
            seqLen:    seqLen
        )

        let gpt: HookedGPT2
        if ToyTransformerTrainer.hasSavedWeights,
           let loaded = try? ToyTransformerTrainer.loadTrained(config: cfg) {
            gpt = loaded
        } else {
            gpt = ToyTransformerTrainer.trainAndSave(
                config: cfg, vocab: vocab, steps: 300)
        }

        let logger = ActivationLogger(nLayers: nLayers)
        return InterpAdapter(model: gpt, logger: logger)
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

    public var name: String { "Hooked GPT-2 · trained (2L 2H d=16)" }

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

    /// Runs a forward pass using the shared ActivationLogger registry so that
    /// ActivationStreamSender hooks fire and stream residPost activations to the
    /// ActivationStreamReceiver running on a background thread.
    public func forward(_ input: CartographyInput) throws -> Trace {
        let tokens = mlxTokens(input)
        // Use logger.registry so the sender's residPost hooks capture activations.
        let logits = model(tokens, registry: logger.registry)
        logits.eval()

        let lastPos = tokens.shape[0] - 1
        let probs   = softmaxRow(logits[lastPos])

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

    // MARK: Activation patching sweep (IOI-style) via Interp.ActivationPatching

    public func patchingSweep(clean: CartographyInput,
                               corrupted: CartographyInput) throws -> PatchingMatrix {
        let cleanToks = mlxTokensPadded(clean)
        let corrToks  = mlxTokensPadded(corrupted)

        let cleanLen = Self.tokenIDs(clean.display ?? "").count
        let corrLen  = Self.tokenIDs(corrupted.display ?? "").count

        let cleanLogits = model(cleanToks, registry: HookRegistry())
        cleanLogits.eval()
        let cleanProbs = softmaxRow(cleanLogits[cleanLen - 1])
        let ioTokenID  = cleanProbs.indices.max(by: { cleanProbs[$0] < cleanProbs[$1] }) ?? 0

        let corrLogits = model(corrToks, registry: HookRegistry())
        corrLogits.eval()
        let corrProbs = softmaxRow(corrLogits[corrLen - 1])
        let sTokenID  = corrProbs.indices
            .sorted { corrProbs[$0] > corrProbs[$1] }
            .first(where: { $0 != ioTokenID })
            ?? ((ioTokenID + 1) % Self.vocab.count)

        let ioi = IOISpec(ioTokenID: ioTokenID, sTokenID: sTokenID,
                          targetPosition: min(cleanLen - 1, corrLen - 1))

        let hookPoints: [HookPoint] = (0..<Self.nLayers).flatMap { l in
            (0..<Self.nHeads).map { h in HookPoint.attnHeadOut(layer: l, head: h) }
        }
        let results = Interp.ActivationPatching.sweep(
            clean: cleanToks, corrupted: corrToks,
            model: model, ioi: ioi, hookPoints: hookPoints
        )

        var scores = Array(repeating: Array(repeating: 0.0, count: Self.nHeads),
                           count: Self.nLayers)
        for (idx, r) in results.enumerated() {
            let l = idx / Self.nHeads
            let h = idx % Self.nHeads
            scores[l][h] = Double(max(0, min(1, r.normalizedPatchingScore)))
        }
        return PatchingMatrix(layers: Self.nLayers, heads: Self.nHeads, scores: scores)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private func headID(_ layer: Int, _ head: Int) -> String { "L\(layer)H\(head)" }

    private func mlxTokens(_ input: CartographyInput) -> MLXArray {
        let ids = Self.tokenIDs(input.display ?? "")
        return MLXArray(ids.map { Int32($0) }, [ids.count])
    }

    private func mlxTokensPadded(_ input: CartographyInput) -> MLXArray {
        let ids = Self.tokenIDs(input.display ?? "")
        let padded = (ids + [Int](repeating: 0, count: max(0, Self.seqLen - ids.count)))
            .prefix(Self.seqLen)
        return MLXArray(padded.map { Int32($0) }, [padded.count])
    }

    private static func tokenIDs(_ text: String) -> [Int] {
        let words = text.lowercased().split(separator: " ").map(String.init)
        let mapped = words.prefix(seqLen).map { vocab.firstIndex(of: $0) ?? 0 }
        return Array(mapped.isEmpty ? [0] : mapped)
    }

    private func softmaxRow(_ arr: MLXArray) -> [Float] {
        arr.eval()
        let raw  = arr.asArray(Float.self)
        let maxV = raw.max() ?? 0
        let exps = raw.map { Foundation.exp($0 - maxV) }
        let sum  = exps.reduce(0, +)
        return exps.map { $0 / max(sum, 1e-9) }
    }
}

#endif // os(macOS) || os(iOS)
