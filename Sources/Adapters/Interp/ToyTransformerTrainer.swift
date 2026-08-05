//
//  ToyTransformerTrainer.swift
//  Trains the toy HookedGPT2 on a Shakespeare-like word corpus, then saves /
//  loads the resulting weights as safetensors so the app always starts from
//  a trained checkpoint rather than random noise.
//
//  Weight persistence uses MLX's loadArrays(url:) / save(arrays:url:) — the
//  same functions that SafetensorsLoader (SwiftSci Interp) uses for real
//  Hugging Face checkpoints.
//
//  Platform guard: MLX is only available on macOS and iOS.
//

#if os(macOS) || os(iOS)

import Foundation
import MLX
import Interp

// MARK: - Safetensors key names (must be consistent between pack and unpack)

private enum WK {
    static let wte          = "wte"
    static let wpe          = "wpe"
    static let lnFinalScale = "ln_f.weight"
    static let lnFinalBias  = "ln_f.bias"
    static func ln1Scale(_ l: Int) -> String { "blocks.\(l).ln1.weight" }
    static func ln1Bias(_ l: Int)  -> String { "blocks.\(l).ln1.bias" }
    static func wQ(_ l: Int)       -> String { "blocks.\(l).attn.wQ" }
    static func wK(_ l: Int)       -> String { "blocks.\(l).attn.wK" }
    static func wV(_ l: Int)       -> String { "blocks.\(l).attn.wV" }
    static func wO(_ l: Int)       -> String { "blocks.\(l).attn.wO" }
    static func ln2Scale(_ l: Int) -> String { "blocks.\(l).ln2.weight" }
    static func ln2Bias(_ l: Int)  -> String { "blocks.\(l).ln2.bias" }
    static func w1(_ l: Int)       -> String { "blocks.\(l).mlp.w1" }
    static func b1(_ l: Int)       -> String { "blocks.\(l).mlp.b1" }
    static func w2(_ l: Int)       -> String { "blocks.\(l).mlp.w2" }
    static func b2(_ l: Int)       -> String { "blocks.\(l).mlp.b2" }
}

// MARK: - ToyTransformerTrainer

/// Trains, saves, and loads the toy HookedGPT2 that backs InterpAdapter.
/// Each method is a static utility; no state is held.
enum ToyTransformerTrainer {

    // MARK: Persistence helpers

    /// URL for the trained weights in the app's Application Support directory.
    static var savedWeightsURL: URL {
        let support = FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let dir = support.appendingPathComponent(
            "org.americancode.cartography", isDirectory: true)
        try? FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("toy_gpt2_trained.safetensors")
    }

    static var hasSavedWeights: Bool {
        FileManager.default.fileExists(atPath: savedWeightsURL.path)
    }

    // MARK: SafetensorsLoader-pattern load

    /// Loads a trained HookedGPT2 from disk.
    /// Uses MLX `loadArrays(url:)` — the same call `LocalModelLocator.loadAllShards`
    /// (SwiftSci Interp's SafetensorsLoader) uses for real HF checkpoints.
    static func loadTrained(config: GPT2Config) throws -> HookedGPT2 {
        let tensors = try loadArrays(url: savedWeightsURL)
        return try buildFromDict(tensors, config: config)
    }

    // MARK: Train → save → return

    /// Trains from scratch and persists the result.
    static func trainAndSave(config: GPT2Config, vocab: [String],
                              steps: Int = 300) -> HookedGPT2 {
        let model = train(config: config, vocab: vocab, steps: steps)
        let dict  = packWeights(model, config: config)
        try? save(arrays: dict, url: savedWeightsURL)
        return model
    }

    // MARK: Training loop (MLX autograd + manual SGD)

    /// Trains the model with next-token prediction on a Shakespeare-like word corpus.
    /// Uses MLX `valueAndGrad` to compute gradients through the full transformer
    /// forward pass, then applies vanilla SGD for `steps` iterations.
    static func train(config: GPT2Config, vocab: [String],
                      steps: Int = 300) -> HookedGPT2 {
        var weights = initWeights(config: config)
        let corpus  = shakespeareCorpus(vocab: vocab, seqLen: config.seqLen)
        let lr      = Float(3e-3)

        for step in 0..<steps {
            let tokenIDs = corpus[step % corpus.count]
            let tokens   = MLXArray(tokenIDs, [tokenIDs.count])

            // Build the value-and-gradient function. Captures `tokens` and
            // `config` by value (both are Sendable); `ws` carries all weights.
            let capturedTokens = tokens
            let capturedConfig = config
            let vg = valueAndGrad { (ws: [MLXArray]) -> [MLXArray] in
                [Self.computeLoss(weights: ws,
                                  tokens: capturedTokens,
                                  config: capturedConfig)]
            }

            let (_, grads) = vg(weights)
            let lrArr = MLXArray(lr)
            for i in weights.indices {
                weights[i] = weights[i] - lrArr * grads[i]
            }
            eval(weights)
        }

        return buildModel(weights, config: config)
    }

    // MARK: Shakespeare-like corpus

    /// Word sequences constructed from the toy vocabulary that teach the model
    /// real co-occurrence statistics (subject-verb-object patterns in English).
    /// These are the training documents — the model is not random noise after training.
    static func shakespeareCorpus(vocab: [String], seqLen: Int) -> [[Int32]] {
        let sentences: [String] = [
            "the quick fox jumped the lazy dog",
            "the lazy dog sat the quick fox",
            "the quick cat jumped the lazy bird",
            "the lazy cat sat the quick dog",
            "the bird flew the quick fox jumped",
            "the quick fox jumped quick the bird",
            "the lazy dog sat cat the fox",
            "bird flew quick the lazy cat sat",
            "the quick bird flew the lazy fox",
            "fox jumped quick the lazy dog sat",
            "the quick cat sat the lazy fox",
            "the lazy bird flew the quick cat",
            "quick fox jumped the lazy bird flew",
            "the lazy cat the quick dog jumped",
            "bird flew the lazy dog cat sat",
            "the quick dog jumped the lazy cat",
            "fox jumped the quick cat bird flew",
            "the lazy fox the quick bird flew",
        ]

        let wordIndex = Dictionary(
            uniqueKeysWithValues: vocab.enumerated().map { ($1, $0) })

        return sentences.map { sentence in
            let ids = sentence.split(separator: " ")
                .compactMap { wordIndex[String($0)] }
                .map { Int32($0) }
            if ids.count >= seqLen {
                return Array(ids.prefix(seqLen))
            }
            return ids + [Int32](repeating: 0, count: seqLen - ids.count)
        }
    }

    // MARK: Loss (called inside valueAndGrad)

    private static func computeLoss(weights: [MLXArray],
                                    tokens: MLXArray,
                                    config: GPT2Config) -> MLXArray {
        let model   = buildModel(weights, config: config)
        let logits  = model(tokens, registry: HookRegistry())   // [seqLen, vocabSize]
        let seqLen  = tokens.shape[0]
        guard seqLen > 1 else { return MLXArray(Float(0)) }

        let n            = seqLen - 1
        let inputLogits  = logits[0..<n]                        // [n, vocabSize]
        let targets      = tokens[1..<seqLen].asType(.int32)    // [n]

        // Numerically stable cross-entropy with a gather over the flat logit matrix.
        let maxL         = inputLogits.max(axis: -1, keepDims: true)
        let shifted      = inputLogits - maxL
        let logZ         = log(exp(shifted).sum(axis: -1, keepDims: true))
        let logProbs     = shifted - logZ                       // [n, vocabSize]

        let vocabSize    = config.vocabSize
        let flatLogProbs = logProbs.reshaped([n * vocabSize])
        let bases        = MLXArray(
            Array(0..<n).map { Int32($0 * vocabSize) }, [n])
        let flatIdx      = bases + targets                      // [n]
        let selectedLP   = flatLogProbs[flatIdx]                // [n]
        return -selectedLP.mean()
    }

    // MARK: Weight layout helpers (flat [MLXArray] ↔ HookedGPT2)
    //
    // Layout (fixed index order — training + buildModel must agree):
    //   0: wte, 1: wpe, then per block: ln1Scale, ln1Bias, wQ, wK, wV, wO,
    //   ln2Scale, ln2Bias, w1, b1, w2, b2,  finally lnFinalScale, lnFinalBias.

    private static func initWeights(config: GPT2Config) -> [MLXArray] {
        // Deterministic initialisation using the same LCG as the old seeded() helper.
        func seeded(_ shape: [Int], seed: Int) -> MLXArray {
            let n = shape.reduce(1, *)
            var data  = [Float](repeating: 0, count: n)
            var state = UInt32(truncatingIfNeeded: seed &* 1664525 &+ 1013904223)
            for i in data.indices {
                state    = state &* 1664525 &+ 1013904223
                data[i]  = (Float(state & 0xFFFF) / Float(0xFFFF) - 0.5) * 0.2
            }
            return MLXArray(data, shape)
        }
        func ones(_ n: Int) -> MLXArray {
            MLXArray([Float](repeating: 1, count: n), [n])
        }
        func zeros(_ n: Int) -> MLXArray {
            MLXArray([Float](repeating: 0, count: n), [n])
        }

        var w: [MLXArray] = [
            seeded([config.vocabSize, config.dModel], seed: 0),
            seeded([config.seqLen,    config.dModel], seed: 99),
        ]
        for l in 0..<config.nLayers {
            w += [
                ones(config.dModel),
                zeros(config.dModel),
                seeded([config.dModel, config.dModel], seed: l * 10 + 1),
                seeded([config.dModel, config.dModel], seed: l * 10 + 2),
                seeded([config.dModel, config.dModel], seed: l * 10 + 3),
                seeded([config.dModel, config.dModel], seed: l * 10 + 4),
                ones(config.dModel),
                zeros(config.dModel),
                seeded([config.dMLP,   config.dModel], seed: l * 10 + 5),
                zeros(config.dMLP),
                seeded([config.dModel, config.dMLP],  seed: l * 10 + 6),
                zeros(config.dModel),
            ]
        }
        w += [ones(config.dModel), zeros(config.dModel)]
        return w
    }

    private static func buildModel(_ weights: [MLXArray],
                                   config: GPT2Config) -> HookedGPT2 {
        var i  = 0
        let wte = weights[i]; i += 1
        let wpe = weights[i]; i += 1
        var blocks: [GPT2BlockWeights] = []
        for _ in 0..<config.nLayers {
            let ln1Scale = weights[i]; i += 1
            let ln1Bias  = weights[i]; i += 1
            let wQ       = weights[i]; i += 1
            let wK       = weights[i]; i += 1
            let wV       = weights[i]; i += 1
            let wO       = weights[i]; i += 1
            let ln2Scale = weights[i]; i += 1
            let ln2Bias  = weights[i]; i += 1
            let w1       = weights[i]; i += 1
            let b1       = weights[i]; i += 1
            let w2       = weights[i]; i += 1
            let b2       = weights[i]; i += 1
            blocks.append(GPT2BlockWeights(
                ln1Scale: ln1Scale, ln1Bias: ln1Bias,
                wQ: wQ, wK: wK, wV: wV, wO: wO,
                ln2Scale: ln2Scale, ln2Bias: ln2Bias,
                w1: w1, b1: b1, w2: w2, b2: b2
            ))
        }
        let lnFinalScale = weights[i]; i += 1
        let lnFinalBias  = weights[i]
        return HookedGPT2(config: config, wte: wte, wpe: wpe, blocks: blocks,
                          lnFinalScale: lnFinalScale, lnFinalBias: lnFinalBias)
    }

    // MARK: Weight dict helpers (safetensors round-trip)

    static func packWeights(_ model: HookedGPT2,
                            config: GPT2Config) -> [String: MLXArray] {
        var d: [String: MLXArray] = [
            WK.wte:          model.wte,
            WK.wpe:          model.wpe,
            WK.lnFinalScale: model.lnFinalScale,
            WK.lnFinalBias:  model.lnFinalBias,
        ]
        for (l, blk) in model.blocks.enumerated() {
            d[WK.ln1Scale(l)] = blk.ln1Scale
            d[WK.ln1Bias(l)]  = blk.ln1Bias
            d[WK.wQ(l)]       = blk.wQ
            d[WK.wK(l)]       = blk.wK
            d[WK.wV(l)]       = blk.wV
            d[WK.wO(l)]       = blk.wO
            d[WK.ln2Scale(l)] = blk.ln2Scale
            d[WK.ln2Bias(l)]  = blk.ln2Bias
            d[WK.w1(l)]       = blk.w1
            d[WK.b1(l)]       = blk.b1
            d[WK.w2(l)]       = blk.w2
            d[WK.b2(l)]       = blk.b2
        }
        return d
    }

    private static func buildFromDict(_ d: [String: MLXArray],
                                      config: GPT2Config) throws -> HookedGPT2 {
        func t(_ key: String) throws -> MLXArray {
            guard let a = d[key] else { throw ModelLoadError.missingTensor(key) }
            return a.asType(.float32)
        }
        var blocks: [GPT2BlockWeights] = []
        for l in 0..<config.nLayers {
            blocks.append(try GPT2BlockWeights(
                ln1Scale: t(WK.ln1Scale(l)), ln1Bias: t(WK.ln1Bias(l)),
                wQ:       t(WK.wQ(l)),       wK:      t(WK.wK(l)),
                wV:       t(WK.wV(l)),       wO:      t(WK.wO(l)),
                ln2Scale: t(WK.ln2Scale(l)), ln2Bias: t(WK.ln2Bias(l)),
                w1:       t(WK.w1(l)),       b1:      t(WK.b1(l)),
                w2:       t(WK.w2(l)),       b2:      t(WK.b2(l))
            ))
        }
        return HookedGPT2(
            config:       config,
            wte:          try t(WK.wte),
            wpe:          try t(WK.wpe),
            blocks:       blocks,
            lnFinalScale: try t(WK.lnFinalScale),
            lnFinalBias:  try t(WK.lnFinalBias)
        )
    }
}

#endif // os(macOS) || os(iOS)
