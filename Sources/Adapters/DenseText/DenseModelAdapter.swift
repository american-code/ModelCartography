//
//  DenseModelAdapter.swift
//  Presents the dense residual model + its sparse autoencoder through the universal
//  interface — now with Map B (features), logit lens, attribution, and steering.
//
//  The cortex for this model is its FEATURES, not raw neurons: for a dense model the
//  meaningful, human-legible domains live in activation space, which is exactly what the
//  SAE recovers. That is the whole Phase 2 point.
//

import Foundation

public final class DenseModelAdapter: ModelAdapter {
    private let net: DenseTextModel
    private let sae: SparseAutoencoder
    private let featureLabel: [String]      // per feature m
    private let featureTopic: [String?]
    private let featureSel: [Double]
    private let inject: [Double]?           // steering vector added to r2, or nil

    private let featureLayer = 2            // r2

    public var name: String { "Dense Text Model · residual + SAE (\(sae.count) features)" }
    public var classLabels: [String] { net.classes }

    public var capabilities: Capabilities {
        [.internalActivations, .semanticFeatures, .steering,
         .routingPath, .verification, .logitLens, .attribution]
    }

    private init(net: DenseTextModel, sae: SparseAutoencoder,
                 featureLabel: [String], featureTopic: [String?], featureSel: [Double],
                 inject: [Double]?) {
        self.net = net; self.sae = sae
        self.featureLabel = featureLabel; self.featureTopic = featureTopic
        self.featureSel = featureSel; self.inject = inject
    }

    /// Build + train the model and its SAE, then auto-label the features.
    public static func make(features: Int = 16) -> DenseModelAdapter {
        let net = DenseTextModel(vocab: TopicDataset.vocabSize, hidden: 24,
                                 classes: TopicDataset.topics, seed: 1)
        let train = TopicDataset.corpus(perTopic: 60, seed: 2)
        DenseTextTrainer.train(net, corpus: train)

        let residuals = train.map { net.forward($0.flattened).r2 }
        let sae = SparseAutoencoder(dim: TopicDataset.vocabSize, count: features, seed: 3)
        sae.train(on: residuals)

        let (labels, topics, sels) = Self.labelFeatures(sae: sae, corpus: train, residuals: residuals)
        return DenseModelAdapter(net: net, sae: sae, featureLabel: labels,
                                 featureTopic: topics, featureSel: sels, inject: nil)
    }

    // MARK: Auto-labeling — name each feature by the inputs that most activate it.

    private static func labelFeatures(sae: SparseAutoencoder, corpus: [CartographyInput],
                                      residuals: [[Double]])
    -> (labels: [String], topics: [String?], sels: [Double]) {
        let codes = residuals.map { sae.encode($0) }
        var labels = [String](repeating: "(silent)", count: sae.count)
        var topics = [String?](repeating: nil, count: sae.count)
        var sels = [Double](repeating: 0, count: sae.count)

        for m in 0..<sae.count {
            var topicMass: [String: Double] = [:]
            var tokenWeight: [String: Double] = [:]
            for (i, input) in corpus.enumerated() {
                let f = codes[i][m]
                if f <= 1e-6 { continue }
                if let t = input.truth { topicMass[t, default: 0] += f }
                for tok in (input.display ?? "").split(separator: " ") {
                    tokenWeight[String(tok), default: 0] += f
                }
            }
            let total = topicMass.values.reduce(0, +)
            guard total > 1e-6, let dominant = topicMass.max(by: { $0.value < $1.value }) else { continue }
            let topWords = tokenWeight.sorted { $0.value > $1.value }.prefix(2).map(\.key)
            topics[m] = dominant.key
            sels[m] = dominant.value / total
            labels[m] = topWords.isEmpty ? dominant.key : "\(dominant.key): \(topWords.joined(separator: ", "))"
        }
        return (labels, topics, sels)
    }

    // MARK: Structure — features are the cortex

    public func regions() -> [Region] {
        var out: [Region] = []
        for m in 0..<sae.count {
            out.append(Region(id: "F\(m)", kind: .feature, layerIndex: featureLayer,
                              indexInLayer: m, label: featureLabel[m], selectivity: featureSel[m]))
        }
        for (i, topic) in net.classes.enumerated() {
            out.append(Region(id: "OUT.\(topic)", kind: .logit, layerIndex: featureLayer + 1,
                              indexInLayer: i, label: topic))
        }
        return out
    }

    public func topology() -> [Edge] {
        var edges: [Edge] = []
        for m in 0..<sae.count {
            for topic in net.classes { edges.append(Edge(from: "F\(m)", to: "OUT.\(topic)")) }
        }
        return edges
    }

    public func features(layerIndex: Int) -> [Feature] {
        guard layerIndex == featureLayer else { return [] }
        return (0..<sae.count).map {
            Feature(id: "F\($0)", layerIndex: featureLayer, direction: sae.direction($0),
                    label: featureLabel[$0])
        }
    }

    // MARK: Inference

    public func forward(_ input: CartographyInput) throws -> Trace {
        let x = input.flattened
        guard x.count == net.vocab else {
            throw CartographyError.badInput("expected \(net.vocab)-dim input")
        }
        let f = net.forward(x, inject: inject)
        let code = sae.encode(f.r2)

        var acts: [String: Double] = [:]
        for m in 0..<sae.count { acts["F\(m)"] = code[m] }
        for (i, topic) in net.classes.enumerated() { acts["OUT.\(topic)"] = f.probs[i] }

        let predIdx = f.probs.indices.max(by: { f.probs[$0] < f.probs[$1] }) ?? 0
        let predicted = net.classes[predIdx]
        let topFeature = code.indices.max(by: { code[$0] < code[$1] }) ?? 0

        var probs: [String: Double] = [:]
        for (i, t) in net.classes.enumerated() { probs[t] = f.probs[i] }

        return Trace(inputID: input.id, activations: acts,
                     routingPath: ["F\(topFeature)", "OUT.\(predicted)"],
                     probabilities: probs, predicted: predicted)
    }

    // MARK: Logit lens

    public func logitLens(_ input: CartographyInput) throws -> [LayerReadout] {
        let f = net.forward(input.flattened, inject: inject)
        let points: [(Int, String, [Double])] = [
            (0, "embed (r0)", f.r0),
            (1, "block 1 (r1)", f.r1),
            (2, "block 2 (r2)", f.r2),
        ]
        return points.map { (idx, nm, residual) in
            let probs = net.readout(residual)
            var dict: [String: Double] = [:]
            for (i, t) in net.classes.enumerated() { dict[t] = probs[i] }
            let top = net.classes[probs.indices.max(by: { probs[$0] < probs[$1] }) ?? 0]
            return LayerReadout(layerIndex: idx, name: nm, probabilities: dict, top: top)
        }
    }

    // MARK: Attribution graph

    public func attributionGraph(_ input: CartographyInput) throws -> AttributionGraph {
        let f = net.forward(input.flattened)          // base features, no steering
        let code = sae.encode(f.r2)
        let predIdx = f.probs.indices.max(by: { f.probs[$0] < f.probs[$1] }) ?? 0
        let predicted = net.classes[predIdx]
        let headRow = net.wh[predIdx]                  // wh[c] · dir(m) = per-feature push on this logit

        var contribs: [AttributionGraph.Contribution] = []
        for m in 0..<sae.count where code[m] > 1e-6 {
            let dir = sae.direction(m)
            var dot = 0.0
            for j in 0..<dir.count { dot += headRow[j] * dir[j] }
            let value = code[m] * dot
            if abs(value) < 1e-6 { continue }
            contribs.append(.init(id: "F\(m)", label: featureLabel[m], value: value))
        }
        contribs.sort { abs($0.value) > abs($1.value) }
        return AttributionGraph(predicted: predicted, contributions: Array(contribs.prefix(6)))
    }

    // MARK: Steering

    public func steered(featureID: String, gain: Double) throws -> ModelAdapter {
        guard featureID.hasPrefix("F"), let m = Int(featureID.dropFirst()), m < sae.count else {
            throw CartographyError.badInput("unknown feature \(featureID)")
        }
        let dir = sae.direction(m)
        let vec = dir.map { $0 * gain }
        return DenseModelAdapter(net: net, sae: sae, featureLabel: featureLabel,
                                 featureTopic: featureTopic, featureSel: featureSel, inject: vec)
    }
}
