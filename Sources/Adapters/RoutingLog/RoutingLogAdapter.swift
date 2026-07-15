//
//  RoutingLogAdapter.swift
//  Presents an imported RoutingLog through the universal interface.
//
//  Honest capability set: a static log lets you map the experts, read their routing/
//  utilization, let attribution label them, and verify accuracy against the logged
//  predictions — but it carries no weights, so it declares no ablation, steering, logit
//  lens, or per-feature attribution. The UI degrades accordingly (Intervene shows the
//  utilization diagnostic read-only and explains what's missing).
//

import Foundation

public final class RoutingLogAdapter: ModelAdapter {
    private let log: RoutingLog
    private let byID: [String: RoutingLog.Record]

    public init(log: RoutingLog) {
        self.log = log
        self.byID = Dictionary(log.records.map { ($0.id, $0) }, uniquingKeysWith: { a, _ in a })
    }

    public var name: String { "Imported Routing Log · \(log.model)" }
    public var classLabels: [String] { log.classes }

    // No weights → mapping + routing + verification only. Attribution over the corpus
    // (which labels experts) still works because it only needs forward activations.
    public var capabilities: Capabilities { [.internalActivations, .routingPath, .verification] }

    /// The inputs this adapter can replay — the store uses these as its corpus.
    public var inputs: [CartographyInput] { log.inputs }

    public func regions() -> [Region] {
        var out: [Region] = []
        for l in 0..<log.layers {
            for e in 0..<log.experts {
                out.append(Region(id: "L\(l).E\(e)", kind: .expert, layerIndex: l + 1, indexInLayer: e))
            }
        }
        for (i, topic) in log.classes.enumerated() {
            out.append(Region(id: "OUT.\(topic)", kind: .logit, layerIndex: log.layers + 1, indexInLayer: i))
        }
        return out
    }

    public func forward(_ input: CartographyInput) throws -> Trace {
        guard let rec = byID[input.id] else {
            throw CartographyError.badInput("no log record for input \(input.id)")
        }
        var acts: [String: Double] = [:]
        var path: [String] = []
        for l in 0..<log.layers {
            let gates = l < rec.gates.count ? rec.gates[l] : []
            for e in 0..<log.experts { acts["L\(l).E\(e)"] = e < gates.count ? gates[e] : 0 }
            if let top = gates.indices.max(by: { gates[$0] < gates[$1] }) { path.append("L\(l).E\(top)") }
        }
        for (t, p) in rec.probs { acts["OUT.\(t)"] = p }
        path.append("OUT.\(rec.predicted)")
        return Trace(inputID: input.id, activations: acts, routingPath: path,
                     probabilities: rec.probs, predicted: rec.predicted)
    }

    // MARK: Build a sample log by exporting a freshly trained MoE, round-tripped through JSON.

    public static func sample() -> RoutingLogAdapter {
        let model = MoEModel(vocab: TopicDataset.vocabSize, hidden: 12, experts: 6, layers: 2,
                             classes: TopicDataset.topics, seed: 4)
        MoETrainer.train(model, corpus: TopicDataset.corpus(perTopic: 40, seed: 2))
        let corpus = TopicDataset.corpus(perTopic: 24, seed: 202)
        let log = RoutingLogExporter.export(model: model, name: "external-moe", corpus: corpus)
        // Prove the JSON path: encode then decode before adapting.
        let roundTripped = (try? RoutingLog.decoded(from: try log.encoded())) ?? log
        return RoutingLogAdapter(log: roundTripped)
    }
}
