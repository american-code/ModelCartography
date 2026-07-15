//
//  MoEAdapter.swift
//  Presents our Mixture-of-Experts model through the universal interface.
//
//  For an MoE model the routing IS the map: regions are experts, a region's activation is
//  how much the router used it for this input, and the top expert per layer is the realized
//  pathway. Attribution over a corpus labels each expert with the domain that routes to it —
//  the "annotated brain" the design promised — and expert usage drives pruning of cold
//  experts and pinning of hot ones.
//

import Foundation

public final class MoEAdapter: ModelAdapter {
    private let model: MoEModel
    private let expertLabel: [[String]]     // [layer][e]
    private let expertSel: [[Double]]       // topic purity 0...1
    private let mask: [[Double]]?           // nil = intact; else pruned experts zeroed

    public var name: String {
        "MoE Model · \(model.layerCount)×\(model.expertCount) experts, top-routed"
    }
    public var classLabels: [String] { model.classes }

    public var capabilities: Capabilities {
        [.internalActivations, .routingPath, .ablation, .verification, .attribution, .logitLens]
    }

    private init(model: MoEModel, expertLabel: [[String]], expertSel: [[Double]], mask: [[Double]]?) {
        self.model = model; self.expertLabel = expertLabel; self.expertSel = expertSel; self.mask = mask
    }

    public static func make() -> MoEAdapter {
        let model = MoEModel(vocab: TopicDataset.vocabSize, hidden: 12, experts: 6, layers: 2,
                             classes: TopicDataset.topics, seed: 4)
        let train = TopicDataset.corpus(perTopic: 40, seed: 2)
        MoETrainer.train(model, corpus: train)
        let (labels, sels) = labelExperts(model: model, corpus: train)
        return MoEAdapter(model: model, expertLabel: labels, expertSel: sels, mask: nil)
    }

    // Attribute each expert to the topic whose inputs route to it most.
    private static func labelExperts(model: MoEModel, corpus: [CartographyInput])
    -> (labels: [[String]], sels: [[Double]]) {
        let L = model.layerCount, E = model.expertCount
        var mass = Array(repeating: Array(repeating: [String: Double](), count: E), count: L)
        var used = Array(repeating: Array(repeating: 0.0, count: E), count: L)
        for input in corpus {
            guard let t = input.truth else { continue }
            let f = model.forward(input.flattened)
            for l in 0..<L { for e in 0..<E { mass[l][e][t, default: 0] += f.gates[l][e]; used[l][e] += f.gates[l][e] } }
        }
        let n = Double(max(1, corpus.count))
        var labels = Array(repeating: Array(repeating: "(cold)", count: E), count: L)
        var sels = Array(repeating: Array(repeating: 0.0, count: E), count: L)
        for l in 0..<L {
            for e in 0..<E {
                let total = mass[l][e].values.reduce(0, +)
                if used[l][e] / n < 0.02 || total < 1e-6 { continue }   // cold expert
                guard let dom = mass[l][e].max(by: { $0.value < $1.value }) else { continue }
                labels[l][e] = dom.key
                sels[l][e] = dom.value / total
            }
        }
        return (labels, sels)
    }

    // MARK: Structure

    public func regions() -> [Region] {
        var out: [Region] = []
        for l in 0..<model.layerCount {
            for e in 0..<model.expertCount {
                out.append(Region(id: "L\(l).E\(e)", kind: .expert, layerIndex: l + 1,
                                  indexInLayer: e, label: expertLabel[l][e], selectivity: expertSel[l][e]))
            }
        }
        for (i, topic) in model.classes.enumerated() {
            out.append(Region(id: "OUT.\(topic)", kind: .logit, layerIndex: model.layerCount + 1,
                              indexInLayer: i, label: topic))
        }
        return out
    }

    public func topology() -> [Edge] {
        var edges: [Edge] = []
        for l in 0..<(model.layerCount - 1) {
            for a in 0..<model.expertCount { for b in 0..<model.expertCount {
                edges.append(Edge(from: "L\(l).E\(a)", to: "L\(l + 1).E\(b)"))
            } }
        }
        return edges
    }

    // MARK: Inference

    public func forward(_ input: CartographyInput) throws -> Trace {
        let x = input.flattened
        guard x.count == model.vocab else { throw CartographyError.badInput("expected \(model.vocab)-dim input") }
        let f = model.forward(x, mask: mask)

        var acts: [String: Double] = [:]
        var path: [String] = []
        for l in 0..<model.layerCount {
            for e in 0..<model.expertCount { acts["L\(l).E\(e)"] = f.gates[l][e] }
            let top = f.gates[l].indices.max(by: { f.gates[l][$0] < f.gates[l][$1] }) ?? 0
            path.append("L\(l).E\(top)")
        }
        let predIdx = f.probs.indices.max(by: { f.probs[$0] < f.probs[$1] }) ?? 0
        let predicted = model.classes[predIdx]
        path.append("OUT.\(predicted)")

        var probs: [String: Double] = [:]
        for (i, t) in model.classes.enumerated() {
            probs[t] = f.probs[i]; acts["OUT.\(t)"] = f.probs[i]
        }
        return Trace(inputID: input.id, activations: acts, routingPath: path,
                     probabilities: probs, predicted: predicted)
    }

    // MARK: Logit lens (head decoded at each residual point)

    public func logitLens(_ input: CartographyInput) throws -> [LayerReadout] {
        let f = model.forward(input.flattened, mask: mask)
        var out: [LayerReadout] = []
        for idx in 0...model.layerCount {
            let probs = model.readout(f.r[idx])
            var dict: [String: Double] = [:]
            for (i, t) in model.classes.enumerated() { dict[t] = probs[i] }
            let top = model.classes[probs.indices.max(by: { probs[$0] < probs[$1] }) ?? 0]
            let name = idx == 0 ? "input (r0)" : "MoE layer \(idx) (r\(idx))"
            out.append(LayerReadout(layerIndex: idx, name: name, probabilities: dict, top: top))
        }
        return out
    }

    // MARK: Attribution (per-expert contribution to the predicted logit)

    public func attributionGraph(_ input: CartographyInput) throws -> AttributionGraph {
        let f = model.forward(input.flattened, mask: mask)
        let predIdx = f.probs.indices.max(by: { f.probs[$0] < f.probs[$1] }) ?? 0
        let predicted = model.classes[predIdx]
        let head = model.wh[predIdx]

        var contribs: [AttributionGraph.Contribution] = []
        for l in 0..<model.layerCount {
            for e in 0..<model.expertCount {
                let ge = f.gates[l][e]
                if ge < 1e-6 { continue }
                var dot = 0.0
                let y = f.y[l][e]
                for j in 0..<model.vocab { dot += head[j] * y[j] }
                let value = ge * dot
                if abs(value) < 1e-6 { continue }
                contribs.append(.init(id: "L\(l).E\(e)", label: expertLabel[l][e], value: value))
            }
        }
        contribs.sort { abs($0.value) > abs($1.value) }
        return AttributionGraph(predicted: predicted, contributions: Array(contribs.prefix(6)))
    }

    // MARK: Intervention — prune experts

    public func ablated(regionIDs: Set<String>) throws -> ModelAdapter {
        var m = Array(repeating: Array(repeating: 1.0, count: model.expertCount), count: model.layerCount)
        for id in regionIDs {
            // id form "L{l}.E{e}"
            let parts = id.split(separator: ".")
            guard parts.count == 2, parts[0].hasPrefix("L"), parts[1].hasPrefix("E"),
                  let l = Int(parts[0].dropFirst()), let e = Int(parts[1].dropFirst()),
                  l < model.layerCount, e < model.expertCount else { continue }
            m[l][e] = 0
        }
        return MoEAdapter(model: model, expertLabel: expertLabel, expertSel: expertSel, mask: m)
    }
}
