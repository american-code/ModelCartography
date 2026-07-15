//
//  MockNetworkAdapter.swift
//  Presents the pure-Swift MockNetwork through the universal ModelAdapter interface.
//
//  This adapter supports the full capability set — it is the reference implementation
//  that proves the whole pipeline, including true internal ablation and gradient saliency.
//

import Foundation

public final class MockNetworkAdapter: ModelAdapter {
    private let net: MockNetwork
    private let gridSide: Int

    public init(net: MockNetwork, gridSide: Int) {
        self.net = net
        self.gridSide = gridSide
    }

    public var name: String { "Demo Net · 8×8 → 16 → 16 → 3" }

    public var capabilities: Capabilities {
        [.internalActivations, .saliency, .ablation, .routingPath, .verification]
    }

    public var classLabels: [String] { net.classes }

    // MARK: Structure

    // Layer indices: 1 = h1, 2 = h2, 3 = output logits. (Input grid is layer 0, not
    // enumerated as regions — it's shown as the trace thumbnail instead.)
    public func regions() -> [Region] {
        var out: [Region] = []
        for i in 0..<net.h1Size {
            out.append(Region(id: "L1.N\(i)", kind: .neuron, layerIndex: 1, indexInLayer: i))
        }
        for i in 0..<net.h2Size {
            out.append(Region(id: "L2.N\(i)", kind: .neuron, layerIndex: 2, indexInLayer: i))
        }
        for (i, label) in net.classes.enumerated() {
            out.append(Region(id: "OUT.\(label)", kind: .logit, layerIndex: 3, indexInLayer: i))
        }
        return out
    }

    public func topology() -> [Edge] {
        var edges: [Edge] = []
        for a in 0..<net.h1Size {
            for b in 0..<net.h2Size { edges.append(Edge(from: "L1.N\(a)", to: "L2.N\(b)")) }
        }
        for b in 0..<net.h2Size {
            for (c, label) in net.classes.enumerated() {
                _ = c
                edges.append(Edge(from: "L2.N\(b)", to: "OUT.\(label)"))
            }
        }
        return edges
    }

    // MARK: Inference

    public func forward(_ input: CartographyInput) throws -> Trace {
        let x = input.flattened
        guard x.count == net.inputSize else {
            throw CartographyError.badInput("expected \(net.inputSize) values, got \(x.count)")
        }
        let a = net.activations(x)

        var acts: [String: Double] = [:]
        for i in 0..<net.h1Size { acts["L1.N\(i)"] = a.h1[i] }
        for i in 0..<net.h2Size { acts["L2.N\(i)"] = a.h2[i] }
        for (i, label) in net.classes.enumerated() { acts["OUT.\(label)"] = a.probs[i] }

        // Realized pathway: the strongest neuron at each layer -> predicted class.
        let topH1 = a.h1.indices.max(by: { a.h1[$0] < a.h1[$1] }) ?? 0
        let topH2 = a.h2.indices.max(by: { a.h2[$0] < a.h2[$1] }) ?? 0
        let predIdx = a.probs.indices.max(by: { a.probs[$0] < a.probs[$1] }) ?? 0
        let predicted = net.classes[predIdx]
        let path = ["L1.N\(topH1)", "L2.N\(topH2)", "OUT.\(predicted)"]

        var probs: [String: Double] = [:]
        for (i, label) in net.classes.enumerated() { probs[label] = a.probs[i] }

        return Trace(inputID: input.id, activations: acts, routingPath: path,
                     probabilities: probs, predicted: predicted)
    }

    public func saliency(for input: CartographyInput) throws -> SaliencyMap {
        let x = input.flattened
        guard x.count == net.inputSize else {
            throw CartographyError.badInput("expected \(net.inputSize) values")
        }
        let g = net.inputGradientMagnitude(x)
        let maxV = g.max() ?? 1
        let norm = maxV > 0 ? g.map { $0 / maxV } : g
        var grid = [[Double]](repeating: [Double](repeating: 0, count: gridSide), count: gridSide)
        for r in 0..<gridSide { for c in 0..<gridSide { grid[r][c] = norm[r * gridSide + c] } }
        return SaliencyMap(rows: gridSide, cols: gridSide, values: grid)
    }

    // MARK: Intervention

    public func ablated(regionIDs: Set<String>) throws -> ModelAdapter {
        let clone = net.copy()
        for id in regionIDs {
            if id.hasPrefix("L1.N"), let i = Int(id.dropFirst(4)), i < clone.h1Size {
                clone.maskH1[i] = 0
            } else if id.hasPrefix("L2.N"), let i = Int(id.dropFirst(4)), i < clone.h2Size {
                clone.maskH2[i] = 0
            }
            // Ablating a logit region is a no-op here (outputs aren't computation to remove).
        }
        return MockNetworkAdapter(net: clone, gridSide: gridSide)
    }

    // Direct access to per-region activation across a corpus — used by Attribution.
    public func rawActivation(regionID: String, input: CartographyInput) -> Double {
        (try? forward(input))?.activations[regionID] ?? 0
    }
}
