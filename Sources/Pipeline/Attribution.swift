//
//  Attribution.swift
//  Stage 3 of the pipeline: turn raw activations into a labeled map.
//
//  This is the lightweight, forward-only form of "class probes": for each region,
//  measure how its activation covaries with each class across the corpus, then label
//  the region with the class it most prefers and score how selective it is.
//  (The heavier Map B — sparse-autoencoder features — arrives in Phase 2.)
//

import Foundation

public enum Attribution {

    public struct Labeling {
        /// region id -> assigned domain label
        public var labels: [String: String]
        /// region id -> selectivity 0...1 (1 = fires for exactly one class)
        public var selectivity: [String: Double]
    }

    /// Label every internal region by the class that most excites it.
    public static func label(adapter: ModelAdapter,
                             corpus: [CartographyInput],
                             classLabels: [String]) -> Labeling {
        let regionIDs = adapter.regions()
            .filter { $0.kind == .neuron || $0.kind == .channel || $0.kind == .expert }
            .map(\.id)

        // Accumulate mean activation per (region, class).
        var sum: [String: [String: Double]] = [:]
        var count: [String: Int] = [:]
        for id in regionIDs { sum[id] = Dictionary(uniqueKeysWithValues: classLabels.map { ($0, 0.0) }) }
        for c in classLabels { count[c] = 0 }

        for input in corpus {
            guard let truth = input.truth, let trace = try? adapter.forward(input) else { continue }
            count[truth, default: 0] += 1
            for id in regionIDs {
                sum[id]?[truth, default: 0] += trace.activations[id] ?? 0
            }
        }

        var labels: [String: String] = [:]
        var selectivity: [String: Double] = [:]
        for id in regionIDs {
            var means: [(String, Double)] = []
            for c in classLabels {
                let n = max(1, count[c] ?? 1)
                means.append((c, (sum[id]?[c] ?? 0) / Double(n)))
            }
            let total = means.reduce(0) { $0 + max(0, $1.1) }
            let best = means.max(by: { $0.1 < $1.1 }) ?? ("—", 0)
            labels[id] = best.1 > 1e-6 ? best.0 : "(silent)"
            // Selectivity: share of total activation captured by the top class.
            selectivity[id] = total > 1e-9 ? max(0, best.1) / total : 0
        }
        return Labeling(labels: labels, selectivity: selectivity)
    }
}
