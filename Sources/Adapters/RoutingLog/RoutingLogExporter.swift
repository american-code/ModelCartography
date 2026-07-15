//
//  RoutingLogExporter.swift
//  Emits our MoE model's routing over a corpus into the portable RoutingLog schema —
//  standing in for a real external engine's log so the importer can be exercised end-to-end.
//

import Foundation

public enum RoutingLogExporter {
    public static func export(model: MoEModel, name: String,
                              corpus: [CartographyInput]) -> RoutingLog {
        var records: [RoutingLog.Record] = []
        for input in corpus {
            let f = model.forward(input.flattened)
            let predIdx = f.probs.indices.max(by: { f.probs[$0] < f.probs[$1] }) ?? 0
            var probs: [String: Double] = [:]
            for (i, t) in model.classes.enumerated() { probs[t] = f.probs[i] }
            records.append(.init(id: input.id, truth: input.truth,
                                 predicted: model.classes[predIdx], display: input.display,
                                 gates: f.gates, probs: probs))
        }
        return RoutingLog(model: name, classes: model.classes,
                          layers: model.layerCount, experts: model.expertCount, records: records)
    }
}
