//
//  RoutingLog.swift
//  A portable JSON schema for an external MoE engine's routing logs.
//
//  This is the bridge back to the original Colibrì reference: rather than wrapping their
//  engine, we define the log an engine like it *emits* (per-input, per-layer expert gates +
//  the prediction) and ingest it. Any tool that can write this shape gets the mapped,
//  labeled cortex + verification for free — without exposing its weights.
//
//  Example:
//  {
//    "model": "demo-moe",
//    "classes": ["weather","finance","food"],
//    "layers": 2, "experts": 6,
//    "records": [
//      { "id":"r0", "truth":"finance", "predicted":"finance", "display":"invest stock …",
//        "gates":[[0.05,0.20,0.03,0.04,0.62,0.06],[0.51,0.10,0.18,0.07,0.08,0.06]],
//        "probs":{"weather":0.01,"finance":0.98,"food":0.01} }
//    ]
//  }
//

import Foundation

public struct RoutingLog: Codable, Sendable {
    public var model: String
    public var classes: [String]
    public var layers: Int
    public var experts: Int
    public var records: [Record]

    public struct Record: Codable, Sendable {
        public var id: String
        public var truth: String?
        public var predicted: String
        public var display: String?
        public var gates: [[Double]]           // [layer][expert]
        public var probs: [String: Double]

        public init(id: String, truth: String?, predicted: String, display: String?,
                    gates: [[Double]], probs: [String: Double]) {
            self.id = id; self.truth = truth; self.predicted = predicted
            self.display = display; self.gates = gates; self.probs = probs
        }
    }

    public init(model: String, classes: [String], layers: Int, experts: Int, records: [Record]) {
        self.model = model; self.classes = classes
        self.layers = layers; self.experts = experts; self.records = records
    }

    /// The logged inputs, as browsable cartography inputs (payload is unused — the adapter
    /// replays each record by id).
    public var inputs: [CartographyInput] {
        records.map { CartographyInput(id: $0.id, truth: $0.truth, display: $0.display,
                                       payload: .vector([])) }
    }

    public func encoded() throws -> Data {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try enc.encode(self)
    }
    public static func decoded(from data: Data) throws -> RoutingLog {
        try JSONDecoder().decode(RoutingLog.self, from: data)
    }
}
