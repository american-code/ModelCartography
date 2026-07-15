//
//  CartographyTypes.swift
//  Model Cartography — the normalized vocabulary every architecture is mapped into.
//
//  The core of the tool never touches a model directly. It only ever sees these
//  value types, produced by a per-architecture `ModelAdapter`. A Core ML classifier
//  and a 744B Mixture-of-Experts model both reduce to the same Map A (structure)
//  and, when affordable, Map B (semantics).
//

import Foundation

// MARK: - Input

/// A single thing you can run through a model. Kept deliberately architecture-neutral:
/// the adapter is responsible for turning this into whatever tensor its model wants.
public struct CartographyInput: Identifiable, Hashable, Sendable {
    public let id: String
    /// Ground-truth label, when known (used by the verification harness).
    public let truth: String?
    public let payload: Payload

    public enum Payload: Hashable, Sendable {
        case vector([Double])
        /// A grayscale grid, row-major, each value in 0...1. Doubles as a tiny image.
        case grid([[Double]])
    }

    public init(id: String, truth: String? = nil, payload: Payload) {
        self.id = id
        self.truth = truth
        self.payload = payload
    }

    /// Flattened numeric view, regardless of payload shape.
    public var flattened: [Double] {
        switch payload {
        case .vector(let v): return v
        case .grid(let g): return g.flatMap { $0 }
        }
    }

    public var gridSize: (rows: Int, cols: Int)? {
        if case .grid(let g) = payload, let first = g.first {
            return (g.count, first.count)
        }
        return nil
    }
}

// MARK: - Map A: structure

public enum RegionKind: String, Codable, Sendable, CaseIterable {
    case neuron      // dense hidden unit
    case channel     // conv channel / filter
    case expert      // MoE expert sub-network
    case head        // attention head
    case logit       // an output class
    case inputPatch  // a region of the input (receptive field)
}

/// A node in the model's map — a place where computation happens.
/// `label` and `selectivity` are filled in later by the Attribution stage.
public struct Region: Identifiable, Hashable, Sendable {
    public let id: String
    public let kind: RegionKind
    /// Depth in the network. Used to lay the cortex out in columns.
    public let layerIndex: Int
    /// Position within its layer. Used for the row within a column.
    public let indexInLayer: Int
    /// Human-legible domain assigned by attribution, e.g. "horizontal".
    public var label: String?
    /// How sharply this region prefers one class over others, 0...1.
    public var selectivity: Double?

    public init(id: String, kind: RegionKind, layerIndex: Int,
                indexInLayer: Int, label: String? = nil, selectivity: Double? = nil) {
        self.id = id
        self.kind = kind
        self.layerIndex = layerIndex
        self.indexInLayer = indexInLayer
        self.label = label
        self.selectivity = selectivity
    }
}

/// A directed connection two regions *can* use. (The realized path for one input
/// is a `Trace.routingPath`.)
public struct Edge: Hashable, Sendable {
    public let from: String
    public let to: String
    public init(from: String, to: String) {
        self.from = from
        self.to = to
    }
}

// MARK: - Map B: semantics (defined now, produced starting Phase 2)

/// An interpretable direction in a layer's activation space — the real "domain".
/// Phase 1 adapters return `[]`; the dense-LLM adapter in Phase 2 fills these via SAEs.
public struct Feature: Identifiable, Hashable, Sendable {
    public let id: String
    public let layerIndex: Int
    public let direction: [Double]
    public var label: String?
    public init(id: String, layerIndex: Int, direction: [Double], label: String? = nil) {
        self.id = id
        self.layerIndex = layerIndex
        self.direction = direction
        self.label = label
    }
}

// MARK: - Inference trace

/// The result of running one input: what lit up, the path it took, and the output.
public struct Trace: Sendable {
    public let inputID: String
    /// region id -> activation magnitude for this input.
    public let activations: [String: Double]
    /// Ordered region ids — the realized pathway (one hop per layer here; the literal
    /// expert sequence for an MoE model).
    public let routingPath: [String]
    /// class label -> probability.
    public let probabilities: [String: Double]
    public let predicted: String

    public init(inputID: String, activations: [String: Double],
                routingPath: [String], probabilities: [String: Double], predicted: String) {
        self.inputID = inputID
        self.activations = activations
        self.routingPath = routingPath
        self.probabilities = probabilities
        self.predicted = predicted
    }
}

/// Per-input importance over a grid input (occlusion or gradient based).
public struct SaliencyMap: Sendable {
    public let rows: Int
    public let cols: Int
    public let values: [[Double]]   // normalized 0...1
    public init(rows: Int, cols: Int, values: [[Double]]) {
        self.rows = rows
        self.cols = cols
        self.values = values
    }
}

// MARK: - Errors

public enum CartographyError: Error, CustomStringConvertible {
    case unsupported(String)
    case badInput(String)
    case modelLoad(String)

    public var description: String {
        switch self {
        case .unsupported(let what): return "This model does not support \(what)."
        case .badInput(let why): return "Bad input: \(why)."
        case .modelLoad(let why): return "Could not load model: \(why)."
        }
    }
}
