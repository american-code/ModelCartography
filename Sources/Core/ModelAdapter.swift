//
//  ModelAdapter.swift
//  The one interface every architecture must implement.
//
//  The whole "works for every model, large and small" claim rests here: the app's
//  UI, tracer, and optimizer are written once against this protocol. Each adapter
//  DECLARES what it can honestly do via `capabilities`, so the tool degrades
//  gracefully instead of faking a capability a given model can't support.
//

import Foundation

public struct Capabilities: OptionSet, Sendable {
    public let rawValue: Int
    public init(rawValue: Int) { self.rawValue = rawValue }

    /// Can expose activations of *internal* regions, not just outputs.
    public static let internalActivations = Capabilities(rawValue: 1 << 0)
    /// Can produce a per-input saliency map.
    public static let saliency            = Capabilities(rawValue: 1 << 1)
    /// Can truly remove/zero an internal region and return a modified model.
    public static let ablation            = Capabilities(rawValue: 1 << 2)
    /// Can add a direction at runtime to steer behavior.
    public static let steering            = Capabilities(rawValue: 1 << 3)
    /// Can recover interpretable features (Map B).
    public static let semanticFeatures    = Capabilities(rawValue: 1 << 4)
    /// Emits a realized routing path per input.
    public static let routingPath         = Capabilities(rawValue: 1 << 5)
    /// Can be re-run over a corpus to verify an intervention.
    public static let verification        = Capabilities(rawValue: 1 << 6)
    /// Can decode the running prediction at each layer (logit lens).
    public static let logitLens           = Capabilities(rawValue: 1 << 7)
    /// Can attribute an output back to contributing features.
    public static let attribution         = Capabilities(rawValue: 1 << 8)
    /// Can extract post-softmax attention weights via the Interp hook system.
    public static let attentionPatterns   = Capabilities(rawValue: 1 << 9)
    /// Can run an activation patching sweep to produce an IOI-style [layers × heads] importance matrix.
    public static let activationPatching  = Capabilities(rawValue: 1 << 10)
}

/// A model, normalized into the cartography vocabulary.
///
/// Reference type: an adapter owns mutable model state (weights, masks). Everything
/// runs on the main actor in this app, so the protocol is not `Sendable`.
public protocol ModelAdapter: AnyObject {
    var name: String { get }
    var capabilities: Capabilities { get }
    var classLabels: [String] { get }

    /// Map A — the regions that make up the cortex.
    func regions() -> [Region]
    /// How regions can feed one another.
    func topology() -> [Edge]

    /// Run one input; hand back what lit up, the path, and the output.
    func forward(_ input: CartographyInput) throws -> Trace

    /// Per-input importance. Only meaningful when `.saliency` is present.
    func saliency(for input: CartographyInput) throws -> SaliencyMap

    /// Map B — interpretable directions in a layer. Empty unless `.semanticFeatures`.
    func features(layerIndex: Int) -> [Feature]

    /// Return a copy of this model with the given regions removed. Only when `.ablation`.
    func ablated(regionIDs: Set<String>) throws -> ModelAdapter

    /// Return a copy that adds `gain` × the feature's direction at inference. Only when `.steering`.
    func steered(featureID: String, gain: Double) throws -> ModelAdapter

    /// The running prediction decoded at each layer. Only when `.logitLens`.
    func logitLens(_ input: CartographyInput) throws -> [LayerReadout]

    /// How this input's output decomposes over features. Only when `.attribution`.
    func attributionGraph(_ input: CartographyInput) throws -> AttributionGraph

    /// Post-softmax attention weights for every head in every layer. Only when `.attentionPatterns`.
    func attentionPatterns(_ input: CartographyInput) throws -> [AttentionHeadPattern]

    /// IOI-style sweep: for each head, measure how much its attention pattern shifts between
    /// `clean` and `corrupted` inputs. Returns a normalized [layers × heads] importance matrix.
    /// Only when `.activationPatching`.
    func patchingSweep(clean: CartographyInput, corrupted: CartographyInput) throws -> PatchingMatrix
}

// Sensible defaults so a minimal adapter only implements what it truly supports.
public extension ModelAdapter {
    func topology() -> [Edge] { [] }
    func features(layerIndex: Int) -> [Feature] { [] }
    func saliency(for input: CartographyInput) throws -> SaliencyMap {
        throw CartographyError.unsupported("saliency")
    }
    func ablated(regionIDs: Set<String>) throws -> ModelAdapter {
        throw CartographyError.unsupported("ablation")
    }
    func steered(featureID: String, gain: Double) throws -> ModelAdapter {
        throw CartographyError.unsupported("steering")
    }
    func logitLens(_ input: CartographyInput) throws -> [LayerReadout] {
        throw CartographyError.unsupported("logit lens")
    }
    func attributionGraph(_ input: CartographyInput) throws -> AttributionGraph {
        throw CartographyError.unsupported("attribution")
    }
    func attentionPatterns(_ input: CartographyInput) throws -> [AttentionHeadPattern] {
        throw CartographyError.unsupported("attention patterns")
    }
    func patchingSweep(clean: CartographyInput, corrupted: CartographyInput) throws -> PatchingMatrix {
        throw CartographyError.unsupported("activation patching")
    }

    /// Convenience: regions grouped into cortex columns by depth.
    func regionsByLayer() -> [Int: [Region]] {
        Dictionary(grouping: regions(), by: \.layerIndex)
    }
}
