# Model Cartography — Phase 2 Feature Spec

**Status:** Draft · 2026-08-02  
**Scope:** Three capabilities to implement after the Interp/Circuit work shipped in Phase 1.

---

## What Phase 1 delivered

| Area | What shipped |
|---|---|
| Abstraction | `ModelAdapter` + `Capabilities` option set; five adapters covering CoreML, MockNet, Dense+SAE, MoE, RoutingLog, Interp GPT-2 |
| Normalized types | `Region`, `Edge`, `Trace`, `Feature`, `SaliencyMap`, `LayerReadout`, `AttentionHeadPattern`, `PatchingMatrix`, `AttributionGraph` |
| UI | Cortex map · Trace · Intervene · Verify · Circuit (five tabs, three platforms) |
| Interpretability | Logit lens, attribution, steering, SAE features (Map B), IOI patching sweep, attention patterns |
| UX | Simplify toggle (plain-language mode throughout) |

The architecture is complete in the sense that adding a new adapter requires one file. Phase 2 is about deepening what the existing adapter surface can show — not adding a sixth adapter.

---

## Feature 1 — Cross-Layer Activation Comparison

**What it is:** A side-by-side view that shows how the same input's activations change with depth — or how two different inputs diverge across layers — plotted as a heatmap or sparkline matrix.

**Why it's Phase 2:** The Trace tab shows a single run's routing path. The Logit Lens tab shows how the _predicted token_ changes layer-by-layer. Neither shows how the _activation geometry_ evolves. This is the most natural next view for any user who has just understood what a logit lens is: "how does the residual stream grow more confident, and does it grow the same way for two similar inputs?"

**Protocol change:**

```swift
// New optional method on ModelAdapter
func layerActivations(_ input: CartographyInput) throws -> [[Double]]
// Returns [layerIndex][regionIndex] — one entry per layer returned by regions().
// Only adapters with .internalActivations need implement it; others throw .unsupported.
```

Add `.layerActivations` to `Capabilities`.

**Type needed:** None new — `[[Double]]` is sufficient. The UI layer owns the display.

**UI:** New "Layers" tab (between Trace and Intervene). Two controls:
- Input A / Input B picker (from the adapter's demo corpus).
- Mode: "Single input across layers" or "Two inputs, same layer."

Display: heatmap grid, rows = layers, columns = regions within each layer, color = activation magnitude (reuse `Theme.heat`). For two-input mode, show the _difference_ heatmap (A − B).

**Adapters that support it at launch:** MockNet, Dense+SAE, Interp GPT-2. MoE and RoutingLog can add it incrementally.

---

## Feature 2 — Attention-Head Role Annotation

**What it is:** A taxonomy of known mechanistic roles (induction head, previous-token head, name-mover, copy, etc.) assigned to heads after a multi-prompt sweep, surfaced as labels in the Circuit view's heatmap and in the Cortex map's head regions.

**Why it's Phase 2:** The Circuit tab already has a [layers × heads] patching matrix. Right now every cell is just a brightness score. Giving heads names is the next interpretability step — it turns the heatmap from a sensor readout into a vocabulary.

**Protocol change:**

```swift
public struct HeadRole: Sendable {
    public enum Kind: String, Sendable {
        case previousToken    // attends one position back
        case inductionHead    // completes repeated sequences
        case nameMover        // copies a token from earlier in context
        case copyHead         // copies the immediately preceding token to the next
        case broadcastHead    // high entropy; attends uniformly
        case unknown
    }
    public let kind: Kind
    public let confidence: Double   // 0…1
    public let notes: String?
}

// New optional method
func headRoles(patterns: [AttentionHeadPattern]) throws -> [[HeadRole]]
// Returns [layer][head].
```

Add `.headRoleAnnotation` to `Capabilities`.

**Classification heuristics (no model needed — computed from `AttentionHeadPattern`):**

| Role | Signal in attention weights |
|---|---|
| `previousToken` | Weights[row, row−1] > 0.6 for most rows |
| `inductionHead` | Weights[row, row−(seqLen/2)] elevated; correlate across duplicate-prefix test prompts |
| `broadcastHead` | Row entropy > 0.85 · log(seqLen) for most rows |
| `nameMover` | High weight on the token that appears in the output position at final-layer logit lens |
| `copyHead` | Weights[row, row] > 0.5 (diagonal-heavy) |

For the demo (2L 2H d=16) these heuristics will hit `unknown` often — that's fine. The annotation framework matters more than the labels for now.

**UI changes:** Circuit heatmap cells gain a small label badge (e.g. "prev", "ind", "?"). Tapping a cell in the detail card shows the `kind` + `confidence` + any `notes`. Cortex map head regions display the same badge alongside the existing selectivity bar.

**Adapters that support it at launch:** InterpAdapter — it already has `attentionPatterns`. Add `headRoles(patterns:)` inline (no SwiftSci dependency needed — pure Swift from the weights). MockNet and Dense+SAE heads are MLPs, not attention; `.headRoleAnnotation` stays out of their `capabilities`.

---

## Feature 3 — Feature Universality Visualization

**What it is:** A panel that compares SAE feature directions across two adapters (or two training runs of the same adapter) and visualizes which features are "universal" — shared geometry — and which are idiosyncratic. Ties the existing `SparseAutoencoder` work to the literature on feature universality in SAEs.

**Why it's Phase 2:** The Dense+SAE adapter already trains a sparse autoencoder and surfaces features as Map B regions. The natural question is: "are these features the same features every time, or do they differ by seed?" Answering that at toy scale is a stepping stone to comparing ModelCartography's features against published SAE dictionaries (e.g., Anthropic's Gemma or GPT-2 SAE releases).

**Types needed:**

```swift
public struct FeatureCorrespondence: Identifiable, Sendable {
    public let id: String
    /// Index into the first SAE's feature set.
    public let featureA: Int
    /// Index into the second SAE's feature set.
    public let featureB: Int
    /// Cosine similarity between decoder directions, −1…1.
    public let similarity: Double
    public let labelA: String?
    public let labelB: String?
}
```

**Computation:** For two `SparseAutoencoder` instances with the same `dim`, form the cosine similarity matrix (M × M). For each feature in SAE A, record its nearest neighbor in SAE B. Features with similarity > 0.85 are "universal"; below 0.3 are "idiosyncratic". This is a pure Swift operation with no ML framework dependency.

**Export hook:** A new `FeatureDictionary` codable struct wrapping `[Feature]` with their decoder directions, serializable to JSON so that an external SAE checkpoint (ResearchPapers format or Anthropic's open-source format) can be imported as a second bank for comparison.

```swift
public struct FeatureDictionary: Codable, Sendable {
    public let adapterName: String
    public let layerIndex: Int
    public let dim: Int
    public let features: [(label: String?, direction: [Double])]
    // Codable via CodingKeys; omit non-Codable fields as needed.
}
```

**UI:** New sub-tab within Intervene (or a card in the existing Intervene panel, gated on `.semanticFeatures`):
- Two-column list: "Universal features" / "Adapter-specific features".
- Tap a universal feature → shows both directions side-by-side as a bar chart (one bar per dimension) with similarity score.
- Import button (macOS/iPadOS, `#if os(macOS) || os(iOS)`) to load an external `FeatureDictionary` JSON file and run the comparison.

**Link to ResearchPapers work:** The import path accepts any JSON that deserializes to `FeatureDictionary`. Once external SAE checkpoints are available (from the ResearchPapers SAE training work), dropping the JSON here surfaces cross-model correspondences with no code change.

---

## What Phase 2 does NOT include

| Candidate | Rationale for deferral |
|---|---|
| Real GPT-2 / Llama weights | Disk constraint (local machine); use lab-02 once needed. The adapter interface already supports real weights — no code change needed when weight loading is added. |
| Superposition / feature geometry plots (PCA / UMAP) | Needs a 2-D layout engine not in the current SwiftUI stack. Worth a phase of its own. |
| Causal scrubbing | More complex graph rewriting than patching; design not settled. |
| Multi-token IOI | Current `PatchingMatrix` is token-agnostic; extending to track which token position the circuit is "about" is a deeper change to `CartographyTypes`. |

---

## Protocol surface summary

Three additions to `ModelAdapter` (all optional, guarded by capability flags):

```swift
// Feature 1
func layerActivations(_ input: CartographyInput) throws -> [[Double]]

// Feature 2
func headRoles(patterns: [AttentionHeadPattern]) throws -> [[HeadRole]]

// Feature 3 — adapter-side export; comparison logic lives in pipeline
func exportFeatureDictionary(layerIndex: Int) throws -> FeatureDictionary
```

Three new `Capabilities` bits:

```swift
static let layerActivations:    Capabilities = .init(rawValue: 1 << 9)
static let headRoleAnnotation:  Capabilities = .init(rawValue: 1 << 10)
static let featureExport:       Capabilities = .init(rawValue: 1 << 11)
```

No existing adapter's declared `capabilities` changes — new bits are additive.

---

## Implementation order

1. **Feature 1 (Cross-layer comparison)** — touches only `ModelAdapter`, `Capabilities`, and a new UI tab. Lowest risk, highest payoff for understanding. InterpAdapter and MockNet both support it with a one-method addition.

2. **Feature 2 (Head role annotation)** — extends the Circuit tab already built. The heuristic classifier is pure Swift. InterpAdapter is the only adapter in scope.

3. **Feature 3 (Feature universality)** — deepest but also the most self-contained: new types, new pipeline utility, new Intervene card, no changes to existing tabs or adapters.

Each feature is independently shippable.
