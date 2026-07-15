# Model Cartography

Turn an AI model into a place you can navigate — map its regions, trace a single
inference from input to output, and surgically target what to keep, cut, or replace.

One SwiftUI codebase, three platforms: **macOS · iPadOS · tvOS**.

This repository covers **Phase 1 and Phase 2** of the design in the accompanying design
note: the full five-stage pipeline (Instrument → Capture → Attribute → Map → Intervene),
plus Map B (interpretable features), logit lens, attribution, and steering — all behind one
adapter interface that scales from a Core ML classifier to a dense residual model without a
rewrite.

---

## The idea in one paragraph

The app never touches a model directly. It only ever sees a normalized **Cartography Map**
(regions, edges, activations, traces) produced by a thin per-architecture **`ModelAdapter`**.
Each adapter *declares* what it can honestly do via `capabilities`, so the tool degrades
gracefully instead of faking a capability the model can't support. Write the UI, the tracer,
and the optimizer once; add a new architecture by writing one adapter.

## What's here (Phase 1)

| Piece | File | Role |
|---|---|---|
| Normalized vocabulary | `Sources/Core/CartographyTypes.swift` | Region, Edge, Trace, Feature, SaliencyMap |
| The one interface | `Sources/Core/ModelAdapter.swift` | protocol + `Capabilities` option set |
| Reference adapter (Phase 1) | `Sources/Adapters/MockNetwork/` | a real MLP trained in pure Swift — **fully** introspectable |
| Honest-degradation adapter | `Sources/Adapters/CoreML/CoreMLAdapter.swift` | wraps a Core ML classifier; occlusion saliency; **no** internal ablation |
| Dense adapter (Phase 2) | `Sources/Adapters/DenseText/` | residual model + sparse autoencoder → features, logit lens, attribution, steering |
| Attribution | `Sources/Pipeline/Attribution.swift` | labels each region with the class it prefers |
| Verification | `Sources/Pipeline/Verification.swift` | ablate → re-verify on held-out data + collateral detection |
| UI | `Sources/App/` | Cortex map · Trace view · Intervene panel |

### Two adapters, on purpose

- **Demo Net** (`MockNetworkAdapter`) — an 8×8 → 16 → 16 → 3 classifier trained on synthetic
  line patterns. Because we own the graph, it supports the *full* capability set: internal
  activations, true gradient saliency, and real per-neuron ablation. It exists to prove the
  entire loop end-to-end.
- **Core ML** (`CoreMLAdapter`) — wraps any Core ML image classifier via Vision. Core ML is a
  black box for internal activations, so this adapter advertises only `.saliency`
  (forward-only occlusion) and `.verification`. The Intervene panel then *hides* internal
  ablation rather than pretending — the honest-degradation design in action.

## Build & run

Requires Xcode 16+ (Swift 6) and [XcodeGen](https://github.com/yonaskolb/XcodeGen).

```sh
xcodegen generate
open ModelCartography.xcodeproj
# pick the My Mac / iPad / Apple TV destination and Run
```

The project file is generated from `project.yml` and git-ignored; regenerate any time.

### Verifying the pipeline without a full app build

The core is pure Swift with no UI dependency, so the whole loop can be compiled to a CLI and
exercised directly (useful in CI or a headless VM):

```sh
swiftc -O Sources/Core/*.swift Sources/Adapters/MockNetwork/*.swift \
  Sources/Pipeline/*.swift your_main.swift -o check && ./check
```

A representative run: the net trains to ~95% held-out accuracy, neurons develop clean
single-class selectivity, and group-ablating every "horizontal" neuron drops that class ~25%
while the harness flags `horizontal` as collateral damage.

## Using it

1. **Cortex** — columns are layers, cells are regions, brightness is activation for the
   current input. A teal ring marks the routing path; tap hidden neurons to mark them for
   ablation.
2. **Trace** — pick an input, watch it flow to a prediction, see per-class probabilities, the
   routing path, and a saliency overlay.
3. **Intervene** — quick-mark a whole domain (or hand-pick neurons), then **Ablate & Re-verify**.
   The result diffs before/after accuracy per class and flags any good class you damaged.

## Phase 2 — the dense model (`Sources/Adapters/DenseText/`)

A small **dense residual classifier** over a synthetic topic corpus (weather / finance / food),
trained on-device in pure Swift. Because a real 7B LLM can't run in a VM — or on a TV — this
model is the stand-in that makes each Phase 2 *technique* real and inspectable:

- **Map B — features.** A **sparse autoencoder** trains on the model's residual stream and
  recovers an overcomplete dictionary of directions. Each feature is **auto-labeled** by the
  inputs that most activate it (e.g. `finance: invest, stock`). For a dense model these
  features *are* the cortex — the meaningful domains live in activation space, not in raw
  neurons.
- **Logit lens.** The classifier head is applied to every residual point, so you watch the
  running prediction sharpen with depth (e.g. `P(finance)` 30% → 84% → 100%).
- **Attribution graph.** The predicted logit is decomposed over features via
  `feature_activation × (head · feature_direction)` — a first-order account of how this input
  became this output.
- **Steering.** Adding `gain × feature_direction` to the residual at inference pushes behavior
  without retraining — enough to flip a *food* sentence to *finance*.

All four appear in the UI: features in the **Cortex** and **Trace** tabs, the logit lens and
attribution in **Trace**, and steering in **Intervene** (which swaps its ablation controls for
steering controls based on the adapter's declared `capabilities`).

### Not yet here — Phase 3

The MoE adapter (wrap Colibrì's routing logs as a native, labeled expert cortex) and a unified
verification harness across all adapters.
