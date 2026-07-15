# Model Cartography

Turn an AI model into a place you can navigate — map its regions, trace a single
inference from input to output, and surgically target what to keep, cut, or replace.

One SwiftUI codebase, three platforms: **macOS · iPadOS · tvOS**.

This repository is **Phase 1** of the design in the accompanying design note: the full
five-stage pipeline (Instrument → Capture → Attribute → Map → Intervene) running on the
smallest real model, built around an abstraction that later scales up to dense LLMs and
Mixture-of-Experts models without a rewrite.

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
| Reference adapter | `Sources/Adapters/MockNetwork/` | a real MLP trained in pure Swift — **fully** introspectable |
| Honest-degradation adapter | `Sources/Adapters/CoreML/CoreMLAdapter.swift` | wraps a Core ML classifier; occlusion saliency; **no** internal ablation |
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

## Not yet here — Phase 2

- A dense-LLM adapter (logit lens, attribution graphs).
- **Map B**: sparse-autoencoder features + auto-labeling — the human-legible "domains".
- Steering vectors.

The `Feature` type and `.semanticFeatures` capability are already defined so Phase 2 slots in
behind the same interface.
