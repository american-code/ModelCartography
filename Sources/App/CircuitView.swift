//
//  CircuitView.swift
//  IOI activation patching via Interp.ActivationPatching: supply a clean/corrupted
//  prompt pair, run the sweep, and read the [layers × heads] importance heatmap.
//  Each cell's brightness is the normalizedPatchingScore for that head — how much
//  injecting the corrupted head activation shifts the IOI logit diff. Brighter =
//  that head is more critical to the behavioral difference between the two prompts.
//

import SwiftUI

struct CircuitView: View {
    @Bindable var store: MapStore

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                PageHeader(store: store, title: "Attention Shift (TVD)",
                           expertSubtitle: "Activation patching sweep: which heads are critical to the circuit?",
                           plainSubtitle: "Find which attention heads change most between two sentences.")

                if store.isCircuitSupported {
                    promptPanel
                    if let matrix = store.patchingMatrix {
                        heatmapCard(matrix: matrix)
                        if let l = store.circuitSelectedLayer,
                           let h = store.circuitSelectedHead,
                           let pat = store.circuitHeadPattern {
                            headDetailCard(layer: l, head: h, pat: pat)
                        }
                    } else {
                        placeholderCard
                    }
                } else {
                    unsupportedCard
                }
            }
            .padding(20)
            .frame(maxWidth: 900, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
    }

    // MARK: Prompt entry

    private var promptPanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(store.t("Prompt pair", "Your two sentences")).font(.headline)
            Text(store.t(
                "Clean is the baseline; corrupted is the altered version. Each head is scored by how much its attention shifts between them.",
                "Type two versions of a sentence. The tool finds which parts of the model change most."
            )).font(.caption2).foregroundStyle(.secondary)

            HStack(spacing: 12) {
                LabeledTextField(label: store.t("Clean", "Original"),
                                 placeholder: "the quick fox jumped",
                                 text: $store.circuitCleanPrompt)
                LabeledTextField(label: store.t("Corrupted", "Changed version"),
                                 placeholder: "the lazy dog sat",
                                 text: $store.circuitCorruptedPrompt)
            }

            if let oov = store.circuitOOVWarning {
                Label(oov, systemImage: "exclamationmark.triangle")
                    .font(.caption2)
                    .foregroundStyle(Theme.trace)
            }

            Button { store.runPatchingSweep() } label: {
                Label(store.t("Run sweep", "Find important heads"),
                      systemImage: "waveform.badge.magnifyingglass")
            }
            .buttonStyle(.borderedProminent)
            .tint(Theme.signal)
        }
        .card()
    }

    // MARK: Heatmap

    private func heatmapCard(matrix: PatchingMatrix) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(store.t("IOI importance heatmap", "Head importance map")).font(.headline)
            Text(store.t(
                "Brighter cell = larger total-variation distance between clean and corrupted attention at that head. These heads drive the behavioral difference between your two prompts.",
                "Brighter squares are attention heads that changed the most — they matter most to how the model responds differently."
            )).font(.caption2).foregroundStyle(.secondary)

            heatmapGrid(matrix: matrix)
            headAxisLabels(count: matrix.heads)

            if let l = store.circuitSelectedLayer, let h = store.circuitSelectedHead {
                let score = matrix.scores[l][h]
                HStack(spacing: 6) {
                    Chip(text: "L\(l) · H\(h)", tint: Theme.signal)
                    Text(store.t(
                        String(format: "importance %.2f", score),
                        String(format: "impact %.0f%%", score * 100)
                    ))
                    .font(.caption2).foregroundStyle(.secondary)
                    Text(store.t("· tap another cell to compare",
                                 "· tap another square to compare"))
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }
        }
        .card()
    }

    private func heatmapGrid(matrix: PatchingMatrix) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(0..<matrix.layers, id: \.self) { l in
                HStack(spacing: 4) {
                    Text("L\(l)")
                        .font(.caption2.monospaced())
                        .frame(width: 24, alignment: .trailing)
                        .foregroundStyle(.secondary)
                    ForEach(0..<matrix.heads, id: \.self) { h in
                        let score = matrix.scores[l][h]
                        let isSel = store.circuitSelectedLayer == l && store.circuitSelectedHead == h
                        CircuitCell(score: score, isSelected: isSel) {
                            store.selectCircuitHead(layer: l, head: h)
                        }
                    }
                }
            }
        }
    }

    private func headAxisLabels(count: Int) -> some View {
        HStack(spacing: 4) {
            Spacer().frame(width: 28)
            ForEach(0..<count, id: \.self) { h in
                Text("H\(h)")
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
                    .frame(width: 44)
            }
        }
    }

    // MARK: Head detail

    private func headDetailCard(layer: Int, head: Int, pat: AttentionHeadPattern) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(store.t("L\(layer) · H\(head) — attention on clean input",
                         "Layer \(layer), Head \(head) — what it focuses on"))
                .font(.headline)
            Text(store.t(
                "Post-softmax weights on the clean prompt. Rows are query positions, columns are key positions; each row sums to 1.",
                "Shows where this head looks when reading the clean sentence. Each row adds up to 100%."
            )).font(.caption2).foregroundStyle(.secondary)

            HStack(alignment: .top, spacing: 18) {
                VStack(alignment: .leading, spacing: 6) {
                    AttentionHeatmap(pattern: pat)
                        .frame(width: CGFloat(pat.seqLen) * 22,
                               height: CGFloat(pat.seqLen) * 22)
                    Text(store.t("darker = stronger attention weight",
                                 "darker = looks at that position more"))
                        .font(.caption2).foregroundStyle(.secondary)
                }

                let score = store.patchingMatrix?.scores[layer][head] ?? 0
                VStack(alignment: .leading, spacing: 6) {
                    Text(store.t("Importance", "Impact")).font(.caption.bold())
                    Text(String(format: "%.2f", score))
                        .font(.title2.bold().monospacedDigit())
                        .foregroundStyle(Theme.heat(max(score, 0.1)))
                    Text(store.t("normalized TVD", "relative shift"))
                        .font(.caption2).foregroundStyle(.secondary)
                }
                .padding(12)
                .background(Theme.heat(score).opacity(0.18),
                            in: RoundedRectangle(cornerRadius: 10))

                Spacer()
            }
        }
        .card()
    }

    // MARK: Fallback cards

    private var placeholderCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(store.t("Ready to sweep", "Ready to go"),
                  systemImage: "arrow.triangle.2.circlepath")
                .font(.headline)
            Text(store.t(
                "Enter a prompt pair above and tap Run sweep to compute the [layers × heads] importance matrix.",
                "Fill in the two sentences above and tap the button to see which attention heads matter most."
            )).font(.caption).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
    }

    private var unsupportedCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label(store.t("Activation patching not available",
                          "This model doesn't support this yet"),
                  systemImage: "exclamationmark.triangle")
                .font(.headline).foregroundStyle(Theme.trace)
            Text(store.t(
                "Switch to the Toy Transformer adapter on the Cortex tab to enable the circuit sweep.",
                "Go to the Cortex tab and tap \"Toy Transformer (untrained weights)\" to try this feature."
            )).font(.caption).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
    }
}

// MARK: - Circuit cell

struct CircuitCell: View {
    let score: Double
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                RoundedRectangle(cornerRadius: Theme.cellCorner).fill(Theme.heat(score))
                Text(String(format: "%.0f", score * 100))
                    .font(.caption2.bold().monospacedDigit())
                    .foregroundStyle(score > 0.55 ? Color.white : .primary)
            }
            .frame(width: 44, height: 30)
            .overlay(
                RoundedRectangle(cornerRadius: Theme.cellCorner)
                    .stroke(isSelected ? Theme.signal : Color.clear, lineWidth: 2)
            )
            .tvOSFocusRing(cornerRadius: Theme.cellCorner)
        }
        #if os(tvOS)
        .buttonStyle(.card)
        .focusable()
        #else
        .buttonStyle(.plain)
        #endif
        .help(String(format: "Importance: %.3f", score))
    }
}

// MARK: - Labeled text field

struct LabeledTextField: View {
    let label: String
    let placeholder: String
    @Binding var text: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label).font(.caption.bold())
            TextField(placeholder, text: $text)
                #if !os(tvOS)
                .textFieldStyle(.roundedBorder)
                #endif
                .font(.callout)
        }
        .frame(maxWidth: .infinity)
    }
}
