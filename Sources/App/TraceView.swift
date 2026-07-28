//
//  TraceView.swift
//  Stages 1–2 & 4 for a single input: pick something, watch it flow input -> output.
//  Prediction + per-class probabilities + realized pathway, plus (when supported)
//  saliency (image models) and the logit lens + attribution graph (dense models).
//

import SwiftUI

/// A square grid that visualizes one attention head's post-softmax weights.
/// Darker cell = higher probability mass; rows are query positions, columns are key positions.
struct AttentionHeatmap: View {
    let pattern: AttentionHeadPattern

    var body: some View {
        VStack(spacing: 1) {
            ForEach(0..<pattern.seqLen, id: \.self) { row in
                HStack(spacing: 1) {
                    ForEach(0..<pattern.seqLen, id: \.self) { col in
                        let w = pattern.weights[row * pattern.seqLen + col]
                        Color(white: Double(1.0 - w * 0.85))
                            .frame(width: 18, height: 18)
                    }
                }
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 4))
        .overlay(RoundedRectangle(cornerRadius: 4).stroke(Color.primary.opacity(0.12), lineWidth: 0.5))
    }
}

struct TraceView: View {
    @Bindable var store: MapStore
    @State private var showSaliency = true
    @State private var lensAnimPhase = 0

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                PageHeader(store: store, title: "Trace",
                           expertSubtitle: "Pick an input; follow it from input to prediction.",
                           plainSubtitle: "Pick something and watch how the model turns it into an answer.")

                inputPicker

                if let input = store.selectedInput {
                    HStack(alignment: .top, spacing: 18) {
                        selectedInputCard(input: input)
                        predictionCard(input: input)
                    }
                    if store.hasLens { lensCard }
                    if store.hasAttribution, let g = store.currentAttribution { attributionCard(g) }
                    if store.hasAttentionPatterns, !store.currentAttentionPatterns.isEmpty {
                        attentionCard
                    }
                    pathwayCard
                }
            }
            .padding(20)
            .frame(maxWidth: 900, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
    }

    // MARK: Input picker (grid thumbnails or text chips)

    private var inputPicker: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 12) {
                ForEach(store.corpus) { input in
                    Button { store.select(input) } label: { pickerLabel(input) }
                        .buttonStyle(.plain)
                }
            }
            .padding(.vertical, 4)
        }
    }

    @ViewBuilder
    private func pickerLabel(_ input: CartographyInput) -> some View {
        let selected = store.selectedInput?.id == input.id
        if case .grid(let g) = input.payload {
            GridThumbnail(grid: g)
                .frame(width: 54, height: 54)
                .overlay(RoundedRectangle(cornerRadius: 8)
                    .stroke(selected ? Theme.signal : .clear, lineWidth: 3))
        } else {
            Text(input.display ?? input.id)
                .font(.caption).lineLimit(1)
                .padding(.horizontal, 12).padding(.vertical, 10)
                .background(selected ? Theme.signal.opacity(0.18) : Color.primary.opacity(0.06),
                            in: RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8)
                    .stroke(selected ? Theme.signal : .clear, lineWidth: 2))
        }
    }

    // MARK: Selected input

    @ViewBuilder
    private func selectedInputCard(input: CartographyInput) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Input").font(.headline)
            if case .grid(let g) = input.payload {
                GridThumbnail(grid: g, saliency: showSaliency ? store.currentSaliency : nil)
                    .frame(width: 180, height: 180)
                if store.canSaliency {
                    Toggle(store.t("Saliency overlay", "Highlight what mattered"), isOn: $showSaliency)
                        .font(.caption)
                }
            } else if let text = input.display {
                Text("“\(text)”").font(.title3).italic()
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let truth = input.truth {
                Text("labeled: \(truth)").font(.caption).foregroundStyle(.secondary)
            }
        }
        .card()
    }

    private func predictionCard(input: CartographyInput) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Prediction").font(.headline)
            if let trace = store.currentTrace {
                HStack(spacing: 8) {
                    Text(trace.predicted).font(.title2.bold()).foregroundStyle(Theme.signal)
                    if let truth = input.truth {
                        Image(systemName: truth == trace.predicted ? "checkmark.circle.fill" : "xmark.circle.fill")
                            .foregroundStyle(truth == trace.predicted ? Theme.signal : Theme.critical)
                    }
                }
                VStack(spacing: 6) {
                    ForEach(store.adapter.classLabels, id: \.self) { label in
                        let p = trace.probabilities[label] ?? 0
                        ValueBar(label: label, value: p, display: String(format: "%.0f%%", p * 100),
                                 tint: label == trace.predicted ? Theme.signal : Theme.muted,
                                 emphasized: label == trace.predicted)
                    }
                }
            } else {
                Text("Select an input").foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
    }

    // MARK: Logit lens

    private var lensCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(store.t("Logit lens", "How the guess takes shape")).font(.headline)
            Text(store.t(
                "top-5 predicted tokens at each residual stream layer — confidence sharpens with depth",
                "the model's top 5 guesses at each step — it grows more confident deeper in the network"
            ))
            .font(.caption2).foregroundStyle(.secondary)

            ScrollView(.horizontal, showsIndicators: false) {
                VStack(alignment: .leading, spacing: 0) {
                    LensTableHeader()
                    Divider()
                    ForEach(Array(store.currentLens.enumerated()), id: \.element.layerIndex) { idx, readout in
                        LensTableRow(readout: readout, rowIndex: idx, animPhase: lensAnimPhase)
                        if idx < store.currentLens.count - 1 {
                            Divider().opacity(0.35)
                        }
                    }
                }
                .id(lensAnimPhase)
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
            .animation(.easeOut(duration: 0.4), value: lensAnimPhase)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
        .onChange(of: store.selectedInput?.id) { _, _ in
            withAnimation(.easeOut(duration: 0.35)) { lensAnimPhase += 1 }
        }
    }

    // MARK: Attribution graph

    private func attributionCard(_ graph: AttributionGraph) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(store.t("Attribution", "What convinced it")).font(.headline)
            Text(store.t("which features pushed the output toward “\(graph.predicted)”",
                         "which parts pushed the answer toward “\(graph.predicted)”"))
                .font(.caption2).foregroundStyle(.secondary)
            let maxMag = graph.contributions.map { abs($0.value) }.max() ?? 1
            ForEach(graph.contributions) { c in
                ValueBar(label: c.id,
                         value: abs(c.value) / maxMag,
                         display: String(format: "%+.2f", c.value),
                         tint: c.value >= 0 ? Theme.signal : Theme.critical)
                Text(c.label).font(.caption2).foregroundStyle(.secondary)
                    .padding(.leading, 100).padding(.top, -4)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
    }

    // MARK: Attention patterns (Interp)

    private var attentionCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(store.t("Attention patterns", "What each layer focuses on")).font(.headline)
            Text(store.t("post-softmax weights — row i attends to column j; rows sum to 1",
                         "how much each step looks back at earlier positions; each row adds up to 100%"))
                .font(.caption2).foregroundStyle(.secondary)

            let cols = [GridItem(.flexible()), GridItem(.flexible())]
            LazyVGrid(columns: cols, alignment: .leading, spacing: 14) {
                ForEach(store.currentAttentionPatterns) { pat in
                    VStack(alignment: .leading, spacing: 4) {
                        Text("L\(pat.layerIndex) · H\(pat.headIndex)")
                            .font(.caption2.monospaced()).foregroundStyle(.secondary)
                        AttentionHeatmap(pattern: pat)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
    }

    private var pathwayCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(store.t("Routing path", "The path it took")).font(.headline)
            if let trace = store.currentTrace, !trace.routingPath.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(Array(trace.routingPath.enumerated()), id: \.offset) { idx, id in
                            Chip(text: id, tint: Theme.signal)
                            if idx < trace.routingPath.count - 1 {
                                Image(systemName: "arrow.right").font(.caption2).foregroundStyle(.secondary)
                            }
                        }
                    }
                }
                Text(store.t("the strongest region at each layer — the literal expert sequence in an MoE model",
                             "the busiest part at each step, from input to answer"))
                    .font(.caption2).foregroundStyle(.secondary)
            } else {
                Text("No path (output-only model)").font(.caption).foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
    }
}

// MARK: - Logit lens table helpers

private struct LensTableHeader: View {
    var body: some View {
        HStack(spacing: 4) {
            Text("Layer")
                .frame(width: 82, alignment: .leading)
            ForEach(1...5, id: \.self) { rank in
                Text("#\(rank)")
                    .frame(width: 84, alignment: .center)
            }
        }
        .font(.caption2.bold())
        .foregroundStyle(.secondary)
        .padding(.vertical, 6)
        .padding(.horizontal, 6)
    }
}

private struct LensTableRow: View {
    let readout: LayerReadout
    let rowIndex: Int
    let animPhase: Int

    private var top5: [(token: String, prob: Double)] {
        readout.probabilities
            .sorted { $0.value > $1.value }
            .prefix(5)
            .map { (token: $0.key, prob: $0.value) }
    }

    var body: some View {
        HStack(spacing: 4) {
            Text(readout.name)
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
                .frame(width: 82, alignment: .leading)
            ForEach(Array(top5.enumerated()), id: \.offset) { _, entry in
                LensTokenCell(token: entry.token, prob: entry.prob,
                              isTop: entry.token == readout.top)
            }
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 6)
        .animation(
            .easeOut(duration: 0.3).delay(Double(rowIndex) * 0.06),
            value: animPhase
        )
    }
}

private struct LensTokenCell: View {
    let token: String
    let prob: Double
    let isTop: Bool

    var body: some View {
        VStack(spacing: 2) {
            Text(token)
                .font(.caption.monospaced())
                .fontWeight(isTop ? .semibold : .regular)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
            Text(String(format: "%.0f%%", prob * 100))
                .font(.system(size: 9).monospaced())
                .foregroundStyle(isTop ? Theme.signal.opacity(0.85) : .secondary)
        }
        .padding(.vertical, 5)
        .padding(.horizontal, 6)
        .frame(width: 84)
        .background(
            isTop
                ? Theme.heat(prob)
                : Color.primary.opacity(0.06 + 0.18 * prob),
            in: RoundedRectangle(cornerRadius: Theme.cellCorner)
        )
        .foregroundStyle(isTop ? Theme.signal : .primary)
    }
}
