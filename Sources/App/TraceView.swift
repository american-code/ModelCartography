//
//  TraceView.swift
//  Stages 1–2 & 4 for a single input: pick something, watch it flow input -> output.
//  Prediction + per-class probabilities + realized pathway, plus (when supported)
//  saliency (image models) and the logit lens + attribution graph (dense models).
//

import SwiftUI

struct TraceView: View {
    @Bindable var store: MapStore
    @State private var showSaliency = true

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
            Text(store.t("the model's running guess decoded at each layer — watch it sharpen with depth",
                         "the model's best guess at each step — it gets more sure as it goes"))
                .font(.caption2).foregroundStyle(.secondary)
            ForEach(store.currentLens, id: \.layerIndex) { readout in
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(readout.name).font(.caption.monospaced())
                        Spacer()
                        Text("→ \(readout.top)").font(.caption.bold()).foregroundStyle(Theme.signal)
                    }
                    HStack(spacing: 4) {
                        ForEach(store.adapter.classLabels, id: \.self) { label in
                            let p = readout.probabilities[label] ?? 0
                            Capsule()
                                .fill(label == readout.top ? Theme.signal : Color.primary.opacity(0.12))
                                .frame(height: 6)
                                .frame(maxWidth: .infinity)
                                .opacity(0.35 + 0.65 * p)
                        }
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
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
