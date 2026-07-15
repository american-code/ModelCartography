//
//  TraceView.swift
//  Stages 1–2 & 4 for a single input: pick something, watch it flow input -> output.
//  Shows the prediction, per-class probabilities, the realized pathway, and (when the
//  adapter supports it) a saliency map over the input.
//

import SwiftUI

struct TraceView: View {
    @Bindable var store: MapStore
    @State private var showSaliency = true

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                Text("Trace").font(.largeTitle.bold())
                Text("Pick an input; follow it from pixels to prediction.")
                    .foregroundStyle(.secondary)

                inputPicker

                if let input = store.selectedInput, case .grid(let g) = input.payload {
                    HStack(alignment: .top, spacing: 18) {
                        selectedInputCard(input: input, grid: g)
                        predictionCard(input: input)
                    }
                    pathwayCard
                }
            }
            .padding(20)
            .frame(maxWidth: 900, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
    }

    private var inputPicker: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 12) {
                ForEach(store.corpus) { input in
                    if case .grid(let g) = input.payload {
                        Button {
                            store.select(input)
                        } label: {
                            GridThumbnail(grid: g)
                                .frame(width: 54, height: 54)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 8)
                                        .stroke(store.selectedInput?.id == input.id ? Theme.signal : .clear,
                                                lineWidth: 3)
                                )
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
            .padding(.vertical, 4)
        }
    }

    private func selectedInputCard(input: CartographyInput, grid: [[Double]]) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Input").font(.headline)
            GridThumbnail(grid: grid, saliency: showSaliency ? store.currentSaliency : nil)
                .frame(width: 180, height: 180)
            if store.canSaliency {
                Toggle("Saliency overlay", isOn: $showSaliency)
                    .font(.caption)
            } else {
                Text("Saliency not available for this model").font(.caption2).foregroundStyle(.secondary)
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

    private var pathwayCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Routing path").font(.headline)
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
                Text("the strongest region at each layer — the literal expert sequence in an MoE model")
                    .font(.caption2).foregroundStyle(.secondary)
            } else {
                Text("No path (output-only model)").font(.caption).foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
    }
}
