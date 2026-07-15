//
//  VerifyView.swift
//  The unified, model-agnostic health view. Every classifier adapter — demo net, dense,
//  MoE, Core ML — reports the same two things over the held-out set: overall + per-class
//  accuracy, and a confusion matrix showing exactly which classes it mixes up.
//

import SwiftUI

struct VerifyView: View {
    @Bindable var store: MapStore

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                PageHeader(store: store, title: "Verify",
                           expertSubtitle: "The same held-out check for any model — where it's right, and where it fails.",
                           plainSubtitle: "A quick report card: how often the model is right, and what it gets wrong.")

                if let c = store.confusion, c.total > 0 {
                    healthCard(c)
                    confusionCard(c)
                } else {
                    Text(store.t("This model doesn't expose a verifiable held-out score.",
                                 "This model can't be scored here."))
                        .foregroundStyle(.secondary).card()
                }
            }
            .padding(20)
            .frame(maxWidth: 820, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
    }

    private func healthCard(_ c: ConfusionMatrix) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                Text(store.t("Model health", "How good is it?")).font(.headline)
                Spacer()
                Text(String(format: "%.1f%%", c.accuracy * 100))
                    .font(.title2.bold().monospacedDigit()).foregroundStyle(Theme.signal)
            }
            Text("\(store.adapterName) · \(c.total) held-out inputs")
                .font(.caption).foregroundStyle(.secondary)
            ForEach(c.labels.indices, id: \.self) { i in
                let rt = c.rowTotal(i)
                let acc = rt > 0 ? Double(c.counts[i][i]) / Double(rt) : 0
                ValueBar(label: c.labels[i], value: acc,
                         display: String(format: "%.0f%%", acc * 100),
                         tint: acc >= 0.8 ? Theme.signal : (acc >= 0.5 ? Theme.trace : Theme.critical),
                         emphasized: acc < 0.5)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
    }

    private func confusionCard(_ c: ConfusionMatrix) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(store.t("Confusion matrix", "What it mixes up")).font(.headline)
            Text(store.t("rows = actual · columns = predicted · diagonal = correct",
                         "each row is the real answer; green = right, red = mistakes"))
                .font(.caption2).foregroundStyle(.secondary)

            ScrollView(.horizontal, showsIndicators: false) {
                Grid(horizontalSpacing: 4, verticalSpacing: 4) {
                    GridRow {
                        Color.clear.frame(width: labelW, height: 22)
                        ForEach(c.labels, id: \.self) { p in
                            Text(short(p)).font(.caption2.monospaced())
                                .foregroundStyle(.secondary)
                                .frame(width: cellW, height: 22)
                        }
                    }
                    ForEach(c.labels.indices, id: \.self) { i in
                        GridRow {
                            Text(short(c.labels[i])).font(.caption2.monospaced())
                                .foregroundStyle(.secondary)
                                .frame(width: labelW, alignment: .trailing)
                            ForEach(c.labels.indices, id: \.self) { j in
                                cell(c, i, j)
                            }
                        }
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
    }

    private func cell(_ c: ConfusionMatrix, _ i: Int, _ j: Int) -> some View {
        let count = c.counts[i][j]
        let rt = c.rowTotal(i)
        let frac = rt > 0 ? Double(count) / Double(rt) : 0
        let isDiagonal = i == j
        let base = isDiagonal ? Theme.signal : Theme.critical
        return Text(count == 0 ? "·" : "\(count)")
            .font(.caption.monospacedDigit())
            .foregroundStyle(frac > 0.55 ? Color.white : .primary)
            .frame(width: cellW, height: 34)
            .background(base.opacity(count == 0 ? 0.04 : 0.15 + 0.75 * frac),
                        in: RoundedRectangle(cornerRadius: 6))
    }

    private let cellW: CGFloat = 46
    private let labelW: CGFloat = 74
    private func short(_ s: String) -> String { String(s.prefix(6)) }
}
