//
//  IntervenePanel.swift
//  Stage 5 (the closed loop): remove marked regions, then re-verify on held-out data.
//  If the adapter can't ablate internals, this panel says so plainly instead of faking it.
//

import SwiftUI

struct IntervenePanel: View {
    @Bindable var store: MapStore

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                Text("Intervene").font(.largeTitle.bold())

                if !store.canAblate {
                    unsupportedCard
                } else {
                    baselineCard
                    selectionCard
                    if let d = store.diff { resultCard(d) }
                }
            }
            .padding(20)
            .frame(maxWidth: 760, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
    }

    private var unsupportedCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Internal ablation unavailable", systemImage: "lock.fill")
                .font(.headline).foregroundStyle(Theme.trace)
            Text("""
            \(store.adapterName) exposes only its outputs, so there are no internal regions to \
            remove. This is the framework degrading honestly — the same UI, minus the capability \
            the model can't support. Switch to the Demo Net (Cortex tab) to exercise the full \
            ablate → re-verify loop.
            """)
            .font(.callout).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
    }

    private var baselineCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Baseline").font(.headline)
            if let b = store.baseline {
                Text(String(format: "%.1f%% accuracy", b.overallAccuracy * 100))
                    .font(.title2.bold().monospacedDigit())
                Text("held-out set · \(b.count) inputs").font(.caption).foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
    }

    private var selectionCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Marked for removal").font(.headline)

            if store.canMarkDomains {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Quick-mark a whole domain").font(.caption).foregroundStyle(.secondary)
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 8) {
                            ForEach(store.adapter.classLabels, id: \.self) { label in
                                Button("All \(label)") { store.markDomain(label) }
                                    .buttonStyle(.bordered)
                                    .font(.caption)
                            }
                        }
                    }
                }
            }

            if store.selectedRegionIDs.isEmpty {
                Text("Tap hidden neurons in the Cortex tab to mark them, then remove them here.")
                    .font(.callout).foregroundStyle(.secondary)
            } else {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(store.selectedRegionIDs.sorted(), id: \.self) { id in
                            Chip(text: id, tint: Theme.trace)
                        }
                    }
                }
            }
            HStack(spacing: 12) {
                Button {
                    store.ablateAndVerify()
                } label: {
                    Label("Ablate & Re-verify", systemImage: "scissors")
                }
                .buttonStyle(.borderedProminent)
                .tint(Theme.trace)
                .disabled(store.selectedRegionIDs.isEmpty)

                Button("Clear") { store.clearIntervention() }
                    .buttonStyle(.bordered)
                    .disabled(store.selectedRegionIDs.isEmpty && store.diff == nil)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
    }

    private func resultCard(_ d: DiffReport) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Result").font(.headline)

            HStack(alignment: .firstTextBaseline, spacing: 16) {
                metric("Before", String(format: "%.1f%%", d.base.overallAccuracy * 100), Theme.muted)
                Image(systemName: "arrow.right").foregroundStyle(.secondary)
                metric("After", String(format: "%.1f%%", d.modified.overallAccuracy * 100),
                       d.accuracyDelta < -0.001 ? Theme.critical : Theme.signal)
                metric("Δ", String(format: "%+.1f pts", d.accuracyDelta * 100),
                       d.accuracyDelta < -0.001 ? Theme.critical : Theme.signal)
            }

            Divider()

            Text("Per-class change").font(.subheadline.bold())
            ForEach(store.adapter.classLabels, id: \.self) { cls in
                let delta = d.perClassDelta[cls] ?? 0
                ValueBar(label: cls,
                         value: min(1, abs(delta) / 0.5),
                         display: String(format: "%+.0f%%", delta * 100),
                         tint: delta < -0.001 ? Theme.critical : Theme.signal,
                         emphasized: d.collateral.contains(cls))
            }

            if d.collateral.isEmpty {
                Label("No collateral damage — good domains survived.", systemImage: "checkmark.seal.fill")
                    .font(.callout).foregroundStyle(Theme.signal)
            } else {
                Label("Collateral damage: \(d.collateral.joined(separator: ", ")). "
                      + "You removed something a good class depended on.",
                      systemImage: "exclamationmark.triangle.fill")
                    .font(.callout).foregroundStyle(Theme.critical)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
    }

    private func metric(_ title: String, _ value: String, _ tint: Color) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.title3.bold().monospacedDigit()).foregroundStyle(tint)
        }
    }
}
