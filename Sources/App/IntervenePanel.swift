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

                if store.canSteer {
                    steeringIntro
                    steeringPicker
                    steeringControls
                    if let base = store.steerBase, let after = store.steerResult {
                        steeringResult(base: base, after: after)
                    }
                } else if store.canAblate {
                    baselineCard
                    if store.hasExperts { utilizationCard }
                    selectionCard
                    if let d = store.diff { resultCard(d) }
                } else {
                    unsupportedCard
                }
            }
            .padding(20)
            .frame(maxWidth: 760, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
    }

    // MARK: Steering (Phase 2)

    private var steeringIntro: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Steering").font(.headline)
            Text("""
            Add a feature's direction to the residual stream at inference to push behavior — \
            replacing retraining. Pick a feature, set the gain, and apply it to the input \
            currently selected on the Trace tab.
            """)
            .font(.callout).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
    }

    private var steeringPicker: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Feature").font(.subheadline.bold())
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(store.featureRegions) { region in
                        Button { store.selectFeature(region.id) } label: {
                            Text(region.label ?? region.id)
                                .font(.caption)
                                .padding(.horizontal, 10).padding(.vertical, 7)
                                .background(store.steerFeatureID == region.id
                                            ? Theme.trace.opacity(0.22) : Color.primary.opacity(0.06),
                                            in: Capsule())
                                .overlay(Capsule().stroke(
                                    store.steerFeatureID == region.id ? Theme.trace : .clear, lineWidth: 2))
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
    }

    private var steeringControls: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Gain").font(.subheadline.bold())
                Spacer()
                Text(String(format: "×%.1f", store.steerGain))
                    .font(.callout.monospacedDigit()).foregroundStyle(Theme.trace)
            }
            HStack(spacing: 12) {
                Button { store.steerGain = max(-8, store.steerGain - 1) } label: {
                    Image(systemName: "minus")
                }
                #if os(tvOS)
                // Slider is unavailable on tvOS — show the value as a track the steppers drive.
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        Capsule().fill(Color.primary.opacity(0.10))
                        Capsule().fill(Theme.trace)
                            .frame(width: geo.size.width * CGFloat((store.steerGain + 8) / 16))
                    }
                }
                .frame(height: 10)
                #else
                Slider(value: $store.steerGain, in: -8...8, step: 0.5)
                    .tint(Theme.trace)
                #endif
                Button { store.steerGain = min(8, store.steerGain + 1) } label: {
                    Image(systemName: "plus")
                }
            }
            .buttonStyle(.bordered)

            HStack(spacing: 12) {
                Button {
                    store.applySteering()
                } label: {
                    Label("Apply steering", systemImage: "dial.high")
                }
                .buttonStyle(.borderedProminent).tint(Theme.trace)
                .disabled(store.steerFeatureID == nil || store.selectedInput == nil)
            }
            if store.selectedInput == nil {
                Text("Select an input on the Trace tab first.").font(.caption).foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
    }

    private func steeringResult(base: [String: Double], after: [String: Double]) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Output shift").font(.headline)
            ForEach(store.adapter.classLabels, id: \.self) { cls in
                let b = base[cls] ?? 0, a = after[cls] ?? 0
                VStack(spacing: 3) {
                    ValueBar(label: cls, value: b, display: String(format: "%.0f%%", b * 100),
                             tint: Theme.muted)
                    ValueBar(label: "", value: a, display: String(format: "%.0f%%", a * 100),
                             tint: a >= b ? Theme.signal : Theme.critical, emphasized: true)
                }
            }
            Text("top row = before · bottom row = after steering")
                .font(.caption2).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
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

    // MARK: Expert utilization (Phase 3 — the prune/pin signal)

    private var utilizationCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Expert utilization").font(.headline)
            Text("how much the router uses each expert across the corpus — cold experts are safe to prune")
                .font(.caption2).foregroundStyle(.secondary)
            let maxU = store.expertsByUtilization.map { $0.util }.max() ?? 1
            ForEach(store.expertsByUtilization, id: \.region.id) { item in
                let cold = item.util < 0.05
                ValueBar(label: "\(item.region.id) · \(item.region.label ?? "?")",
                         value: maxU > 0 ? item.util / maxU : 0,
                         display: String(format: "%.0f%%", item.util * 100),
                         tint: cold ? Theme.critical : Theme.signal,
                         emphasized: cold)
            }
            Button { store.markColdExperts() } label: {
                Label("Mark cold experts", systemImage: "thermometer.snowflake")
            }
            .buttonStyle(.bordered)
            .padding(.top, 2)
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
