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
                PageHeader(store: store, title: store.t("Intervene", "Experiment"),
                           expertSubtitle: "Remove or steer parts of the model, then re-verify what changed.",
                           plainSubtitle: "Change part of the model, then check whether it got better or worse.")

                if store.canSteer {
                    steeringIntro
                    steeringPicker
                    steeringControls
                    if let base = store.steerBase, let after = store.steerResult {
                        steeringResult(base: base, after: after)
                    }
                } else {
                    if store.canAblate { baselineCard }
                    if store.hasExperts { utilizationCard }
                    if store.canAblate {
                        selectionCard
                        if let d = store.diff { resultCard(d) }
                    } else {
                        unsupportedCard
                    }
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
            Text(store.t("Steering", "Nudge the model")).font(.headline)
            Text(store.t("""
            Add a feature's direction to the residual stream at inference to push behavior — \
            replacing retraining. Pick a feature, set the gain, and apply it to the input \
            currently selected on the Trace tab.
            """, """
            Turn one of the model's learned concepts up or down to push its answer — no \
            retraining needed. Pick a concept, set how hard to push, and apply it to whatever \
            input you picked on the Trace tab.
            """))
            .font(.callout).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
    }

    private var steeringPicker: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(store.t("Feature", "Concept")).font(.subheadline.bold())
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
                Text(store.t("Gain", "How hard to push")).font(.subheadline.bold())
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
                    Label(store.t("Apply steering", "Apply nudge"), systemImage: "dial.high")
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
            Text(store.t("Output shift", "How the answer changed")).font(.headline)
            ForEach(store.adapter.classLabels, id: \.self) { cls in
                let b = base[cls] ?? 0, a = after[cls] ?? 0
                VStack(spacing: 3) {
                    ValueBar(label: cls, value: b, display: String(format: "%.0f%%", b * 100),
                             tint: Theme.muted)
                    ValueBar(label: "", value: a, display: String(format: "%.0f%%", a * 100),
                             tint: a >= b ? Theme.signal : Theme.critical, emphasized: true)
                }
            }
            Text(store.t("top row = before · bottom row = after steering",
                         "top bar = before · bottom bar = after the nudge"))
                .font(.caption2).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
    }

    private var unsupportedCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label(store.t("Interventions unavailable", "Can't change this model"), systemImage: "lock.fill")
                .font(.headline).foregroundStyle(Theme.trace)
            Text(unsupportedReason)
                .font(.callout).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
    }

    private var unsupportedReason: String {
        if store.hasExperts {
            return """
            \(store.adapterName) is a static routing snapshot — you can map and label its experts \
            and verify accuracy, but removing or steering them needs the live engine or its \
            weights. This is the framework degrading honestly: same UI, minus the capabilities a \
            log can't support.
            """
        }
        return """
        \(store.adapterName) exposes only its outputs, so there are no internal regions to remove. \
        This is the framework degrading honestly — the same UI, minus the capability the model \
        can't support. Switch to the Demo Net or MoE Model (Cortex tab) to exercise the full \
        ablate → re-verify loop.
        """
    }

    // MARK: Expert utilization (Phase 3 — the prune/pin signal)

    private var utilizationCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(store.t("Expert utilization", "How much each expert is used")).font(.headline)
            Text(store.t("how much the router uses each expert across the corpus — cold experts are safe to prune",
                         "how often the model uses each expert — barely-used ones are safe to remove"))
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
            if store.canAblate {
                Button { store.markColdExperts() } label: {
                    Label(store.t("Mark cold experts", "Pick the barely-used ones"), systemImage: "thermometer.snowflake")
                }
                .buttonStyle(.bordered)
                .padding(.top, 2)
            } else {
                Text(store.t("Read-only: this is an imported snapshot, so cold experts can be identified but not pruned here.",
                             "View only: this is an imported snapshot, so unused experts can be spotted but not removed here."))
                    .font(.caption).foregroundStyle(.secondary).padding(.top, 2)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
    }

    private var baselineCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(store.t("Baseline", "Starting point")).font(.headline)
            if let b = store.baseline {
                Text(String(format: "%.1f%% accuracy", b.overallAccuracy * 100))
                    .font(.title2.bold().monospacedDigit())
                Text(store.t("held-out set · \(b.count) inputs",
                             "tested on \(b.count) fresh inputs")).font(.caption).foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
    }

    private var selectionCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(store.t("Marked for removal", "Chosen to remove")).font(.headline)

            if store.canMarkDomains {
                VStack(alignment: .leading, spacing: 6) {
                    Text(store.t("Quick-mark a whole domain", "Pick everything for one topic"))
                        .font(.caption).foregroundStyle(.secondary)
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
                Text(store.t("Tap hidden neurons in the Cortex tab to mark them, then remove them here.",
                             "Tap squares in the Cortex tab to pick them, then remove them here."))
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
                    Label(store.t("Ablate & Re-verify", "Remove & recheck"), systemImage: "scissors")
                }
                .buttonStyle(.borderedProminent)
                .tint(Theme.trace)
                .disabled(store.selectedRegionIDs.isEmpty)

                Button(store.t("Clear", "Reset")) { store.clearIntervention() }
                    .buttonStyle(.bordered)
                    .disabled(store.selectedRegionIDs.isEmpty && store.diff == nil)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
    }

    private func resultCard(_ d: DiffReport) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(store.t("Result", "What happened")).font(.headline)

            HStack(alignment: .firstTextBaseline, spacing: 16) {
                metric(store.t("Before", "Before"), String(format: "%.1f%%", d.base.overallAccuracy * 100), Theme.muted)
                Image(systemName: "arrow.right").foregroundStyle(.secondary)
                metric(store.t("After", "After"), String(format: "%.1f%%", d.modified.overallAccuracy * 100),
                       d.accuracyDelta < -0.001 ? Theme.critical : Theme.signal)
                metric(store.t("Δ", "Change"), String(format: "%+.1f pts", d.accuracyDelta * 100),
                       d.accuracyDelta < -0.001 ? Theme.critical : Theme.signal)
            }

            Divider()

            Text(store.t("Per-class change", "Change for each topic")).font(.subheadline.bold())
            ForEach(store.adapter.classLabels, id: \.self) { cls in
                let delta = d.perClassDelta[cls] ?? 0
                ValueBar(label: cls,
                         value: min(1, abs(delta) / 0.5),
                         display: String(format: "%+.0f%%", delta * 100),
                         tint: delta < -0.001 ? Theme.critical : Theme.signal,
                         emphasized: d.collateral.contains(cls))
            }

            if d.collateral.isEmpty {
                Label(store.t("No collateral damage — good domains survived.",
                              "Nothing important broke — the topics that worked still work."),
                      systemImage: "checkmark.seal.fill")
                    .font(.callout).foregroundStyle(Theme.signal)
            } else {
                Label(store.t("Collateral damage: \(d.collateral.joined(separator: ", ")). "
                              + "You removed something a good class depended on.",
                              "This broke something that was working: \(d.collateral.joined(separator: ", ")). "
                              + "You removed a part that topic needed."),
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
