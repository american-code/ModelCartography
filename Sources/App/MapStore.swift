//
//  MapStore.swift
//  The single source of truth the UI observes. Owns the active adapter and drives the
//  five stages: it instruments/captures on selection, holds the attributed map, and
//  runs the ablate -> re-verify loop.
//

import Foundation
import Observation
#if canImport(CoreML)
import CoreML
#endif

@MainActor
@Observable
final class MapStore {
    private(set) var adapter: ModelAdapter
    private(set) var regions: [Region] = []

    /// Browsable inputs shown in the Trace tab + used for attribution.
    private(set) var corpus: [CartographyInput] = []
    /// Held-out inputs used only for verification, so an edit can't "cheat" its eval.
    private var evalCorpus: [CartographyInput] = []

    private(set) var labeling = Attribution.Labeling(labels: [:], selectivity: [:])
    private(set) var baseline: VerificationReport?

    var selectedInput: CartographyInput?
    private(set) var currentTrace: Trace?
    private(set) var currentSaliency: SaliencyMap?

    /// Regions the user has marked for ablation (neurons only).
    var selectedRegionIDs: Set<String> = []
    private(set) var diff: DiffReport?
    private(set) var status: String = ""

    private let gridSide = PatternDataset.side

    init() {
        // Build + train the demo net synchronously — it is tiny.
        let net = MockNetwork(inputSize: PatternDataset.side * PatternDataset.side,
                              h1Size: 16, h2Size: 16,
                              classes: PatternDataset.labels, seed: 42)
        let train = PatternDataset.corpus(perClass: 40, seed: 1)
        MockNetworkTrainer.train(net, corpus: train)

        self.adapter = MockNetworkAdapter(net: net, gridSide: PatternDataset.side)
        self.corpus = train
        self.evalCorpus = PatternDataset.corpus(perClass: 20, seed: 99)
        refreshForNewAdapter()
    }

    // MARK: Adapter lifecycle

    private func refreshForNewAdapter() {
        var regs = adapter.regions()
        if adapter.capabilities.contains(.internalActivations) {
            labeling = Attribution.label(adapter: adapter, corpus: corpus, classLabels: adapter.classLabels)
            for i in regs.indices {
                regs[i].label = labeling.labels[regs[i].id]
                regs[i].selectivity = labeling.selectivity[regs[i].id]
            }
        } else {
            labeling = Attribution.Labeling(labels: [:], selectivity: [:])
        }
        regions = regs
        baseline = adapter.capabilities.contains(.verification)
            ? Verification.evaluate(adapter: adapter, corpus: evalCorpus) : nil
        selectedRegionIDs = []
        diff = nil
        select(corpus.first)
        status = "Loaded \(adapter.name) · \(regions.count) regions · \(capabilitySummary)"
    }

    var capabilitySummary: String {
        var names: [String] = []
        let c = adapter.capabilities
        if c.contains(.internalActivations) { names.append("internals") }
        if c.contains(.saliency) { names.append("saliency") }
        if c.contains(.ablation) { names.append("ablation") }
        if c.contains(.semanticFeatures) { names.append("features") }
        return names.joined(separator: " · ")
    }

    var adapterName: String { adapter.name }
    var canAblate: Bool { adapter.capabilities.contains(.ablation) }
    var canSaliency: Bool { adapter.capabilities.contains(.saliency) }

    // MARK: Stage 1–2: capture on selection

    func select(_ input: CartographyInput?) {
        selectedInput = input
        guard let input else { currentTrace = nil; currentSaliency = nil; return }
        currentTrace = try? adapter.forward(input)
        currentSaliency = canSaliency ? try? adapter.saliency(for: input) : nil
    }

    // MARK: Cortex display helpers

    var columns: [(layer: Int, regions: [Region])] {
        Dictionary(grouping: regions, by: \.layerIndex)
            .map { (layer: $0.key, regions: $0.value.sorted { $0.indexInLayer < $1.indexInLayer }) }
            .sorted { $0.layer < $1.layer }
    }

    /// Activation for a region under the current trace, normalized 0...1 for display.
    func displayActivation(_ region: Region) -> Double {
        guard let trace = currentTrace else { return 0 }
        let raw = trace.activations[region.id] ?? 0
        if region.kind == .logit { return raw }           // already a probability
        let peak = maxNeuronActivation
        return peak > 0 ? raw / peak : 0
    }

    private var maxNeuronActivation: Double {
        guard let trace = currentTrace else { return 0 }
        return regions.filter { $0.kind != .logit }
            .map { trace.activations[$0.id] ?? 0 }.max() ?? 0
    }

    func isOnPath(_ region: Region) -> Bool {
        currentTrace?.routingPath.contains(region.id) ?? false
    }

    func isAblatable(_ region: Region) -> Bool {
        canAblate && (region.kind == .neuron || region.kind == .channel || region.kind == .expert)
    }

    func toggleSelection(_ region: Region) {
        guard isAblatable(region) else { return }
        if selectedRegionIDs.contains(region.id) { selectedRegionIDs.remove(region.id) }
        else { selectedRegionIDs.insert(region.id) }
    }

    /// Whether "mark a whole domain" quick actions make sense for this model.
    var canMarkDomains: Bool {
        canAblate && adapter.capabilities.contains(.internalActivations)
    }

    /// Mark every ablatable neuron whose attributed domain is `label`.
    /// Removing a whole domain is what actually moves accuracy in a redundant net.
    func markDomain(_ label: String) {
        let ids = regions.filter { isAblatable($0) && $0.label == label }.map(\.id)
        selectedRegionIDs.formUnion(ids)
    }

    // MARK: Stage 5: intervene + verify

    func ablateAndVerify() {
        guard canAblate, !selectedRegionIDs.isEmpty else { return }
        do {
            diff = try Verification.ablateAndCompare(adapter: adapter,
                                                     regionIDs: selectedRegionIDs,
                                                     corpus: evalCorpus)
            if let d = diff {
                let pct = String(format: "%+.1f", d.accuracyDelta * 100)
                status = d.collateral.isEmpty
                    ? "Removed \(selectedRegionIDs.count) region(s): accuracy \(pct) pts, no collateral damage."
                    : "Removed \(selectedRegionIDs.count) region(s): accuracy \(pct) pts — collateral: \(d.collateral.joined(separator: ", "))."
            }
        } catch {
            status = "Ablation failed: \(error)"
        }
    }

    func clearIntervention() {
        selectedRegionIDs = []
        diff = nil
        status = "Selection cleared."
    }

    // MARK: Adapter switching

    func useDemoNet() {
        let net = MockNetwork(inputSize: gridSide * gridSide, h1Size: 16, h2Size: 16,
                              classes: PatternDataset.labels, seed: 42)
        MockNetworkTrainer.train(net, corpus: PatternDataset.corpus(perClass: 40, seed: 1))
        adapter = MockNetworkAdapter(net: net, gridSide: gridSide)
        refreshForNewAdapter()
    }

    #if canImport(CoreML)
    /// Load a Core ML image classifier (.mlmodel / .mlmodelc / .mlpackage). macOS & iOS.
    func loadCoreML(url: URL) {
        do {
            let needsScope = url.startAccessingSecurityScopedResource()
            defer { if needsScope { url.stopAccessingSecurityScopedResource() } }

            let compiled: URL = (url.pathExtension == "mlmodelc")
                ? url : try MLModel.compileModel(at: url)
            let model = try MLModel(contentsOf: compiled)
            adapter = try CoreMLAdapter(model: model,
                                        name: url.deletingPathExtension().lastPathComponent)
            refreshForNewAdapter()
        } catch {
            status = "Couldn't load model: \(error)"
        }
    }
    #endif
}
