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
    private(set) var confusion: ConfusionMatrix?

    var selectedInput: CartographyInput?
    private(set) var currentTrace: Trace?
    private(set) var currentSaliency: SaliencyMap?

    // Phase 2 per-input readouts.
    private(set) var currentLens: [LayerReadout] = []
    private(set) var currentAttribution: AttributionGraph?

    // Interp hook-extracted data.
    private(set) var currentAttentionPatterns: [AttentionHeadPattern] = []

    // Circuit / activation patching state.
    var circuitCleanPrompt: String = "the quick fox jumped"
    var circuitCorruptedPrompt: String = "the lazy dog sat"
    private(set) var patchingMatrix: PatchingMatrix?
    private(set) var circuitSelectedLayer: Int?
    private(set) var circuitSelectedHead: Int?
    private(set) var circuitHeadPattern: AttentionHeadPattern?

    /// Regions the user has marked for ablation (neurons only).
    var selectedRegionIDs: Set<String> = []
    private(set) var diff: DiffReport?
    private(set) var status: String = ""

    /// Plain-language mode: swaps jargon labels/captions for everyday wording.
    var simplified = false
    /// Pick the wording for the current mode. `t("Logit lens", "How the guess forms")`.
    func t(_ expert: String, _ plain: String) -> String { simplified ? plain : expert }

    // Steering (Phase 2 intervention).
    var steerFeatureID: String?
    var steerGain: Double = 4.0
    private(set) var steerBase: [String: Double]?
    private(set) var steerResult: [String: Double]?

    // Expert utilization (Phase 3): mean router gate per expert over the corpus.
    private(set) var expertUtil: [String: Double] = [:]

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
        // Neuron-style adapters get their labels from attribution; feature-style adapters
        // (dense + SAE) arrive already labeled, so only overwrite where attribution has data.
        if adapter.capabilities.contains(.internalActivations) {
            labeling = Attribution.label(adapter: adapter, corpus: corpus, classLabels: adapter.classLabels)
            for i in regs.indices {
                if let l = labeling.labels[regs[i].id] { regs[i].label = l }
                if let s = labeling.selectivity[regs[i].id] { regs[i].selectivity = s }
            }
        } else {
            labeling = Attribution.Labeling(labels: [:], selectivity: [:])
        }
        regions = regs
        if adapter.capabilities.contains(.verification) {
            baseline = Verification.evaluate(adapter: adapter, corpus: evalCorpus)
            confusion = Verification.confusion(adapter: adapter, corpus: evalCorpus,
                                               labels: adapter.classLabels)
        } else {
            baseline = nil; confusion = nil
        }
        selectedRegionIDs = []
        diff = nil
        steerFeatureID = regs.first { $0.kind == .feature }?.id
        steerBase = nil; steerResult = nil
        patchingMatrix = nil; circuitSelectedLayer = nil; circuitSelectedHead = nil; circuitHeadPattern = nil
        computeExpertUtilization()
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
        if c.contains(.steering) { names.append("steering") }
        if c.contains(.logitLens) { names.append("logit-lens") }
        if c.contains(.attentionPatterns) { names.append("attention") }
        if c.contains(.activationPatching) { names.append("patching") }
        return names.joined(separator: " · ")
    }

    var adapterName: String { adapter.name }
    var isCircuitSupported: Bool { adapter.capabilities.contains(.activationPatching) }

    /// Non-nil when the active adapter has a fixed vocabulary and the prompt pair
    /// contains words outside it. Unknown tokens are silently mapped to index 0.
    var circuitOOVWarning: String? {
        #if os(macOS) || os(iOS)
        guard adapter is InterpAdapter else { return nil }
        let vocab = InterpAdapter.vocab
        let oov: (String) -> [String] = { text in
            text.lowercased().split(separator: " ").map(String.init).filter { !vocab.contains($0) }
        }
        let words = Array(Set(oov(circuitCleanPrompt) + oov(circuitCorruptedPrompt))).sorted()
        guard !words.isEmpty else { return nil }
        return "Unknown tokens (mapped to "\(vocab[0])"): \(words.joined(separator: ", "))"
        #else
        return nil
        #endif
    }
    var canAblate: Bool { adapter.capabilities.contains(.ablation) }
    var canSaliency: Bool { adapter.capabilities.contains(.saliency) }
    var canSteer: Bool { adapter.capabilities.contains(.steering) }
    var hasLens: Bool { adapter.capabilities.contains(.logitLens) }
    var hasAttribution: Bool { adapter.capabilities.contains(.attribution) }
    var hasAttentionPatterns: Bool { adapter.capabilities.contains(.attentionPatterns) }
    var featureRegions: [Region] { regions.filter { $0.kind == .feature } }
    var hasExperts: Bool { regions.contains { $0.kind == .expert } }

    /// Experts ranked by how much the router uses them across the corpus (hot → cold).
    var expertsByUtilization: [(region: Region, util: Double)] {
        regions.filter { $0.kind == .expert }
            .map { ($0, expertUtil[$0.id] ?? 0) }
            .sorted { $0.1 > $1.1 }
    }

    private func computeExpertUtilization() {
        expertUtil = [:]
        let experts = regions.filter { $0.kind == .expert }
        guard !experts.isEmpty, !corpus.isEmpty else { return }
        var sum: [String: Double] = [:]
        for input in corpus {
            guard let tr = try? adapter.forward(input) else { continue }
            for e in experts { sum[e.id, default: 0] += tr.activations[e.id] ?? 0 }
        }
        for e in experts { expertUtil[e.id] = (sum[e.id] ?? 0) / Double(corpus.count) }
    }

    /// Mark every expert the router barely uses — safe pruning candidates.
    func markColdExperts(threshold: Double = 0.05) {
        let cold = expertsByUtilization.filter { $0.util < threshold }.map { $0.region.id }
        if cold.isEmpty {
            status = "No cold experts below \(Int(threshold * 100))% utilization — this MoE is well-packed."
        } else {
            selectedRegionIDs.formUnion(cold)
            status = "Marked \(cold.count) cold expert(s) for pruning."
        }
    }

    // MARK: Stage 1–2: capture on selection

    func select(_ input: CartographyInput?) {
        selectedInput = input
        steerBase = nil; steerResult = nil
        guard let input else {
            currentTrace = nil; currentSaliency = nil; currentLens = []
            currentAttribution = nil; currentAttentionPatterns = []
            return
        }
        currentTrace = try? adapter.forward(input)
        currentSaliency = canSaliency ? try? adapter.saliency(for: input) : nil
        currentLens = hasLens ? ((try? adapter.logitLens(input)) ?? []) : []
        currentAttribution = hasAttribution ? try? adapter.attributionGraph(input) : nil
        currentAttentionPatterns = hasAttentionPatterns
            ? ((try? adapter.attentionPatterns(input)) ?? []) : []
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

    // MARK: Stage 5: steering (Phase 2)

    func selectFeature(_ id: String) {
        steerFeatureID = id
        steerBase = nil; steerResult = nil
    }

    func applySteering() {
        guard canSteer, let id = steerFeatureID, let input = selectedInput else { return }
        do {
            let steered = try adapter.steered(featureID: id, gain: steerGain)
            steerBase = currentTrace?.probabilities
            steerResult = try steered.forward(input).probabilities
            let before = steerBase?.max(by: { $0.value < $1.value })?.key ?? "?"
            let after = steerResult?.max(by: { $0.value < $1.value })?.key ?? "?"
            status = before == after
                ? "Steered by \(featureLabel(id)) ×\(String(format: "%.1f", steerGain)): still \(after)."
                : "Steered by \(featureLabel(id)) ×\(String(format: "%.1f", steerGain)): \(before) → \(after)."
        } catch {
            status = "Steering failed: \(error)"
        }
    }

    func featureLabel(_ id: String) -> String {
        regions.first { $0.id == id }?.label ?? id
    }

    // MARK: Circuit / activation patching

    func runPatchingSweep() {
        guard adapter.capabilities.contains(.activationPatching) else { return }
        let clean = CartographyInput(id: "ckt_clean", display: circuitCleanPrompt, payload: .vector([]))
        let corrupted = CartographyInput(id: "ckt_corrupted", display: circuitCorruptedPrompt, payload: .vector([]))
        do {
            patchingMatrix = try adapter.patchingSweep(clean: clean, corrupted: corrupted)
            circuitSelectedLayer = nil; circuitSelectedHead = nil; circuitHeadPattern = nil
            let m = patchingMatrix!
            status = "Patching sweep complete · \(m.layers)L × \(m.heads)H"
        } catch {
            status = "Patching sweep failed: \(error)"
        }
    }

    func selectCircuitHead(layer: Int, head: Int) {
        circuitSelectedLayer = layer
        circuitSelectedHead = head
        let clean = CartographyInput(id: "ckt_clean", display: circuitCleanPrompt, payload: .vector([]))
        circuitHeadPattern = (try? adapter.attentionPatterns(clean))?
            .first { $0.layerIndex == layer && $0.headIndex == head }
    }

    // MARK: Adapter switching

    private func setGridCorpora() {
        corpus = PatternDataset.corpus(perClass: 40, seed: 1)
        evalCorpus = PatternDataset.corpus(perClass: 20, seed: 99)
    }
    private func setTextCorpora() {
        corpus = TopicDataset.corpus(perTopic: 30, seed: 2)
        evalCorpus = TopicDataset.corpus(perTopic: 20, seed: 77)
    }

    func useDemoNet() {
        setGridCorpora()
        let net = MockNetwork(inputSize: gridSide * gridSide, h1Size: 16, h2Size: 16,
                              classes: PatternDataset.labels, seed: 42)
        MockNetworkTrainer.train(net, corpus: PatternDataset.corpus(perClass: 40, seed: 1))
        adapter = MockNetworkAdapter(net: net, gridSide: gridSide)
        refreshForNewAdapter()
    }

    func useDenseTextModel() {
        setTextCorpora()
        adapter = DenseModelAdapter.make()
        refreshForNewAdapter()
    }

    func useMoEModel() {
        setTextCorpora()
        adapter = MoEAdapter.make()
        refreshForNewAdapter()
    }

    #if os(macOS) || os(iOS)
    /// Load the SwiftSci Interp demo adapter (HookedGPT2 with logit lens + attention patterns).
    func useInterpModel() {
        let a = InterpAdapter.make()
        corpus = InterpAdapter.demoCorpus()
        evalCorpus = InterpAdapter.demoCorpus()
        adapter = a
        refreshForNewAdapter()
    }
    #endif

    /// Import a sample external routing log (generated from our MoE, round-tripped via JSON).
    func useRoutingLogSample() {
        let a = RoutingLogAdapter.sample()
        adapter = a
        corpus = a.inputs
        evalCorpus = a.inputs
        refreshForNewAdapter()
    }

    /// Load a real external routing log in the RoutingLog JSON schema.
    func loadRoutingLog(url: URL) {
        do {
            let scoped = url.startAccessingSecurityScopedResource()
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }
            let log = try RoutingLog.decoded(from: try Data(contentsOf: url))
            let a = RoutingLogAdapter(log: log)
            adapter = a
            corpus = a.inputs
            evalCorpus = a.inputs
            refreshForNewAdapter()
        } catch {
            status = "Couldn't load routing log: \(error)"
        }
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
            setGridCorpora()   // Core ML image classifier expects the grid inputs
            adapter = try CoreMLAdapter(model: model,
                                        name: url.deletingPathExtension().lastPathComponent)
            refreshForNewAdapter()
        } catch {
            status = "Couldn't load model: \(error)"
        }
    }
    #endif
}
