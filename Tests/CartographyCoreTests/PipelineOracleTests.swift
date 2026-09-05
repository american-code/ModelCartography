import Testing
@testable import CartographyCore

// Ground-truth oracle for the Attribution and Verification pipeline stages —
// the exact machinery T08 originally over-claimed compliance-grade rigor
// for. A MockAdapter gives each region/prediction a fully known, hand-set
// value, so every expected number below is computed by hand from the mock's
// own definition, not re-derived from the code under test.
final class MockAdapter: ModelAdapter {
    let name = "mock"
    let capabilities: Capabilities
    let classLabels: [String]
    private let regionList: [Region]
    /// [regionID: [truthClass: activation]] — a region's activation is a
    /// pure function of the input's ground-truth class, so per-class means
    /// are exactly the table values regardless of corpus composition.
    private let activationTable: [String: [String: Double]]
    /// [inputID: predictedClass] for the un-ablated model.
    private let basePrediction: [String: String]
    /// [regionID: [inputID: predictedClass]] — overrides basePrediction for
    /// any input listed, only once that region has been ablated.
    private let ablationOverride: [String: [String: String]]
    private let ablatedRegions: Set<String>

    init(classLabels: [String], regions: [Region],
         activationTable: [String: [String: Double]],
         basePrediction: [String: String],
         ablationOverride: [String: [String: String]] = [:],
         ablatedRegions: Set<String> = [],
         capabilities: Capabilities = [.verification, .ablation]) {
        self.classLabels = classLabels
        self.regionList = regions
        self.activationTable = activationTable
        self.basePrediction = basePrediction
        self.ablationOverride = ablationOverride
        self.ablatedRegions = ablatedRegions
        self.capabilities = capabilities
    }

    func regions() -> [Region] { regionList }

    func forward(_ input: CartographyInput) throws -> Trace {
        var acts: [String: Double] = [:]
        for r in regionList {
            acts[r.id] = ablatedRegions.contains(r.id) ? 0.0 : (activationTable[r.id]?[input.truth ?? ""] ?? 0.0)
        }
        var predicted = basePrediction[input.id] ?? (input.truth ?? classLabels[0])
        for region in ablatedRegions {
            if let override = ablationOverride[region]?[input.id] {
                predicted = override
            }
        }
        return Trace(inputID: input.id, activations: acts, routingPath: [],
                     probabilities: [:], predicted: predicted)
    }

    func saliency(for input: CartographyInput) throws -> SaliencyMap {
        throw CartographyError.unsupported("saliency")
    }

    func ablated(regionIDs: Set<String>) throws -> ModelAdapter {
        guard capabilities.contains(.ablation) else { throw CartographyError.unsupported("ablation") }
        return MockAdapter(classLabels: classLabels, regions: regionList,
                            activationTable: activationTable, basePrediction: basePrediction,
                            ablationOverride: ablationOverride,
                            ablatedRegions: ablatedRegions.union(regionIDs),
                            capabilities: capabilities)
    }
}

func makeInputs(cat: Int, dog: Int) -> [CartographyInput] {
    var inputs: [CartographyInput] = []
    for i in 0..<cat { inputs.append(CartographyInput(id: "cat\(i)", truth: "cat", payload: .vector([0]))) }
    for i in 0..<dog { inputs.append(CartographyInput(id: "dog\(i)", truth: "dog", payload: .vector([0]))) }
    return inputs
}

@Suite("Attribution ground-truth oracle")
struct AttributionOracleTests {

    @Test("Region labels and selectivity match hand-computed class means")
    func selectivityMatchesHandComputedMeans() {
        let regions = [
            Region(id: "n_catOnly", kind: .neuron, layerIndex: 0, indexInLayer: 0),
            Region(id: "n_dogOnly", kind: .neuron, layerIndex: 0, indexInLayer: 1),
            Region(id: "n_shared",  kind: .neuron, layerIndex: 0, indexInLayer: 2),
            Region(id: "n_silent",  kind: .neuron, layerIndex: 0, indexInLayer: 3),
        ]
        // Each region's mean activation per class is exactly this table (the
        // mock ignores individual inputs and returns the class mean
        // directly), so the expected label/selectivity can be computed by
        // hand from these eight numbers alone.
        let activationTable: [String: [String: Double]] = [
            "n_catOnly": ["cat": 10, "dog": 0],
            "n_dogOnly": ["cat": 0,  "dog": 8],
            "n_shared":  ["cat": 4,  "dog": 6],
            "n_silent":  ["cat": 0,  "dog": 0],
        ]
        let adapter = MockAdapter(classLabels: ["cat", "dog"], regions: regions,
                                   activationTable: activationTable, basePrediction: [:])
        let corpus = makeInputs(cat: 2, dog: 2)
        let labeling = Attribution.label(adapter: adapter, corpus: corpus, classLabels: ["cat", "dog"])

        #expect(labeling.labels["n_catOnly"] == "cat")
        #expect(abs((labeling.selectivity["n_catOnly"] ?? -1) - 1.0) < 1e-9)

        #expect(labeling.labels["n_dogOnly"] == "dog")
        #expect(abs((labeling.selectivity["n_dogOnly"] ?? -1) - 1.0) < 1e-9)

        // mean(cat)=4, mean(dog)=6 -> total=10, best class "dog" at 6/10.
        #expect(labeling.labels["n_shared"] == "dog")
        #expect(abs((labeling.selectivity["n_shared"] ?? -1) - 0.6) < 1e-9)

        // Zero activation for every class -> "(silent)", selectivity 0.
        #expect(labeling.labels["n_silent"] == "(silent)")
        #expect(abs((labeling.selectivity["n_silent"] ?? -1) - 0.0) < 1e-9)
    }
}

@Suite("Verification ground-truth oracle")
struct VerificationOracleTests {

    /// 5 cat inputs (4 correct, 1 predicted "dog"), 5 dog inputs (3 correct,
    /// 2 predicted "cat"). Every count below is exact by construction.
    static func baseAdapter() -> MockAdapter {
        let regions = [Region(id: "n1", kind: .neuron, layerIndex: 0, indexInLayer: 0)]
        let table: [String: [String: Double]] = ["n1": ["cat": 1, "dog": 1]]
        var predictions: [String: String] = [:]
        for i in 0..<5 { predictions["cat\(i)"] = i == 4 ? "dog" : "cat" }   // 1 wrong
        for i in 0..<5 { predictions["dog\(i)"] = i < 2 ? "cat" : "dog" }   // 2 wrong
        return MockAdapter(classLabels: ["cat", "dog"], regions: regions,
                            activationTable: table, basePrediction: predictions)
    }

    @Test("evaluate() matches hand-computed overall and per-class accuracy")
    func evaluateMatchesHandComputation() {
        let adapter = Self.baseAdapter()
        let corpus = makeInputs(cat: 5, dog: 5)
        let report = Verification.evaluate(adapter: adapter, corpus: corpus)

        #expect(report.count == 10)
        #expect(abs(report.overallAccuracy - 0.7) < 1e-9)     // 7/10 correct
        #expect(abs((report.perClass["cat"] ?? -1) - 0.8) < 1e-9)  // 4/5
        #expect(abs((report.perClass["dog"] ?? -1) - 0.6) < 1e-9)  // 3/5
    }

    @Test("confusion() matrix cells match hand-computed truth×predicted counts")
    func confusionMatchesHandComputation() {
        let adapter = Self.baseAdapter()
        let corpus = makeInputs(cat: 5, dog: 5)
        let m = Verification.confusion(adapter: adapter, corpus: corpus, labels: ["cat", "dog"])

        #expect(m.total == 10)
        #expect(m.counts[0][0] == 4)  // cat -> cat
        #expect(m.counts[0][1] == 1)  // cat -> dog
        #expect(m.counts[1][0] == 2)  // dog -> cat
        #expect(m.counts[1][1] == 3)  // dog -> dog
        #expect(abs(m.accuracy - 0.7) < 1e-9)
    }

    @Test("ablateAndCompare() reports the exact delta the ablation was defined to cause")
    func ablateAndCompareMatchesDefinedEffect() throws {
        let regions = [Region(id: "n_cat_detector", kind: .neuron, layerIndex: 0, indexInLayer: 0)]
        let table: [String: [String: Double]] = ["n_cat_detector": ["cat": 1, "dog": 0]]
        // Perfect base model: every input predicted correctly.
        var predictions: [String: String] = [:]
        for i in 0..<5 { predictions["cat\(i)"] = "cat" }
        for i in 0..<5 { predictions["dog\(i)"] = "dog" }
        // Ablating n_cat_detector flips exactly 2 of the 5 cat predictions to
        // "dog"; dog predictions are untouched.
        let ablationOverride: [String: [String: String]] = [
            "n_cat_detector": ["cat0": "dog", "cat1": "dog"]
        ]
        let adapter = MockAdapter(classLabels: ["cat", "dog"], regions: regions,
                                   activationTable: table, basePrediction: predictions,
                                   ablationOverride: ablationOverride)
        let corpus = makeInputs(cat: 5, dog: 5)

        let diff = try Verification.ablateAndCompare(adapter: adapter, regionIDs: ["n_cat_detector"],
                                                      corpus: corpus, collateralThreshold: 0.05)

        #expect(abs(diff.base.overallAccuracy - 1.0) < 1e-9)
        #expect(abs(diff.modified.overallAccuracy - 0.8) < 1e-9)          // 8/10 now correct
        #expect(abs((diff.perClassDelta["cat"] ?? 1) - (-0.4)) < 1e-9)    // 5/5 -> 3/5
        #expect(abs((diff.perClassDelta["dog"] ?? 1) - 0.0) < 1e-9)       // untouched
        #expect(diff.collateral == ["cat"])
        #expect(abs(diff.accuracyDelta - (-0.2)) < 1e-9)
    }

    @Test("ablateAndCompare() throws when the adapter doesn't declare .ablation")
    func ablateAndCompareThrowsWithoutCapability() {
        let regions = [Region(id: "n1", kind: .neuron, layerIndex: 0, indexInLayer: 0)]
        let adapter = MockAdapter(classLabels: ["cat", "dog"], regions: regions,
                                   activationTable: [:], basePrediction: [:],
                                   capabilities: [.verification])
        let corpus = makeInputs(cat: 1, dog: 1)
        #expect(throws: CartographyError.self) {
            _ = try Verification.ablateAndCompare(adapter: adapter, regionIDs: ["n1"], corpus: corpus)
        }
    }
}
