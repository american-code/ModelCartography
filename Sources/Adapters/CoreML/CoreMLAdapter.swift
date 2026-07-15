//
//  CoreMLAdapter.swift
//  Wraps any Core ML image classifier through the same ModelAdapter interface.
//
//  Core ML is a black box for internal activations, so this adapter is honest about
//  it: `capabilities` advertises only `.saliency` (occlusion-based, forward-only) and
//  `.verification`. It cannot ablate an internal region, so the Intervene panel simply
//  won't offer that action — exactly the "degrade gracefully, never fake" design.
//

import Foundation
import CoreML
import Vision
import CoreGraphics

public final class CoreMLAdapter: ModelAdapter {
    private let vnModel: VNCoreMLModel
    private let labels: [String]
    public let name: String

    /// Saliency resolution — the input grid is occluded one block at a time.
    private let occlusionSide = 6

    public init(model: MLModel, name: String) throws {
        self.name = name
        do { self.vnModel = try VNCoreMLModel(for: model) }
        catch { throw CartographyError.modelLoad(String(describing: error)) }

        if let raw = model.modelDescription.classLabels {
            self.labels = raw.map { String(describing: $0) }
        } else {
            throw CartographyError.modelLoad("model is not a classifier (no class labels)")
        }
        guard !labels.isEmpty else {
            throw CartographyError.modelLoad("classifier reported zero classes")
        }
    }

    // Honest capability set: outputs + saliency only. No internals, no ablation, no steering.
    public var capabilities: Capabilities { [.saliency, .verification] }
    public var classLabels: [String] { labels }

    public func regions() -> [Region] {
        labels.enumerated().map {
            Region(id: "OUT.\($1)", kind: .logit, layerIndex: 1, indexInLayer: $0)
        }
    }

    // MARK: Inference

    public func forward(_ input: CartographyInput) throws -> Trace {
        guard case .grid(let g) = input.payload else {
            throw CartographyError.badInput("Core ML adapter expects a grid input")
        }
        let probs = try classify(Self.cgImage(fromGrid: g))
        let predicted = probs.max(by: { $0.value < $1.value })?.key ?? (labels.first ?? "—")
        var acts: [String: Double] = [:]
        for (label, p) in probs { acts["OUT.\(label)"] = p }
        return Trace(inputID: input.id, activations: acts,
                     routingPath: ["OUT.\(predicted)"], probabilities: probs, predicted: predicted)
    }

    public func saliency(for input: CartographyInput) throws -> SaliencyMap {
        guard case .grid(let g) = input.payload, let firstRow = g.first else {
            throw CartographyError.badInput("Core ML adapter expects a grid input")
        }
        let rows = g.count, cols = firstRow.count
        let baseProbs = try classify(Self.cgImage(fromGrid: g))
        guard let top = baseProbs.max(by: { $0.value < $1.value }) else {
            throw CartographyError.badInput("classifier returned no results")
        }

        let P = occlusionSide
        var sal = [[Double]](repeating: [Double](repeating: 0, count: P), count: P)
        for br in 0..<P {
            for bc in 0..<P {
                var occluded = g
                let r0 = br * rows / P, r1 = max(r0 + 1, (br + 1) * rows / P)
                let c0 = bc * cols / P, c1 = max(c0 + 1, (bc + 1) * cols / P)
                for r in r0..<min(r1, rows) {
                    for c in c0..<min(c1, cols) { occluded[r][c] = 0.1 }
                }
                let probs = try classify(Self.cgImage(fromGrid: occluded))
                // Importance = how much occluding this block hurt the winning class.
                sal[br][bc] = max(0, top.value - (probs[top.key] ?? 0))
            }
        }
        let maxV = sal.flatMap { $0 }.max() ?? 1
        if maxV > 0 { sal = sal.map { $0.map { $0 / maxV } } }
        return SaliencyMap(rows: P, cols: P, values: sal)
    }

    // MARK: Vision plumbing

    private func classify(_ cg: CGImage) throws -> [String: Double] {
        let request = VNCoreMLRequest(model: vnModel)
        request.imageCropAndScaleOption = .scaleFill
        try VNImageRequestHandler(cgImage: cg, options: [:]).perform([request])
        let observations = (request.results as? [VNClassificationObservation]) ?? []
        var out: [String: Double] = [:]
        for o in observations { out[o.identifier] = Double(o.confidence) }
        return out
    }

    /// Render a grayscale grid (0...1) into a CGImage for the vision request.
    static func cgImage(fromGrid g: [[Double]], upscale: Int = 8) throws -> CGImage {
        guard let firstRow = g.first else { throw CartographyError.badInput("empty grid") }
        let h = g.count, w = firstRow.count
        let outW = w * upscale, outH = h * upscale
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: outW * outH)
        defer { buffer.deallocate() }
        for y in 0..<outH {
            let sr = y / upscale
            for x in 0..<outW {
                let v = g[sr][x / upscale]
                buffer[y * outW + x] = UInt8(max(0, min(1, v)) * 255)
            }
        }
        let space = CGColorSpaceCreateDeviceGray()
        guard let ctx = CGContext(data: buffer, width: outW, height: outH,
                                  bitsPerComponent: 8, bytesPerRow: outW, space: space,
                                  bitmapInfo: CGImageAlphaInfo.none.rawValue),
              let image = ctx.makeImage() else {
            throw CartographyError.badInput("could not rasterize grid")
        }
        return image
    }
}
