//
//  Verification.swift
//  Stage 5's safety rail: the closed loop that makes "remove the unuseful sections"
//  trustworthy instead of reckless.
//
//  Every intervention is re-run over a held-out corpus and diffed against the baseline.
//  Collateral damage — a *good* class whose accuracy dropped — is flagged explicitly.
//

import Foundation

public struct VerificationReport: Sendable {
    public let overallAccuracy: Double
    public let perClass: [String: Double]
    public let count: Int
}

public struct DiffReport: Sendable {
    public let ablated: [String]
    public let base: VerificationReport
    public let modified: VerificationReport
    public let perClassDelta: [String: Double]
    /// Classes whose accuracy fell by more than the collateral threshold.
    public let collateral: [String]

    public var accuracyDelta: Double { modified.overallAccuracy - base.overallAccuracy }
}

public enum Verification {
    /// Fraction correct, overall and per class.
    public static func evaluate(adapter: ModelAdapter,
                                corpus: [CartographyInput]) -> VerificationReport {
        var correct = 0
        var total = 0
        var perClassCorrect: [String: Int] = [:]
        var perClassTotal: [String: Int] = [:]

        for input in corpus {
            guard let truth = input.truth, let trace = try? adapter.forward(input) else { continue }
            total += 1
            perClassTotal[truth, default: 0] += 1
            if trace.predicted == truth {
                correct += 1
                perClassCorrect[truth, default: 0] += 1
            }
        }

        var perClass: [String: Double] = [:]
        for (cls, t) in perClassTotal {
            perClass[cls] = t > 0 ? Double(perClassCorrect[cls] ?? 0) / Double(t) : 0
        }
        return VerificationReport(overallAccuracy: total > 0 ? Double(correct) / Double(total) : 0,
                                  perClass: perClass, count: total)
    }

    /// Ablate the given regions, re-evaluate, and diff.
    public static func ablateAndCompare(adapter: ModelAdapter,
                                        regionIDs: Set<String>,
                                        corpus: [CartographyInput],
                                        collateralThreshold: Double = 0.05) throws -> DiffReport {
        guard adapter.capabilities.contains(.ablation) else {
            throw CartographyError.unsupported("ablation")
        }
        let base = evaluate(adapter: adapter, corpus: corpus)
        let modifiedAdapter = try adapter.ablated(regionIDs: regionIDs)
        let modified = evaluate(adapter: modifiedAdapter, corpus: corpus)

        var deltas: [String: Double] = [:]
        var collateral: [String] = []
        for cls in Set(base.perClass.keys).union(modified.perClass.keys) {
            let d = (modified.perClass[cls] ?? 0) - (base.perClass[cls] ?? 0)
            deltas[cls] = d
            if d < -collateralThreshold { collateral.append(cls) }
        }
        return DiffReport(ablated: regionIDs.sorted(), base: base, modified: modified,
                          perClassDelta: deltas, collateral: collateral.sorted())
    }
}
