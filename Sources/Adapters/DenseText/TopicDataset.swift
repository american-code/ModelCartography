//
//  TopicDataset.swift
//  A tiny, deterministic text corpus: short sentences drawn from topic vocabularies.
//  Small enough to train on-device, structured enough that sparse-autoencoder features
//  land on recognizable domains.
//

import Foundation

public struct TopicDataset {
    public static let topics = ["weather", "finance", "food"]

    // Topic-specific vocabularies. Overlap is intentionally low so features separate.
    static let vocab: [String: [String]] = [
        "weather": ["rain", "storm", "sunny", "cloud", "wind", "snow", "forecast", "humid", "chilly", "thunder"],
        "finance": ["stock", "market", "profit", "invest", "bond", "revenue", "trade", "interest", "shares", "budget"],
        "food":    ["pasta", "spicy", "recipe", "garlic", "dessert", "grill", "savory", "bakery", "fresh", "dinner"],
    ]
    // Shared filler that carries no topic signal.
    static let filler = ["the", "a", "very", "today", "some", "really"]

    /// Ordered vocabulary -> index, so a sentence becomes a bag-of-words vector.
    public static let words: [String] = {
        var all: [String] = []
        for t in topics { all += vocab[t]! }
        all += filler
        return all
    }()
    public static let index: [String: Int] =
        Dictionary(uniqueKeysWithValues: words.enumerated().map { ($1, $0) })

    public static var vocabSize: Int { words.count }

    /// Normalized bag-of-words vector for a token list.
    public static func encode(_ tokens: [String]) -> [Double] {
        var v = [Double](repeating: 0, count: vocabSize)
        for tok in tokens { if let i = index[tok] { v[i] += 1 } }
        let n = tokens.count
        if n > 0 { for i in v.indices { v[i] /= Double(n) } }
        return v
    }

    static func sentence(_ topic: String, rng: inout SplitMix64) -> [String] {
        let pool = vocab[topic]!
        let count = 4 + Int(rng.unit() * 3)      // 4...6 topic words
        var toks: [String] = []
        for _ in 0..<count { toks.append(pool[Int(rng.unit() * Double(pool.count))]) }
        if rng.unit() > 0.4 { toks.append(filler[Int(rng.unit() * Double(filler.count))]) }
        // Shuffle for order independence.
        for i in stride(from: toks.count - 1, through: 1, by: -1) {
            let j = Int(rng.unit() * Double(i + 1)); toks.swapAt(i, j)
        }
        return toks
    }

    public static func corpus(perTopic: Int, seed: UInt64) -> [CartographyInput] {
        var rng = SplitMix64(seed: seed)
        var out: [CartographyInput] = []
        var idx = 0
        for topic in topics {
            for _ in 0..<perTopic {
                let toks = sentence(topic, rng: &rng)
                out.append(CartographyInput(id: "t\(idx)", truth: topic,
                                            display: toks.joined(separator: " "),
                                            payload: .vector(encode(toks))))
                idx += 1
            }
        }
        return out
    }
}
