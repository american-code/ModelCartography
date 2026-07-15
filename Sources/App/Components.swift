//
//  Components.swift
//  Small shared building blocks used across the three tabs.
//

import SwiftUI

extension View {
    func card() -> some View {
        self.padding(16)
            .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 14))
    }
}

/// A grayscale grid rendered as a tiny image, with an optional amber saliency overlay.
struct GridThumbnail: View {
    let grid: [[Double]]
    var saliency: SaliencyMap? = nil

    var body: some View {
        Canvas { ctx, size in
            let rows = grid.count
            let cols = grid.first?.count ?? 0
            guard rows > 0, cols > 0 else { return }
            let cw = size.width / CGFloat(cols)
            let ch = size.height / CGFloat(rows)
            for r in 0..<rows {
                for c in 0..<cols {
                    let v = max(0, min(1, grid[r][c]))
                    let rect = CGRect(x: CGFloat(c) * cw, y: CGFloat(r) * ch,
                                      width: cw + 0.5, height: ch + 0.5)
                    ctx.fill(Path(rect), with: .color(Color(white: v)))
                }
            }
            if let s = saliency, s.rows > 0, s.cols > 0 {
                let sw = size.width / CGFloat(s.cols)
                let sh = size.height / CGFloat(s.rows)
                for r in 0..<s.rows {
                    for c in 0..<s.cols {
                        let v = s.values[r][c]
                        if v <= 0.05 { continue }
                        let rect = CGRect(x: CGFloat(c) * sw, y: CGFloat(r) * sh,
                                          width: sw, height: sh)
                        ctx.fill(Path(rect), with: .color(Theme.salience(v)))
                    }
                }
            }
        }
        .background(Color.black.opacity(0.18))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

/// A labeled horizontal bar (probabilities, deltas).
struct ValueBar: View {
    let label: String
    let value: Double        // 0...1 of the track width to fill
    let display: String
    var tint: Color = Theme.signal
    var emphasized: Bool = false

    var body: some View {
        HStack(spacing: 10) {
            Text(label)
                .font(.caption.monospaced())
                .frame(width: 92, alignment: .leading)
                .lineLimit(1)
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.primary.opacity(0.08))
                    Capsule().fill(tint)
                        .frame(width: max(2, geo.size.width * max(0, min(1, value))))
                }
            }
            .frame(height: 10)
            Text(display)
                .font(.caption.monospacedDigit())
                .foregroundStyle(emphasized ? tint : .secondary)
                .frame(width: 56, alignment: .trailing)
        }
    }
}

struct Chip: View {
    let text: String
    var tint: Color = Theme.signal
    var body: some View {
        Text(text)
            .font(.caption2.monospaced())
            .padding(.horizontal, 8).padding(.vertical, 4)
            .background(tint.opacity(0.16), in: Capsule())
            .foregroundStyle(tint)
    }
}

/// Human-readable name for a cortex column.
func layerTitle(for regions: [Region]) -> String {
    guard let kind = regions.first?.kind, let idx = regions.first?.layerIndex else { return "Layer" }
    switch kind {
    case .logit: return "Output"
    case .expert: return "Experts L\(idx)"
    case .head: return "Heads L\(idx)"
    default: return "Layer \(idx)"
    }
}
