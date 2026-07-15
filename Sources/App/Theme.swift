//
//  Theme.swift
//  Palette carried over from the Model Cartography design note: teal = structure,
//  amber = trace/action, a restrained rust for collateral warnings.
//

import SwiftUI

enum Theme {
    static let signal   = Color(red: 0.10, green: 0.58, blue: 0.52)   // teal — the map
    static let trace    = Color(red: 0.73, green: 0.39, blue: 0.14)   // amber — action
    static let critical = Color(red: 0.69, green: 0.21, blue: 0.29)   // rust — collateral
    static let muted    = Color.gray                                  // neutral bars

    /// Teal heat ramp for activation cells, legible on either theme.
    static func heat(_ v: Double) -> Color {
        signal.opacity(0.10 + 0.85 * max(0, min(1, v)))
    }

    /// Amber ramp for saliency overlays.
    static func salience(_ v: Double) -> Color {
        trace.opacity(0.9 * max(0, min(1, v)))
    }

    static let cellCorner: CGFloat = 6
}
