//
//  ModelCartographyApp.swift
//  Entry point. One codebase, one window group — runs on macOS, iPadOS, and tvOS.
//

import SwiftUI

@main
struct ModelCartographyApp: App {
    @State private var store = MapStore()

    var body: some Scene {
        WindowGroup {
            RootView(store: store)
        }
        #if os(macOS)
        .defaultSize(width: 1000, height: 760)
        #endif
    }
}

struct RootView: View {
    @Bindable var store: MapStore

    var body: some View {
        TabView {
            CortexView(store: store)
                .tabItem { Label("Cortex", systemImage: "square.grid.3x3.fill") }
            TraceView(store: store)
                .tabItem { Label("Trace", systemImage: "arrow.triangle.branch") }
            IntervenePanel(store: store)
                .tabItem { Label("Intervene", systemImage: "scissors") }
        }
    }
}
