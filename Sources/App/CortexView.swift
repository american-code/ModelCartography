//
//  CortexView.swift
//  Stage 4 (Map): the navigable cortex. Columns are layers, cells are regions, color
//  is activation under the currently-selected input. Tap a hidden neuron to mark it
//  for ablation over on the Intervene tab.
//

import SwiftUI
import UniformTypeIdentifiers

struct CortexView: View {
    @Bindable var store: MapStore
    #if os(macOS) || os(iOS)
    @State private var showingImporter = false
    @State private var showingLogImporter = false
    #endif

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                header
                if let input = store.selectedInput {
                    currentInputStrip(input: input)
                }
                cortex
                legend
            }
            .padding(20)
            .frame(maxWidth: 900, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
        #if os(macOS) || os(iOS)
        .fileImporter(isPresented: $showingImporter,
                      allowedContentTypes: CoreMLModelType.all,
                      allowsMultipleSelection: false) { result in
            if case .success(let urls) = result, let url = urls.first {
                store.loadCoreML(url: url)
            }
        }
        .fileImporter(isPresented: $showingLogImporter,
                      allowedContentTypes: [.json],
                      allowsMultipleSelection: false) { result in
            if case .success(let urls) = result, let url = urls.first {
                store.loadRoutingLog(url: url)
            }
        }
        #endif
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Cortex").font(.largeTitle.bold())
            HStack(spacing: 8) {
                Chip(text: store.adapterName, tint: Theme.signal)
                if !store.capabilitySummary.isEmpty {
                    Chip(text: store.capabilitySummary, tint: Theme.trace)
                }
            }
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 10) {
                    Button("Demo Net") { store.useDemoNet() }
                    Button("Text Model") { store.useDenseTextModel() }
                    Button("MoE Model") { store.useMoEModel() }
                    Button("Import Log") { store.useRoutingLogSample() }
                }
                #if os(macOS) || os(iOS)
                HStack(spacing: 10) {
                    Button("Load Core ML…") { showingImporter = true }
                    Button("Load Log…") { showingLogImporter = true }
                }
                #endif
            }
            .font(.callout)
            .buttonStyle(.bordered)

            if !store.status.isEmpty {
                Text(store.status).font(.footnote).foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder
    private func currentInputStrip(input: CartographyInput) -> some View {
        HStack(spacing: 14) {
            if case .grid(let g) = input.payload {
                GridThumbnail(grid: g).frame(width: 72, height: 72)
            }
            VStack(alignment: .leading, spacing: 4) {
                Text("Current input").font(.caption).foregroundStyle(.secondary)
                if let text = input.display {
                    Text("“\(text)”").font(.callout).italic()
                }
                if let trace = store.currentTrace {
                    Text(trace.predicted).font(.title3.bold()).foregroundStyle(Theme.signal)
                    Text(cortexHint).font(.caption2).foregroundStyle(.secondary)
                } else {
                    Text("—").font(.title3)
                }
            }
            Spacer()
        }
        .card()
    }

    private var cortexHint: String {
        if store.canSteer { return "cells are SAE features; brightness = activation" }
        if store.hasExperts { return "cells are experts; brightness = router gate for this input" }
        return "cells brighten with activation for this input"
    }

    private var cortex: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(alignment: .top, spacing: 22) {
                ForEach(store.columns, id: \.layer) { column in
                    VStack(spacing: 8) {
                        Text(layerTitle(for: column.regions))
                            .font(.caption.bold())
                            .foregroundStyle(.secondary)
                        ForEach(column.regions) { region in
                            RegionCell(
                                region: region,
                                activation: store.displayActivation(region),
                                onPath: store.isOnPath(region),
                                selected: store.selectedRegionIDs.contains(region.id),
                                ablatable: store.isAblatable(region)
                            ) {
                                store.toggleSelection(region)
                            }
                        }
                    }
                }
            }
            .padding(.vertical, 4)
        }
    }

    private var legend: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Reading the map").font(.headline)
            legendRow(color: Theme.signal, text: "Brighter cell = stronger activation for the current input")
            legendRow(color: Theme.signal, text: "Teal ring = on this input's routing path", ring: true)
            legendRow(color: Theme.trace, text: "Amber ring = marked for ablation (tap a hidden neuron)", ring: true)
            Text("Domain labels come from attribution: each neuron is tagged with the class it most prefers.")
                .font(.caption).foregroundStyle(.secondary).padding(.top, 2)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
    }

    private func legendRow(color: Color, text: String, ring: Bool = false) -> some View {
        HStack(spacing: 10) {
            RoundedRectangle(cornerRadius: 4)
                .fill(ring ? Color.clear : color.opacity(0.6))
                .overlay(RoundedRectangle(cornerRadius: 4).stroke(color, lineWidth: ring ? 2 : 0))
                .frame(width: 18, height: 18)
            Text(text).font(.caption)
            Spacer()
        }
    }
}

/// A single cortex cell.
struct RegionCell: View {
    let region: Region
    let activation: Double
    let onPath: Bool
    let selected: Bool
    let ablatable: Bool
    let action: () -> Void

    private var ringColor: Color {
        if selected { return Theme.trace }
        if onPath { return Theme.signal }
        return .clear
    }
    private var ringWidth: CGFloat { selected ? 3 : (onPath ? 2 : 0) }

    var body: some View {
        Button(action: action) {
            ZStack {
                RoundedRectangle(cornerRadius: Theme.cellCorner).fill(Theme.heat(activation))
                Text(labelGlyph)
                    .font(.caption2.bold().monospaced())
                    .foregroundStyle(activation > 0.55 ? Color.white : .primary)
            }
            .frame(width: 50, height: 34)
            .overlay(
                RoundedRectangle(cornerRadius: Theme.cellCorner)
                    .stroke(ringColor, lineWidth: ringWidth)
            )
            .overlay(alignment: .bottomTrailing) {
                if let s = region.selectivity, s > 0.66, region.kind != .logit {
                    Circle().fill(Theme.trace).frame(width: 5, height: 5).padding(3)
                }
            }
        }
        .buttonStyle(.plain)
        .help(regionHelp)
    }

    private var labelGlyph: String {
        if region.kind == .logit { return String(region.id.dropFirst(4).prefix(3)) }
        if let l = region.label, let first = l.first, l != "(silent)" { return String(first).uppercased() }
        return "·"
    }

    private var regionHelp: String {
        var parts = [region.id]
        if let l = region.label { parts.append("prefers \(l)") }
        if let s = region.selectivity { parts.append(String(format: "selectivity %.2f", s)) }
        return parts.joined(separator: " · ")
    }
}
