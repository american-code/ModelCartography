//
//  CoreMLModelType.swift
//  Content types accepted by the "Load Core ML…" importer (macOS / iOS only).
//

#if os(macOS) || os(iOS)
import UniformTypeIdentifiers

enum CoreMLModelType {
    static let all: [UTType] = {
        ["mlmodel", "mlpackage", "mlmodelc"].compactMap { UTType(filenameExtension: $0) }
    }()
}
#endif
