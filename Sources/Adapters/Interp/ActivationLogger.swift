//
//  ActivationLogger.swift
//  Wires ActivationStreamSender (hooked into the toy transformer's residPost
//  layers) to ActivationStreamReceiver over a per-process Unix-domain socket,
//  capturing real live activations to disk every time forward() is called.
//
//  Both types live in SwiftSci/Interp but were previously unused in this app.
//  The logger owns a shared HookRegistry; InterpAdapter passes that registry
//  to every HookedGPT2.callAsFunction so the sender hooks fire automatically.
//
//  Platform guard matches InterpAdapter — MLX is macOS/iOS only.
//

#if os(macOS) || os(iOS)

import Foundation
import Interp

// MARK: - ActivationLogger

/// Owns the lifecycle of one ActivationStreamReceiver + ActivationStreamSender
/// pair for the toy transformer.  On `init`, the receiver binds its socket and
/// the sender connects so that subsequent forward passes immediately stream
/// activations; on `deinit`, the sender is detached and the receiver is stopped.
final class ActivationLogger: @unchecked Sendable {

    /// The registry InterpAdapter must pass to every HookedGPT2 call so the
    /// sender's residPost hooks fire and stream activations to the receiver.
    let registry = HookRegistry()

    private var receiver: ActivationStreamReceiver?
    private var sender:   ActivationStreamSender?

    init(nLayers: Int) {
        startStreaming(nLayers: nLayers)
    }

    deinit {
        stopStreaming()
    }

    // MARK: Output directory

    static var activationsDirectory: URL {
        let support = FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let dir = support.appendingPathComponent(
            "org.americancode.cartography/activations", isDirectory: true)
        try? FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        return dir
    }

    // MARK: Start / stop

    private func startStreaming(nLayers: Int) {
        let pid        = ProcessInfo.processInfo.processIdentifier
        let socketPath = NSTemporaryDirectory()
            + "toyxformer-\(pid).sock"

        // Receiver: bind socket, start accept loop on a background thread.
        let recvConfig = ActivationStreamReceiver.Configuration(
            socketPath:      socketPath,
            outputDirectory: Self.activationsDirectory,
            sessionTag:      "toy"
        )
        let recv = ActivationStreamReceiver(config: recvConfig)
        try? recv.start()

        // Sender: connect to the receiver's socket, attach residPost hooks.
        let sendConfig = ActivationStreamSender.Configuration(
            socketPath:       socketPath,
            maxRetries:       12,
            initialBackoffMs: 20,
            maxBackoffMs:     500
        )
        let send = ActivationStreamSender(config: sendConfig)
        send.attach(to: registry, layers: Array(0..<nLayers))

        receiver = recv
        sender   = send
    }

    func stopStreaming() {
        sender?.detach(from: registry)
        receiver?.stop()
        sender   = nil
        receiver = nil
    }
}

#endif // os(macOS) || os(iOS)
