// Counts how many screen refreshes actually carried a new picture on a display (ScreenCaptureKit frame status).
import Foundation
import ScreenCaptureKit
import CoreMedia

final class Out: NSObject, SCStreamOutput {
    var complete: [Double] = [], idle = 0, other = 0
    func stream(_ stream: SCStream, didOutputSampleBuffer sb: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen, let arr = CMSampleBufferGetSampleAttachmentsArray(sb, createIfNecessary: false) as? [[SCStreamFrameInfo: Any]],
              let raw = arr.first?[.status] as? Int, let st = SCFrameStatus(rawValue: raw) else { return }
        if st == .complete { complete.append(CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sb))) } else if st == .idle { idle += 1 } else { other += 1 }
    }
}
let name = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "Studio Display"
let secs = CommandLine.arguments.count > 2 ? Double(CommandLine.arguments[2])! : 6
let sem = DispatchSemaphore(value: 0)
Task {
    do {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        let ids = NSScreen.screens.reduce(into: [CGDirectDisplayID: String]()) { $0[($1.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? CGDirectDisplayID) ?? 0] = $1.localizedName }
        guard let d = content.displays.first(where: { (ids[$0.displayID] ?? "").contains(name) }) else { print("display not found"); exit(1) }
        let cfg = SCStreamConfiguration()
        cfg.width = 480; cfg.height = 270; cfg.minimumFrameInterval = CMTime(value: 1, timescale: 120); cfg.queueDepth = 8; cfg.showsCursor = false
        let out = Out()
        let stream = SCStream(filter: SCContentFilter(display: d, excludingWindows: []), configuration: cfg, delegate: nil)
        try stream.addStreamOutput(out, type: .screen, sampleHandlerQueue: DispatchQueue(label: "sc"))
        try await stream.startCapture()
        try await Task.sleep(nanoseconds: UInt64(1e9))            // settle
        out.complete.removeAll(); out.idle = 0
        try await Task.sleep(nanoseconds: UInt64(secs * 1e9))
        try await stream.stopCapture()
        let t = out.complete
        var gaps: [Double] = []; for i in 1..<max(1, t.count) { gaps.append((t[i] - t[i-1]) * 1000) }
        let one = gaps.filter { $0 < 25 }.count, two = gaps.filter { $0 >= 25 && $0 < 42 }.count, more = gaps.filter { $0 >= 42 }.count
        print(String(format: "new pictures on screen: %.1f per second | gaps: one refresh %d, two refreshes %d, three or more %d | idle %d", Double(t.count) / secs, one, two, more, out.idle))
    } catch { print("error: \(error)") }
    sem.signal()
}
sem.wait()
