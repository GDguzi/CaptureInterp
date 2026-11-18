import Accelerate
// CaptureInterp: capture device (UVC capture card / camera) -> Apple low-latency
// frame interpolation (X2) -> fixed-cadence presentation. No screen capture.
//
// Usage (inside the .app bundle):
//   CaptureInterp --list
//   CaptureInterp [--device <name substring>] [--screen <index>] [--offset <ms>]
//                 [--no-interp] [--audio <name substring>] [--log <path>] [--seconds <n>]
import AppKit
import AVFoundation
import CoreMedia
import CoreVideo
import VideoToolbox
import RifeMetal
import CoreImage
import Metal

// MARK: - Arguments

struct Options {
    var list = false
    var device = ""
    var screen = 0
    var screenExplicit = false   // --screen <index> was given
    var offsetMs = 0.0       // capture PTS -> interpolated frame on screen; 0 = adaptive
    var followArrival = true // hold = this frame's own delivery delay + processing time (false: fixed hold sized for the slowest recent delivery)
    var leadMs = -1.0        // hand a frame to the display layer this long before the refresh it is meant for; <0 = automatic
    var gpuPhaseMs = -1.0    // GPU mode: start each interpolation this long after a screen refresh (<0 = start at once)
    var votePick = false     // second version of the choice (switch halves by vote)
    var legacyPick = false   // old per-frame choice of which 60 fps capture frames carry a new 30 fps picture
    var multiMid = true      // GPU modes: two in-between pictures when the game held a picture for three refreshes
    var protectOnCPU = false // still-pixel protection on the CPU (the first version) instead of the GPU pass
    var protectStill = true  // GPU modes: keep pixels that are identical in both originals (subtitles, HUD) out of the motion warp
    var steady = true        // once the hold had to go up a step it stays there: every step change cost ~6 misplaced frames on the glass (--stepping: lower average delay, a blip every ~8 s)
    var marginMs = -1.0      // <0 = automatic; was 2.0:      // adaptive mode: headroom above the observed ready time
    var interpolate = true
    var dump = ""            // directory: write a few consecutive original/generated frames as PNG (diagnosis)
    var dumpAfter = 150      // frames to wait before dumping
    var rife = ""            // "fast" / "balanced": start in the GPU (RIFE) mode
    var audio = ""
    var log = NSHomeDirectory() + "/Library/Logs/CaptureInterp/stats.log"
    var seconds = 0.0
    var maxWidth = 0         // 0 = highest available
    var file = ""            // simulate a device: play this video file in real time at ~30 fps
    var tiles = 1            // 4 = 2x2 -> ~2K (2496x1404); 9 = 3x3 -> ~3K (3200x1800)
    var grid: Grid? = nil    // --grid cols rows tileW tileH overlapX overlapY (experiments)
    var origNative = false   // show originals at capture size instead of the working size
    var screenName = ""      // pick the output screen by name, e.g. "Studio Display"
    var vsync = true         // place frames on the display's refresh grid
    var audioGainDb = 20.0   // extra loudness in dB (capture card HDMI sound arrives very quiet)
    var autoPass = false     // 60 fps capture: show a 60 fps game as is instead of interpolating (off: keys 2/3 always interpolate)
    var fps = 30.0           // capture frame rate to lock (60 = take the source as is, no interpolation needed)
    var audioVolume = 1.0    // 0 = process audio silently (testing)
}

func parseOptions() -> Options {
    var o = Options()
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let a = it.next() {
        switch a {
        case "--list": o.list = true
        case "--device": o.device = it.next() ?? ""
        case "--screen": o.screen = Int(it.next() ?? "") ?? 0; o.screenExplicit = true
        case "--offset": o.offsetMs = Double(it.next() ?? "") ?? o.offsetMs
        case "--steady-offset": o.followArrival = false
        case "--lead-ms": o.leadMs = Double(it.next() ?? "") ?? o.leadMs
        case "--gpu-phase-ms": o.gpuPhaseMs = Double(it.next() ?? "") ?? o.gpuPhaseMs
        case "--steady": o.steady = true
        case "--no-protect": o.protectStill = false
        case "--protect-cpu": o.protectOnCPU = true
        case "--single-mid": o.multiMid = false
        case "--legacy-pick": o.legacyPick = true
        case "--vote-pick": o.votePick = true
        case "--stepping": o.steady = false
        case "--margin": o.marginMs = Double(it.next() ?? "") ?? o.marginMs
        case "--no-interp": o.interpolate = false
        case "--dump": o.dump = it.next() ?? ""
        case "--dump-after": o.dumpAfter = Int(it.next() ?? "") ?? 150
        case "--rife": o.rife = it.next() ?? "fast"
        case "--audio": o.audio = it.next() ?? ""
        case "--log": o.log = it.next() ?? o.log
        case "--seconds": o.seconds = Double(it.next() ?? "") ?? 0
        case "--max-width": o.maxWidth = Int(it.next() ?? "") ?? 0
        case "--file": o.file = it.next() ?? ""
        case "--tiles": o.tiles = Int(it.next() ?? "") ?? 1
        case "--orig-native": o.origNative = true
        case "--screen-name": o.screenName = it.next() ?? ""
        case "--no-vsync": o.vsync = false
        case "--audio-volume": o.audioVolume = Double(it.next() ?? "") ?? 1.0
        case "--auto-pass": o.autoPass = true
        case "--fps": o.fps = Double(it.next() ?? "") ?? 30
        case "--audio-gain": o.audioGainDb = Double(it.next() ?? "") ?? 0
        case "--grid":
            let v = (0..<6).map { _ in Int(it.next() ?? "") ?? 0 }
            if v.allSatisfy({ $0 > 0 }) { o.grid = Grid(cols: v[0], rows: v[1], tileW: v[2], tileH: v[3], overlapX: v[4], overlapY: v[5]); o.tiles = v[0] * v[1] }
        default: break
        }
    }
    return o
}

let opts = parseOptions()

final class Log {
    static let shared = Log()
    private var handle: FileHandle?
    func open(_ path: String) {
        try? FileManager.default.createDirectory(atPath: (path as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: path, contents: nil)
        handle = FileHandle(forWritingAtPath: path)
        atexit { Log.shared.flush() }
    }
    // Writing can stall for a second or more (seen with the log on a slow volume), so it must
    // never happen on the capture or processing threads: lines are handed to a background queue.
    private let writer = DispatchQueue(label: "log.writer", qos: .utility)
    func line(_ s: String) {
        writer.async {
            print(s)
            self.handle?.write((s + "\n").data(using: .utf8)!)
        }
    }
    /// Blocks until everything logged so far is on disk (used before exiting).
    func flush() { writer.sync {} }
}

func audioDevices() -> [AVCaptureDevice] {
    AVCaptureDevice.DiscoverySession(deviceTypes: [.microphone, .external], mediaType: .audio, position: .unspecified).devices
}

/// Capture cards show up as external devices; prefer them over built-in cameras and phones.
func preferredVideoDevice(_ name: String) -> AVCaptureDevice? {
    let devs = videoDevices()
    if !name.isEmpty { return devs.first { $0.localizedName.localizedCaseInsensitiveContains(name) } }
    func isPhoneOrVirtual(_ d: AVCaptureDevice) -> Bool {
        ["OBS", "iPhone", "iPad", "Virtual"].contains { d.localizedName.localizedCaseInsensitiveContains($0) }
    }
    return devs.first { $0.deviceType == .external && !isPhoneOrVirtual($0) }
        ?? devs.first { $0.deviceType == .builtInWideAngleCamera } ?? devs.first
}

func videoDevices() -> [AVCaptureDevice] {
    AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera, .external, .continuityCamera],
                                     mediaType: .video, position: .unspecified).devices
}

func fourCC(_ v: OSType) -> String {
    let b = [UInt8((v >> 24) & 255), UInt8((v >> 16) & 255), UInt8((v >> 8) & 255), UInt8(v & 255)]
    return String(bytes: b, encoding: .ascii) ?? "\(v)"
}

if !opts.list { Log.shared.open(opts.log) }

if opts.list {
    for d in videoDevices() {
        Log.shared.line("VIDEO DEVICE \(d.localizedName)\(d.deviceType == .external ? "  [external]" : "")")
        var seen = Set<String>()
        for f in d.formats {
            let dim = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
            let sub = fourCC(CMFormatDescriptionGetMediaSubType(f.formatDescription))
            let fps = f.videoSupportedFrameRateRanges.map { Int($0.maxFrameRate.rounded()) }.max() ?? 0
            let key = "  \(dim.width)x\(dim.height) \(sub) up to \(fps) fps"
            if seen.insert(key).inserted { Log.shared.line(key) }
        }
    }
    for d in audioDevices() { Log.shared.line("AUDIO DEVICE \(d.localizedName)") }
    Log.shared.flush(); exit(0)
}

// MARK: - Helpers

let hostClock = CMClockGetHostTimeClock()
func hostNow() -> CMTime { CMClockGetTime(hostClock) }
func ms(_ t: CMTime) -> Double { CMTimeGetSeconds(t) * 1000.0 }

func makeSample(_ pb: CVPixelBuffer, pts: CMTime) -> CMSampleBuffer? {
    var fmt: CMVideoFormatDescription?
    guard CMVideoFormatDescriptionCreateForImageBuffer(allocator: nil, imageBuffer: pb, formatDescriptionOut: &fmt) == noErr,
          let fmt else { return nil }
    var timing = CMSampleTimingInfo(duration: .invalid, presentationTimeStamp: pts, decodeTimeStamp: .invalid)
    var sb: CMSampleBuffer?
    CMSampleBufferCreateReadyWithImageBuffer(allocator: nil, imageBuffer: pb, formatDescription: fmt,
                                             sampleTiming: &timing, sampleBufferOut: &sb)
    return sb
}

/// Generated frames must carry the same colour tags as the originals, or the display
/// treats alternate frames differently and the picture flickers.
func copyColor(from src: CVPixelBuffer, to dst: CVPixelBuffer) {
    if let att = CVBufferCopyAttachments(src, .shouldPropagate) { CVBufferSetAttachments(dst, att, .shouldPropagate) }
}

/// Mean of each byte channel of a BGRA buffer (sparse sample): B, G, R, A.
func meanBGRA(_ pb: CVPixelBuffer) -> [Double] {
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
    guard let b = CVPixelBufferGetBaseAddress(pb)?.assumingMemoryBound(to: UInt8.self) else { return [-1, -1, -1, -1] }
    let bpr = CVPixelBufferGetBytesPerRow(pb), w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
    var s = [0, 0, 0, 0], n = 0
    for y in stride(from: 0, to: h, by: 16) { for x in stride(from: 0, to: w, by: 16) { for c in 0..<4 { s[c] += Int(b[y * bpr + x * 4 + c]) }; n += 1 } }
    return s.map { Double($0) / Double(max(1, n)) }
}

/// Sparse comparison of two same-size BGRA buffers (green channel): mean |a-b|, and each one's mean horizontal gradient.
func compareBGRA(_ a: CVPixelBuffer, _ b: CVPixelBuffer) -> (diff: Double, gradA: Double, gradB: Double) {
    CVPixelBufferLockBaseAddress(a, .readOnly); CVPixelBufferLockBaseAddress(b, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(a, .readOnly); CVPixelBufferUnlockBaseAddress(b, .readOnly) }
    guard let pa = CVPixelBufferGetBaseAddress(a)?.assumingMemoryBound(to: UInt8.self), let pb = CVPixelBufferGetBaseAddress(b)?.assumingMemoryBound(to: UInt8.self) else { return (-1, -1, -1) }
    let ra = CVPixelBufferGetBytesPerRow(a), rb = CVPixelBufferGetBytesPerRow(b), w = CVPixelBufferGetWidth(a), h = CVPixelBufferGetHeight(a)
    var d = 0, ga = 0, gb = 0, n = 0
    for y in stride(from: 8, to: h - 8, by: 9) { for x in stride(from: 8, to: w - 8, by: 7) {
        let va = Int(pa[y * ra + x * 4 + 1]), vb = Int(pb[y * rb + x * 4 + 1])
        d += abs(va - vb); ga += abs(va - Int(pa[y * ra + (x + 1) * 4 + 1])); gb += abs(vb - Int(pb[y * rb + (x + 1) * 4 + 1])); n += 1 } }
    return (Double(d) / Double(n), Double(ga) / Double(n), Double(gb) / Double(n))
}

let dumpContext = CIContext()
let dumpQueue = DispatchQueue(label: "dump", qos: .utility)
func savePNG(_ pb: CVPixelBuffer, _ path: String) {
    let img = CIImage(cvPixelBuffer: pb)
    try? dumpContext.writePNGRepresentation(of: img, to: URL(fileURLWithPath: path), format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
}

func meanLuma(_ pb: CVPixelBuffer) -> Double {
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
    guard let b = CVPixelBufferGetBaseAddressOfPlane(pb, 0)?.assumingMemoryBound(to: UInt8.self) else { return -1 }
    let bpr = CVPixelBufferGetBytesPerRowOfPlane(pb, 0), w = CVPixelBufferGetWidthOfPlane(pb, 0), h = CVPixelBufferGetHeightOfPlane(pb, 0)
    var sum = 0, n = 0
    for y in stride(from: 0, to: h, by: 16) { for x in stride(from: 0, to: w, by: 16) { sum += Int(b[y * bpr + x]); n += 1 } }
    return Double(sum) / Double(max(1, n))
}

/// Reuses IOSurface-backed buffers instead of allocating new ones for every frame.
final class BufferPool {
    private var pool: CVPixelBufferPool?
    init(width: Int, height: Int, format: OSType, attrs: [String: Any]) {
        var a = attrs
        a[kCVPixelBufferWidthKey as String] = width
        a[kCVPixelBufferHeightKey as String] = height
        a[kCVPixelBufferPixelFormatTypeKey as String] = format
        a[kCVPixelBufferIOSurfacePropertiesKey as String] = [String: Any]()
        CVPixelBufferPoolCreate(nil, nil, a as CFDictionary, &pool)
    }
    func get() -> CVPixelBuffer? {
        guard let pool else { return nil }
        var pb: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pb)
        return pb
    }
}

func pixelFormat(of attrs: [String: Any]) -> OSType {
    let raw = attrs[kCVPixelBufferPixelFormatTypeKey as String]
    return (raw as? OSType) ?? (raw as? [OSType])?.first ?? (raw as? NSNumber)?.uint32Value
        ?? kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
}

func percentile(_ v: [Double], _ p: Double) -> Double {
    guard !v.isEmpty else { return -1 }
    let s = v.sorted()
    return s[min(s.count - 1, Int((Double(s.count - 1) * p).rounded(.up)))]
}

// MARK: - Interpolator (Apple low-latency frame interpolation at <= 1280x720)

final class Interpolator {
    let width: Int, height: Int
    private let config: VTLowLatencyFrameInterpolationConfiguration
    private let processor = VTFrameProcessor()
    private var transfer: VTPixelTransferSession?
    private let srcAttrs: [String: Any]
    private let dstAttrs: [String: Any]

    init?(sourceWidth: Int, sourceHeight: Int) {
        // macOS 26 accepts 16:9 up to 1280x720; keep inside that box, even dimensions.
        let scale = min(1.0, min(1280.0 / Double(sourceWidth), 720.0 / Double(sourceHeight)))
        width = max(2, Int(Double(sourceWidth) * scale) & ~1)
        height = max(2, Int(Double(sourceHeight) * scale) & ~1)
        guard VTLowLatencyFrameInterpolationConfiguration.isSupported,
              let c = VTLowLatencyFrameInterpolationConfiguration(frameWidth: width, frameHeight: height,
                                                                   numberOfInterpolatedFrames: 1) else { return nil }
        config = c
        srcAttrs = c.sourcePixelBufferAttributes
        dstAttrs = c.destinationPixelBufferAttributes
        do { try processor.startSession(configuration: c) } catch {
            Log.shared.line("interp session failed: \(error.localizedDescription)")
            return nil
        }
        VTPixelTransferSessionCreate(allocator: nil, pixelTransferSessionOut: &transfer)
    }

    private lazy var srcPool = BufferPool(width: width, height: height, format: pixelFormat(of: srcAttrs), attrs: srcAttrs)
    private lazy var dstPool = BufferPool(width: width, height: height, format: pixelFormat(of: dstAttrs), attrs: dstAttrs)

    /// Scale a captured frame into the interpolation input size.
    func prepare(_ source: CVPixelBuffer) -> CVPixelBuffer? {
        guard let transfer, let dst = srcPool.get() else { return nil }
        guard VTPixelTransferSessionTransferImage(transfer, from: source, to: dst) == noErr else { return nil }
        copyColor(from: source, to: dst)
        return dst
    }

    /// Blocking: returns the midpoint frame between previous and current.
    func midpoint(previous: CVPixelBuffer, current: CVPixelBuffer, previousPTS: CMTime, currentPTS: CMTime) -> CVPixelBuffer? {
        guard let dst = dstPool.get(),
              let fa = VTFrameProcessorFrame(buffer: previous, presentationTimeStamp: previousPTS),
              let fb = VTFrameProcessorFrame(buffer: current, presentationTimeStamp: currentPTS) else { return nil }
        let mid = CMTimeAdd(previousPTS, CMTimeMultiplyByFloat64(CMTimeSubtract(currentPTS, previousPTS), multiplier: 0.5))
        guard let fd = VTFrameProcessorFrame(buffer: dst, presentationTimeStamp: mid),
              let params = VTLowLatencyFrameInterpolationParameters(sourceFrame: fb, previousFrame: fa,
                                                                    interpolationPhase: [0.5], destinationFrames: [fd]) else { return nil }
        let sem = DispatchSemaphore(value: 0)
        var ok = false
        processor.process(parameters: params) { _, error in
            ok = error == nil
            if let error { Log.shared.line("interp error: \(error.localizedDescription)") }
            sem.signal()
        }
        sem.wait()
        copyColor(from: current, to: dst)
        return ok ? dst : nil
    }
}

// MARK: - Tiled interpolator (grid of overlapping tiles, cross-faded)

struct Grid {
    var cols = 2, rows = 2, tileW = 1280, tileH = 720, overlapX = 64, overlapY = 36
    var workW: Int { cols * tileW - (cols - 1) * overlapX }
    var workH: Int { rows * tileH - (rows - 1) * overlapY }
    static let k2 = Grid()                                                              // 2496 x 1404
    static let k3 = Grid(cols: 3, rows: 3, tileW: 1120, tileH: 632, overlapX: 80, overlapY: 48)   // 3200 x 1800
}

/// The system engine accepts about 1280x720 per pass on macOS 26. Overlapping tiles
/// cover a larger working frame; overlaps are cross-faded to hide seams.
final class TiledInterpolator {
    let g: Grid
    private var configs: [VTLowLatencyFrameInterpolationConfiguration] = []
    private var processors: [VTFrameProcessor] = []
    private var transfer: VTPixelTransferSession?
    private var format: OSType = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
    private var srcAttrs: [String: Any] = [:]
    private var dstAttrs: [String: Any] = [:]
    private var rowA: [UInt8], rowB: [UInt8]
    /// Mean absolute luma difference between neighbouring tiles inside the overlaps (0-255); for comparison,
    /// `lastTextureDiff` is the same measure between adjacent pixel columns of one tile (natural detail level).
    private(set) var lastSeamDiff = 0.0, lastTextureDiff = 0.0

    init?(_ grid: Grid) {
        g = grid
        rowA = [UInt8](repeating: 0, count: max(2, grid.workW))
        rowB = [UInt8](repeating: 0, count: max(2, grid.workW))
        // NV12 chroma is interleaved at half size: everything must be even, and overlaps must leave a body.
        guard g.cols >= 1, g.rows >= 1, g.tileW % 2 == 0, g.tileH % 2 == 0, g.overlapX % 4 == 0, g.overlapY % 4 == 0,
              g.overlapX >= 4, g.overlapY >= 4, g.overlapX * 2 <= g.tileW, g.overlapY * 2 <= g.tileH else {
            Log.shared.line("invalid grid: sizes must be even, overlaps multiples of 4 and at most half a tile"); return nil
        }
        guard VTLowLatencyFrameInterpolationConfiguration.isSupported else { return nil }
        for _ in 0..<(g.cols * g.rows) {
            guard let c = VTLowLatencyFrameInterpolationConfiguration(frameWidth: g.tileW, frameHeight: g.tileH,
                                                                      numberOfInterpolatedFrames: 1) else { return nil }
            let p = VTFrameProcessor()
            do { try p.startSession(configuration: c) } catch {
                Log.shared.line("tile session failed: \(error.localizedDescription)")
                for started in processors { started.endSession() }
                return nil
            }
            configs.append(c); processors.append(p)
        }
        srcAttrs = configs[0].sourcePixelBufferAttributes
        dstAttrs = configs[0].destinationPixelBufferAttributes
        let raw = srcAttrs[kCVPixelBufferPixelFormatTypeKey as String]
        format = (raw as? OSType) ?? (raw as? [OSType])?.first ?? (raw as? NSNumber)?.uint32Value ?? format
        guard format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange || format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange else {
            Log.shared.line("tile mode needs 8-bit NV12, engine wants \(fourCC(format))"); return nil
        }
        VTPixelTransferSessionCreate(allocator: nil, pixelTransferSessionOut: &transfer)
    }

    private lazy var workPool = BufferPool(width: g.workW, height: g.workH, format: format, attrs: [:])
    private lazy var tileSrcPool = BufferPool(width: g.tileW, height: g.tileH, format: format, attrs: srcAttrs)
    private lazy var tileDstPool = BufferPool(width: g.tileW, height: g.tileH, format: format, attrs: dstAttrs)

    /// Where the time of the last frame went, ms: scale, cut into tiles, engine, stitch.
    private(set) var lastScaleMs = 0.0, lastCropMs = 0.0, lastEngineMs = 0.0, lastStitchMs = 0.0

    /// Scale the captured frame to the working size.
    func work(from source: CVPixelBuffer) -> CVPixelBuffer? {
        let t0 = CACurrentMediaTime(); defer { lastScaleMs = (CACurrentMediaTime() - t0) * 1000 }
        guard let transfer, let dst = workPool.get() else { return nil }
        guard VTPixelTransferSessionTransferImage(transfer, from: source, to: dst) == noErr else { return nil }
        copyColor(from: source, to: dst)
        return dst
    }

    /// Cut the working frame into overlapping tiles, row-major.
    func crop(_ work: CVPixelBuffer) -> [CVPixelBuffer]? {
        let t0 = CACurrentMediaTime(); defer { lastCropMs = (CACurrentMediaTime() - t0) * 1000 }
        var tiles: [CVPixelBuffer] = []
        CVPixelBufferLockBaseAddress(work, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(work, .readOnly) }
        for _ in 0..<(g.rows * g.cols) { guard let t = tileSrcPool.get() else { return nil }; tiles.append(t) }
        let all = tiles, g = self.g
        // Plain memory copies: one tile per core.
        DispatchQueue.concurrentPerform(iterations: all.count) { i in
            do {
                let ty = i / g.cols, tx = i % g.cols, t = all[i]
                CVPixelBufferLockBaseAddress(t, [])
                for plane in 0..<2 {
                    let div = plane == 0 ? 1 : 2
                    guard let sb = CVPixelBufferGetBaseAddressOfPlane(work, plane), let db = CVPixelBufferGetBaseAddressOfPlane(t, plane) else { continue }
                    let sbpr = CVPixelBufferGetBytesPerRowOfPlane(work, plane), dbpr = CVPixelBufferGetBytesPerRowOfPlane(t, plane)
                    let x0 = tx * (g.tileW - g.overlapX)             // bytes: luma 1 per px, chroma 2 per half px
                    let y0 = ty * (g.tileH - g.overlapY) / div
                    for y in 0..<(g.tileH / div) { memcpy(db + y * dbpr, sb + (y0 + y) * sbpr + x0, g.tileW) }
                }
                CVPixelBufferUnlockBaseAddress(t, [])
            }
        }
        return tiles
    }

    /// Blocking: interpolate all tiles at once and stitch the midpoint frame.
    func midpoint(previous: [CVPixelBuffer], current: [CVPixelBuffer], previousPTS: CMTime, currentPTS: CMTime, like: CVPixelBuffer) -> CVPixelBuffer? {
        let mid = CMTimeAdd(previousPTS, CMTimeMultiplyByFloat64(CMTimeSubtract(currentPTS, previousPTS), multiplier: 0.5))
        var outs: [CVPixelBuffer] = []
        let group = DispatchGroup()
        let lock = NSLock()
        var failed = false
        for i in 0..<processors.count {
            guard let dst = tileDstPool.get(),
                  let fa = VTFrameProcessorFrame(buffer: previous[i], presentationTimeStamp: previousPTS),
                  let fb = VTFrameProcessorFrame(buffer: current[i], presentationTimeStamp: currentPTS),
                  let fd = VTFrameProcessorFrame(buffer: dst, presentationTimeStamp: mid),
                  let params = VTLowLatencyFrameInterpolationParameters(sourceFrame: fb, previousFrame: fa,
                                                                        interpolationPhase: [0.5], destinationFrames: [fd]) else { return nil }
            outs.append(dst)
            group.enter()
            processors[i].process(parameters: params) { _, error in
                if let error { lock.lock(); failed = true; lock.unlock(); Log.shared.line("tile error: \(error.localizedDescription)") }
                group.leave()
            }
        }
        let tEngine = CACurrentMediaTime()
        group.wait()
        let tStitch = CACurrentMediaTime()
        lastEngineMs = (tStitch - tEngine) * 1000
        guard !failed, let out = stitch(outs) else { return nil }
        lastStitchMs = (CACurrentMediaTime() - tStitch) * 1000
        copyColor(from: like, to: out)
        return out
    }

    /// Reassemble the tiles; cross-fade linearly across the overlaps.
    private func stitch(_ t: [CVPixelBuffer]) -> CVPixelBuffer? {
        guard let out = workPool.get() else { return nil }
        for b in t { CVPixelBufferLockBaseAddress(b, .readOnly) }
        CVPixelBufferLockBaseAddress(out, [])
        defer {
            CVPixelBufferUnlockBaseAddress(out, [])
            for b in t { CVPixelBufferUnlockBaseAddress(b, .readOnly) }
        }
        let cols = g.cols, tileRows = g.rows
        for plane in 0..<2 {
            let div = plane == 0 ? 1 : 2
            let step = plane == 0 ? 1 : 2                      // bytes per horizontal sample
            let rows = g.tileH / div, ovY = g.overlapY / div, outRows = g.workH / div, pitchY = rows - ovY
            let tw = g.tileW, ovBytes = g.overlapX, pitchX = tw - ovBytes, ovSamples = ovBytes / step
            guard let ob = CVPixelBufferGetBaseAddressOfPlane(out, plane)?.assumingMemoryBound(to: UInt8.self) else { return nil }
            let obpr = CVPixelBufferGetBytesPerRowOfPlane(out, plane)
            var base0: [UnsafeMutablePointer<UInt8>] = [], bpr0: [Int] = []
            for b in t {
                guard let p = CVPixelBufferGetBaseAddressOfPlane(b, plane)?.assumingMemoryBound(to: UInt8.self) else { return nil }
                base0.append(p); bpr0.append(CVPixelBufferGetBytesPerRowOfPlane(b, plane))
            }
            let base = base0, bpr = bpr0
            if plane == 0 {   // how much do neighbouring tiles disagree where they overlap?
                var seam = 0, tex = 0, n = 0
                if cols > 1 {
                    for y in stride(from: 0, to: rows, by: 6) {
                        let l = base[0] + y * bpr[0], r = base[1] + y * bpr[1]
                        for i in stride(from: 0, to: ovBytes, by: 2) {
                            seam += abs(Int(l[pitchX + i]) - Int(r[i])); tex += abs(Int(l[pitchX + i]) - Int(l[pitchX + i - 1])); n += 1
                        }
                    }
                }
                if tileRows > 1 {
                    for x in stride(from: 0, to: tw, by: 6) {
                        for j in stride(from: 0, to: ovY, by: 2) {
                            let a = Int(base[0][(pitchY + j) * bpr[0] + x])
                            seam += abs(a - Int(base[cols][j * bpr[cols] + x])); tex += abs(a - Int(base[0][(pitchY + j - 1) * bpr[0] + x])); n += 1
                        }
                    }
                }
                lastSeamDiff = Double(seam) / Double(max(1, n)); lastTextureDiff = Double(tex) / Double(max(1, n))
            }
            // One full-width output row from tile row `tr`, local row `ly`.
            func hrow(_ dst: UnsafeMutablePointer<UInt8>, _ tr: Int, _ ly: Int) {
                for c in 0..<cols {
                    let src = base[tr * cols + c] + ly * bpr[tr * cols + c]
                    let from = c == 0 ? 0 : ovBytes, to = c == cols - 1 ? tw : pitchX
                    memcpy(dst + c * pitchX + from, src + from, to - from)
                    if c < cols - 1 {
                        let r = base[tr * cols + c + 1] + ly * bpr[tr * cols + c + 1]
                        for i in 0..<ovSamples {
                            let wr = 2 * i + 1, wl = 2 * ovSamples - wr
                            for k in (i * step)..<(i * step + step) {
                                dst[(c + 1) * pitchX + k] = UInt8((Int(src[pitchX + k]) * wl + Int(r[k]) * wr + ovSamples) / (2 * ovSamples))
                            }
                        }
                    }
                }
            }
            // Rows are independent: split the frame into bands, one per core, each with its own scratch rows.
            let bands = 8, workW = g.workW
            DispatchQueue.concurrentPerform(iterations: bands) { bi in
                let ra = UnsafeMutablePointer<UInt8>.allocate(capacity: max(2, workW)), rb = UnsafeMutablePointer<UInt8>.allocate(capacity: max(2, workW))
                defer { ra.deallocate(); rb.deallocate() }
                do {
                    for y in (outRows * bi / bands)..<(outRows * (bi + 1) / bands) {
                        let dst = ob + y * obpr
                        let tr = min(tileRows - 1, y / pitchY)        // lower of the tile rows covering y
                        let ly = y - tr * pitchY
                        if tr > 0 && ly < ovY {                        // also covered by the row above
                            hrow(ra, tr - 1, ly + pitchY)
                            hrow(rb, tr, ly)
                            let wb = 2 * ly + 1, wt = 2 * ovY - wb
                            for x in 0..<workW { dst[x] = UInt8((Int(ra[x]) * wt + Int(rb[x]) * wb + ovY) / (2 * ovY)) }
                        } else {
                            hrow(dst, tr, ly)
                        }
                    }
                }
            }
        }
        return out
    }
}

// MARK: - Audio (delayed to match the picture)

/// Plays the capture device's sound later by the same amount the picture is held,
/// so sound and picture stay together.
final class AudioDelay: NSObject, AVCaptureAudioDataOutputSampleBufferDelegate {
    // AVAudioEngine player: buffers are queued behind a block of silence as long as the picture's hold.
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let lock = NSLock()
    private var wantedDelayMs = 0.0, delayMs = 0.0
    private var format: AVAudioFormat?
    private var scheduled: Int64 = 0            // frames handed to the player since the last (re)start
    private var needsRestart = true
    private var baseline: Double?
    private var lastCheck = hostNow()
    private(set) var buffers = 0, resyncs = 0, failures = 0
    weak var session: AVCaptureSession?
    var dbg = ""

    override init() {
        super.init()
        engine.attach(player)
        NotificationCenter.default.addObserver(forName: .AVAudioEngineConfigurationChange, object: engine, queue: nil) { [weak self] _ in
            self?.lock.lock(); self?.needsRestart = true; self?.lock.unlock()      // output device changed
        }
    }

    func setDelay(_ ms: Double) { lock.lock(); wantedDelayMs = ms; lock.unlock() }
    func currentDelay() -> Double { lock.lock(); defer { lock.unlock() }; return delayMs }
    func takeCounters() -> (Int, Int, Int) {
        lock.lock(); defer { lock.unlock() }
        let r = (buffers, resyncs, failures); buffers = 0; resyncs = 0; failures = 0; return r
    }

    private func restart(_ fmt: AVAudioFormat, delay: Double) -> Bool {
        player.stop()
        if format != fmt || !engine.isRunning {
            engine.stop()
            engine.connect(player, to: engine.mainMixerNode, format: fmt)
            do { try engine.start() } catch { dbg = "engine start failed: \(error.localizedDescription)"; return false }
            format = fmt
        }
        player.volume = Float(opts.audioVolume)
        scheduled = 0
        baseline = nil; lastCheck = hostNow()
        let n = AVAudioFrameCount(max(1, fmt.sampleRate * min(500, max(0, delay.isFinite ? delay : 0)) / 1000))   // never trust the delay blindly: an absurd value overflowed here
        if let silence = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: n) {
            silence.frameLength = n
            let abl = UnsafeMutableAudioBufferListPointer(silence.mutableAudioBufferList)
            for b in abl { if let d = b.mData { memset(d, 0, Int(b.mDataByteSize)) } }
            player.scheduleBuffer(silence); scheduled += Int64(n)
        }
        player.play()
        return true
    }

    private func depthMs(_ fmt: AVAudioFormat) -> Double? {
        guard let rt = player.lastRenderTime, let pt = player.playerTime(forNodeTime: rt) else { return nil }
        return Double(scheduled - pt.sampleTime) / fmt.sampleRate * 1000
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let fd = CMSampleBufferGetFormatDescription(sampleBuffer) else { return }
        let fmt = AVAudioFormat(cmAudioFormatDescription: fd)
        let n = AVAudioFrameCount(CMSampleBufferGetNumSamples(sampleBuffer))
        guard n > 0, let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: n) else { return }
        buf.frameLength = n
        guard CMSampleBufferCopyPCMDataIntoAudioBufferList(sampleBuffer, at: 0, frameCount: Int32(n), into: buf.mutableAudioBufferList) == noErr else {
            lock.lock(); failures += 1; lock.unlock(); return
        }
        // The VC-009Pro hands over HDMI sound far below normal level: amplify, with a hard limit at full scale.
        var peak: Float = 0
        if opts.audioGainDb != 0, fmt.commonFormat == .pcmFormatFloat32 {
            var g = Float(pow(10.0, opts.audioGainDb / 20.0)), lo: Float = -1, hi: Float = 1
            for b in UnsafeMutableAudioBufferListPointer(buf.mutableAudioBufferList) {
                guard let d = b.mData else { continue }
                let f = d.assumingMemoryBound(to: Float.self), c = vDSP_Length(Int(b.mDataByteSize) / 4)
                vDSP_vsmul(f, 1, &g, f, 1, c)
                vDSP_vclip(f, 1, &lo, &hi, f, 1, c)
                var p: Float = 0; vDSP_maxmgv(f, 1, &p, c); peak = max(peak, p)
            }
        }
        lock.lock()
        var restartNow = needsRestart || format != fmt
        // Follow the picture's hold, but only in steps: every change is a small audible skip.
        if abs(wantedDelayMs - delayMs) > 12 { delayMs = wantedDelayMs; restartNow = true }
        let delay = delayMs
        needsRestart = false
        lock.unlock()
        // The audio device's clock drifts against the output's; re-queue when the backlog has wandered.
        let now = hostNow()
        if !restartNow, CMTimeGetSeconds(CMTimeSubtract(now, lastCheck)) > 2 {
            lastCheck = now
            // The measured backlog carries a fixed offset (output latency), so compare against the
            // value seen shortly after the last restart rather than against the nominal delay.
            if let d = depthMs(fmt) {
                // Measured backlog sits about 50 ms above the nominal delay when healthy.
                if d < delay + 10 || d > delay + 95 { restartNow = true; lock.lock(); resyncs += 1; lock.unlock() }
            }
        }
        if restartNow, !restart(fmt, delay: delay) {
            lock.lock(); failures += 1; needsRestart = true; lock.unlock(); return
        }
        player.scheduleBuffer(buf); scheduled += Int64(n)
        lock.lock(); buffers += 1; lock.unlock()
        dbg = "queued \(Int(depthMs(fmt) ?? -1)) ms, engine \(engine.isRunning), playing \(player.isPlaying), peak \(String(format: "%.2f", peak)), \(Int(fmt.sampleRate)) Hz x\(fmt.channelCount) \(fmt.commonFormat == .pcmFormatFloat32 ? "f32" : "other") \(fmt.isInterleaved ? "interleaved" : "planar")"
    }
}

// MARK: - Modes

enum Mode: Int {
    case pass = 1      // show frames as they arrive
    case single = 2    // 30 -> 60, generated frames computed at 720p
    case tiled = 3     // 30 -> 60, 2x2 tiles, everything at ~2K
    case rifeFast = 4      // 30 -> 60 on the GPU, whole frame at capture size, motion estimated at 1/4 size
    case rifeBalanced = 5  // same, motion estimated at 1/2 size
    var rifeTier: RifeQualityTier? { self == .rifeFast ? .fast : (self == .rifeBalanced ? .balanced : nil) }
    var title: String {
        switch self {
        case .pass: return "[1] 直通·不补帧"
        case .single: return "[2] 补帧·最低延迟（补的帧偏软）"
        case .tiled: return "[3] 补帧·分块算"
        case .rifeFast: return "[4] 补帧·显卡·延迟低"
        case .rifeBalanced: return "[5] 补帧·显卡·画质好"
        }
    }
}

// MARK: - Still-pixel protection on the GPU

/// Same rule as RifeEngine.protectStill, as one Metal compute pass over the frame (well under a millisecond at 4K
/// instead of 2-8 ms on the CPU): where both originals agree over a short horizontal run, the generated frame
/// takes the current original's pixel.
final class StillProtector {
    private let device: MTLDevice, queue: MTLCommandQueue, pipeline: MTLComputePipelineState
    private var cache: CVMetalTextureCache?
    init?() {
        guard let d = MTLCreateSystemDefaultDevice(), let q = d.makeCommandQueue() else { return nil }
        let src = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void protectStill(texture2d<float, access::read> prev [[texture(0)]],
                                 texture2d<float, access::read> cur [[texture(1)]],
                                 texture2d<float, access::write> mid [[texture(2)]],
                                 uint2 gid [[thread_position_in_grid]]) {
            int w = cur.get_width(), h = cur.get_height();
            if (int(gid.x) >= w || int(gid.y) >= h) return;
            const float tol = 3.5 / 255.0;
            bool s[5];
            for (int k = 0; k < 5; k++) {
                uint2 p = uint2(uint(clamp(int(gid.x) + k - 2, 0, w - 1)), gid.y);
                float3 d = abs(prev.read(p).rgb - cur.read(p).rgb);
                s[k] = max(d.r, max(d.g, d.b)) <= tol;
            }
            if ((s[0] && s[1] && s[2] && s[3]) || (s[1] && s[2] && s[3] && s[4])) mid.write(cur.read(gid), gid);
        }
        """
        guard let lib = try? d.makeLibrary(source: src, options: nil), let f = lib.makeFunction(name: "protectStill"),
              let p = try? d.makeComputePipelineState(function: f) else { return nil }
        device = d; queue = q; pipeline = p
        CVMetalTextureCacheCreate(nil, nil, d, nil, &cache)
    }
    private func texture(_ pb: CVPixelBuffer) -> (CVMetalTexture, MTLTexture)? {
        guard let cache else { return nil }
        var t: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(nil, cache, pb, nil, .bgra8Unorm, CVPixelBufferGetWidth(pb), CVPixelBufferGetHeight(pb), 0, &t)
        guard let t, let m = CVMetalTextureGetTexture(t) else { return nil }
        return (t, m)
    }
    func apply(previous: CVPixelBuffer, current: CVPixelBuffer, mid: CVPixelBuffer) -> Bool {
        guard CVPixelBufferGetWidth(previous) == CVPixelBufferGetWidth(mid), CVPixelBufferGetHeight(previous) == CVPixelBufferGetHeight(mid),
              CVPixelBufferGetWidth(current) == CVPixelBufferGetWidth(mid), CVPixelBufferGetHeight(current) == CVPixelBufferGetHeight(mid),
              let a = texture(previous), let b = texture(current), let m = texture(mid),
              let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return false }
        enc.setComputePipelineState(pipeline)
        enc.setTexture(a.1, index: 0); enc.setTexture(b.1, index: 1); enc.setTexture(m.1, index: 2)
        let tw = pipeline.threadExecutionWidth, th = max(1, pipeline.maxTotalThreadsPerThreadgroup / tw)
        enc.dispatchThreadgroups(MTLSize(width: (m.1.width + tw - 1) / tw, height: (m.1.height + th - 1) / th, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tw, height: th, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        return cb.status == .completed
    }
}

// MARK: - GPU interpolator (RIFE on Metal: one pass over the whole frame, no tiles)

final class RifeEngine {
    private let interpolator: RifeInterpolator
    private let stream: RifeStream
    private var transfer: VTPixelTransferSession?
    private let pool: BufferPool
    private(set) var lastConvertMs = 0.0, lastEngineMs = 0.0

    init?(width: Int, height: Int, tier: RifeQualityTier) {
        do {
            let i = try RifeInterpolator(configuration: .bundled(qualityTier: tier))
            let s = try i.makeStream(width: width, height: height)
            interpolator = i; stream = s
        } catch { Log.shared.line("rife unavailable: \(error)"); return nil }
        pool = BufferPool(width: width, height: height, format: kCVPixelFormatType_32BGRA,
                          attrs: [kCVPixelBufferMetalCompatibilityKey as String: true])
        VTPixelTransferSessionCreate(allocator: nil, pixelTransferSessionOut: &transfer)
    }
    /// The engine takes BGRA; the capture card delivers NV12.
    func convert(_ source: CVPixelBuffer) -> CVPixelBuffer? {
        let t0 = CACurrentMediaTime(); defer { lastConvertMs = (CACurrentMediaTime() - t0) * 1000 }
        guard let transfer, let dst = pool.get() else { return nil }
        guard VTPixelTransferSessionTransferImage(transfer, from: source, to: dst) == noErr else { return nil }
        return dst
    }
    private(set) var lastProtectMs = 0.0, lastProtectedShare = 0.0
    /// Subtitles, HUD and other overlays do not move, but the motion estimate drags them along with the picture
    /// behind them, so in every generated frame they are bent a little: on screen they flicker at 30 Hz.
    /// Wherever a pixel (and its neighbours in the row) is the same in both originals, the generated frame takes
    /// the original pixel instead.
    func protectStill(previous: CVPixelBuffer, current: CVPixelBuffer, mid: CVPixelBuffer) {
        let t0 = CACurrentMediaTime(); defer { lastProtectMs = (CACurrentMediaTime() - t0) * 1000 }
        CVPixelBufferLockBaseAddress(previous, .readOnly); CVPixelBufferLockBaseAddress(current, .readOnly); CVPixelBufferLockBaseAddress(mid, [])
        defer { CVPixelBufferUnlockBaseAddress(mid, []); CVPixelBufferUnlockBaseAddress(current, .readOnly); CVPixelBufferUnlockBaseAddress(previous, .readOnly) }
        guard let pp = CVPixelBufferGetBaseAddress(previous)?.assumingMemoryBound(to: UInt8.self),
              let pc = CVPixelBufferGetBaseAddress(current)?.assumingMemoryBound(to: UInt8.self),
              let pm = CVPixelBufferGetBaseAddress(mid)?.assumingMemoryBound(to: UInt8.self) else { return }
        let w = CVPixelBufferGetWidth(mid), h = CVPixelBufferGetHeight(mid)
        guard CVPixelBufferGetWidth(previous) == w, CVPixelBufferGetWidth(current) == w, CVPixelBufferGetHeight(previous) == h, CVPixelBufferGetHeight(current) == h else { return }
        let rp = CVPixelBufferGetBytesPerRow(previous), rc = CVPixelBufferGetBytesPerRow(current), rm = CVPixelBufferGetBytesPerRow(mid)
        let bands = 12, tol = 3, minRun = 4
        var protected = [Int](repeating: 0, count: bands)
        protected.withUnsafeMutableBufferPointer { counts in
            let counts = counts.baseAddress!
            DispatchQueue.concurrentPerform(iterations: bands) { bi in
                var n = 0
                for y in (h * bi / bands)..<(h * (bi + 1) / bands) {
                    let a = pp + y * rp, b = pc + y * rc, m = pm + y * rm
                    var run = 0
                    for x in 0..<w {
                        let i = x * 4
                        let still = abs(Int(a[i]) - Int(b[i])) <= tol && abs(Int(a[i+1]) - Int(b[i+1])) <= tol && abs(Int(a[i+2]) - Int(b[i+2])) <= tol
                        if still { run += 1 } else {
                            if run >= minRun { memcpy(m + (x - run) * 4, b + (x - run) * 4, run * 4); n += run }
                            run = 0
                        }
                    }
                    if run >= minRun { memcpy(m + (w - run) * 4, b + (w - run) * 4, run * 4); n += run }
                }
                counts[bi] = n
            }
        }
        lastProtectedShare = Double(protected.reduce(0, +)) / Double(max(1, w * h))
    }
    /// Several in-between pictures at once (e.g. 1/3 and 2/3 when the game held a picture for three refreshes).
    func push(_ frame: CVPixelBuffer, timesteps: [Float]) -> [CVPixelBuffer] {
        let t0 = CACurrentMediaTime(); defer { lastEngineMs = (CACurrentMediaTime() - t0) * 1000 }
        do { return try stream.push(frame, timesteps: timesteps) } catch { Log.shared.line("rife error: \(error)"); stream.reset(); return [] }
    }
    func reset() { stream.reset() }
    /// Midpoint between the previously pushed frame and this one; nil on the first frame after a reset.
    func push(_ frame: CVPixelBuffer) -> CVPixelBuffer? {
        let t0 = CACurrentMediaTime(); defer { lastEngineMs = (CACurrentMediaTime() - t0) * 1000 }
        do { return try stream.push(frame) } catch { Log.shared.line("rife error: \(error)"); stream.reset(); return nil }
    }
}

// MARK: - Pipeline

final class Pipeline: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    let layer = AVSampleBufferDisplayLayer()
    private let queue = DispatchQueue(label: "capture.interp", qos: .userInteractive)
    private let captureQueue = DispatchQueue(label: "capture.deliver", qos: .userInteractive)
    private let pendingLock = NSLock()
    private var pendingFrames = 0
    private var interp: Interpolator?
    private var prevScaled: CVPixelBuffer?
    private var tiled: TiledInterpolator?
    private var prevTiles: [CVPixelBuffer]?
    private var tiledFailed = false
    private(set) var mode: Mode = !opts.interpolate ? .pass : (opts.rife == "balanced" ? .rifeBalanced : (!opts.rife.isEmpty ? .rifeFast : (opts.tiles > 1 ? .tiled : .single)))
    private var sourceW = 0, sourceH = 0
    private var fastSourceNoted = false
    private var warmup = 0
    var onStatus: ((String) -> Void)?          // short on-screen status line, called on the main queue
    private var deviceName = ""
    private var reconnectTimer: Timer?
    private var fileReader: AVAssetReader?   // must outlive the decode loop
    private var prevPTS = CMTime.invalid
    private var audio: AudioDelay?
    private let audioQueue = DispatchQueue(label: "capture.audio", qos: .userInteractive)
    // display refresh grid, updated from the screen's display link
    private let vsLock = NSLock()
    private var vsTarget = 0.0, vsPeriod = 0.0
    private var prevMidSlot = 0.0
    private var holdMs: [Double] = []         // capture timestamp -> refresh on which the original appears

    func updateVsync(target: Double, period: Double) {
        vsLock.lock(); vsTarget = target; vsPeriod = period; vsLock.unlock()
    }

    // per-second stats
    private var frames = 0, mids = 0, lateMids = 0, lateOrig = 0, failed = 0
    private var deliveryMs: [Double] = [], readyMs: [Double] = [], interpMs: [Double] = []
    private var lastReport = hostNow()
    private var readyHistory: [Double] = []   // rolling (ready - capture PTS), ms
    private var liveOffsetMs = 0.0
    private var audioHoldHistory: [Double] = []   // last ~20 s of picture holds
    private var effOffsetMs = 0.0             // hold actually used for the current frame, from capture PTS
    private var offsetBootstrapped = false
    private var gapHistory: [Double] = []     // recent source frame intervals, ms
    private var nominalHalfMs = 1000.0 / 60.0 // half a source interval, smoothed
    private var lastEnqueued = CMTime.invalid // presentation times must only move forward
    private var procMs: [Double] = []         // frame arrival -> handed to the display layer
    private var droppedMids = 0
    private var seamDiffs: [Double] = [], textureDiffs: [Double] = []
    private var lumaSteps: [Double] = []      // |mid brightness - average of its two neighbours|
    private var prevWorkLuma = 0.0
    private var sourceSize = ""

    func start() -> Bool {
        guard let dev = preferredVideoDevice(deviceName.isEmpty ? opts.device : deviceName) else {
            Log.shared.line("no matching video device; have: \(videoDevices().map(\.localizedName))")
            return false
        }
        deviceName = dev.localizedName
        // Highest 16:9 resolution that offers ~30 fps (29.97 included); among equals prefer
        // NV12, then other uncompressed formats, and MJPEG last.
        func has30(_ f: AVCaptureDevice.Format) -> Bool {
            f.videoSupportedFrameRateRanges.contains { $0.minFrameRate <= opts.fps + 0.01 && $0.maxFrameRate >= opts.fps - 0.1 }
        }
        func sub(_ f: AVCaptureDevice.Format) -> String { fourCC(CMFormatDescriptionGetMediaSubType(f.formatDescription)).lowercased() }
        func isMJPEG(_ f: AVCaptureDevice.Format) -> Bool { sub(f).contains("jp") || sub(f) == "dmb1" }
        let sized = dev.formats.filter { opts.maxWidth == 0 || Int(CMVideoFormatDescriptionGetDimensions($0.formatDescription).width) <= opts.maxWidth }
        var candidates = sized.filter(has30)
        if candidates.isEmpty { candidates = sized.filter { f in f.videoSupportedFrameRateRanges.contains { $0.maxFrameRate >= 29.9 } } }
        func score(_ f: AVCaptureDevice.Format) -> Int {
            let d = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
            let wide = abs(Double(d.width) / Double(d.height) - 16.0 / 9.0) < 0.02
            let kind = isMJPEG(f) ? 0 : (["420v", "420f"].contains(sub(f)) ? 3 : (["2vuy", "yuvs", "yuy2"].contains(sub(f)) ? 2 : 1))
            return (wide ? 1_000_000_000 : 0) + Int(d.width) * Int(d.height) * 4 + kind
        }
        guard let format = candidates.max(by: { score($0) < score($1) }) else {
            Log.shared.line("device has no 30 fps format"); return false
        }
        if isMJPEG(format) { Log.shared.line("warning: this size is only offered as MJPEG (compressed); expect extra delay. Try --max-width 1920 for an uncompressed format.") }
        do {
            let input = try AVCaptureDeviceInput(device: dev)
            session.beginConfiguration()
            for old in session.inputs { session.removeInput(old) }      // clean slate when reconnecting
            for old in session.outputs { session.removeOutput(old) }
            guard session.canAddInput(input) else { return false }
            session.addInput(input)
            try dev.lockForConfiguration()
            dev.activeFormat = format
            // Lock ~30 fps. A 29.97-only range needs its own duration (1001/30000), not 1/30.
            if let r = format.videoSupportedFrameRateRanges.first(where: { $0.minFrameRate <= opts.fps + 0.01 && $0.maxFrameRate >= opts.fps - 0.1 }) {
                // Capture cards report fixed-point ranges (e.g. 1000000/30000030); anything but the
                // range's own duration throws. Only a real span (e.g. 1–60) takes a plain 1/30.
                let fixed = r.maxFrameRate - r.minFrameRate < 0.5
                let dur = fixed ? r.minFrameDuration : CMTime(value: 1, timescale: Int32(opts.fps))
                dev.activeVideoMinFrameDuration = dur
                dev.activeVideoMaxFrameDuration = dur
            } else {
                Log.shared.line("warning: could not lock 30 fps on this format; source may run faster")
            }
            dev.unlockForConfiguration()
            let out = AVCaptureVideoDataOutput()
            out.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
                                 kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()]
            out.alwaysDiscardsLateVideoFrames = true
            out.setSampleBufferDelegate(self, queue: captureQueue)
            guard session.canAddOutput(out) else { return false }
            session.addOutput(out)
            // "auto": the capture card's own audio, matched by name.
            let wantAudio = opts.audio == "auto" ? String(dev.localizedName.prefix(12)) : opts.audio
            if !opts.audio.isEmpty,
               let mic = audioDevices().first(where: { $0.localizedName.localizedCaseInsensitiveContains(wantAudio) })
                ?? (opts.audio == "auto" ? audioDevices().first(where: { $0.deviceType == .external }) : nil),
               let micIn = try? AVCaptureDeviceInput(device: mic), session.canAddInput(micIn) {
                session.addInput(micIn)
                let delayed = AudioDelay()
                delayed.session = session
                let aout = AVCaptureAudioDataOutput()
                aout.setSampleBufferDelegate(delayed, queue: audioQueue)
                if session.canAddOutput(aout) { session.addOutput(aout); audio = delayed }
                Log.shared.line("audio: \(mic.localizedName) (delayed to match the picture, volume \(opts.audioVolume))")
            }
            session.commitConfiguration()
        } catch {
            Log.shared.line("capture setup failed: \(error.localizedDescription)"); return false
        }
        let d = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
        sourceSize = "\(d.width)x\(d.height)"
        Log.shared.line("device: \(dev.localizedName) format: \(sourceSize) \(fourCC(CMFormatDescriptionGetMediaSubType(format.formatDescription))) interp: \(opts.interpolate) offset: \(opts.offsetMs == 0 ? "adaptive" : "\(opts.offsetMs) ms")")

        // Present on the host clock: every frame carries the exact time it should appear.
        var tb: CMTimebase?
        CMTimebaseCreateWithSourceClock(allocator: nil, sourceClock: hostClock, timebaseOut: &tb)
        if let tb {
            CMTimebaseSetTime(tb, time: hostNow())
            CMTimebaseSetRate(tb, rate: 1)
            layer.controlTimebase = tb
        }
        layer.videoGravity = .resizeAspect
        layer.backgroundColor = NSColor.black.cgColor
        prepareInterpolators(width: Int(d.width), height: Int(d.height))
        sourceW = Int(d.width); sourceH = Int(d.height)
        NotificationCenter.default.removeObserver(self)
        NotificationCenter.default.addObserver(self, selector: #selector(deviceLost(_:)), name: .AVCaptureDeviceWasDisconnected, object: dev)
        NotificationCenter.default.addObserver(self, selector: #selector(deviceLost(_:)), name: .AVCaptureSessionRuntimeError, object: session)
        session.startRunning()
        // The session resets the device's frame rate when it commits/starts (the VC-009Pro fell
        // back to its 25 fps range), so the 30 fps lock has to be applied again once running.
        if let r = format.videoSupportedFrameRateRanges.first(where: { $0.minFrameRate <= opts.fps + 0.01 && $0.maxFrameRate >= opts.fps - 0.1 }),
           (try? dev.lockForConfiguration()) != nil {
            if dev.activeFormat != format { dev.activeFormat = format }
            let dur = r.maxFrameRate - r.minFrameRate < 0.5 ? r.minFrameDuration : CMTime(value: 1, timescale: Int32(opts.fps))
            dev.activeVideoMinFrameDuration = dur
            dev.activeVideoMaxFrameDuration = dur
            dev.unlockForConfiguration()
            Log.shared.line("frame rate locked after start: \(String(format: "%.2f", 1 / dev.activeVideoMinFrameDuration.seconds)) fps")
        }
        return true
    }

    /// The card was unplugged, or the console changed its signal: wait and reconnect.
    @objc private func deviceLost(_ note: Notification) {
        DispatchQueue.main.async {
            guard self.reconnectTimer == nil else { return }
            Log.shared.line("capture device lost (\(note.name.rawValue)); waiting to reconnect")
            self.onStatus?("采集设备断开，等待重新连接…")
            self.session.stopRunning()
            self.reconnectTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { t in
                guard preferredVideoDevice(self.deviceName) != nil else { return }
                t.invalidate(); self.reconnectTimer = nil
                self.queue.async { self.resetTiming() }
                if self.start() { Log.shared.line("reconnected") } else { self.deviceLost(note) }
            }
        }
    }

    private func resetTiming() {
        prevScaled = nil; prevTiles = nil; prevPTS = .invalid
        readyHistory.removeAll(); gapHistory.removeAll(); offsetBootstrapped = false; audioHoldHistory.removeAll()
        liveOffsetMs = 0; prevMidSlot = 0; fastSourceNoted = false; warmup = 0
    }

    /// Switch mode while running (keys 1/2/3). Applied at the next frame, on the processing thread.
    func setMode(_ m: Mode) {
        modeLock.lock(); pendingMode = m; modeLock.unlock()
    }
    private let modeLock = NSLock()
    private var pendingMode: Mode?

    private func applyPendingMode() {
        modeLock.lock(); let m = pendingMode; pendingMode = nil; modeLock.unlock()
        guard let m, m != mode else { return }
        mode = m
        rifeEngines[m]?.reset()
        resetTiming()
        ensureInterpolators()
        Log.shared.line("mode: \(m.rawValue) \(m.title)")
        let text = "模式：" + m.title + "　　按 1 2 3 4 5 切换"
        DispatchQueue.main.async { self.onStatus?(text) }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let arrival = hostNow()
        // Capture timestamps are on the session clock; express them on the host clock.
        var pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        if let clock = session.synchronizationClock { pts = CMSyncConvertTime(pts, from: clock, to: hostClock) }
        guard acceptFrame(pb, pts: pts) else { return }
        pendingLock.lock()
        if pendingFrames >= 2 { pendingLock.unlock(); return }      // processing is behind: skip rather than queue up delay
        pendingFrames += 1
        pendingLock.unlock()
        queue.async {
            let t0 = hostNow()
            defer { let took = ms(CMTimeSubtract(hostNow(), t0)); if took > 100 { Log.shared.line(String(format: "slow frame: %.0f ms total, waited %.0f ms in queue", took, ms(CMTimeSubtract(t0, arrival)))) } }
            self.handle(pb, pts: pts, arrival: arrival)
            self.pendingLock.lock(); self.pendingFrames -= 1; self.pendingLock.unlock()
        }
    }

    /// File simulation: decode a video and hand frames over on a steady ~30 fps clock,
    /// as a capture device would. Decode happens ahead of time so it is not counted.
    func startFile(_ path: String) -> Bool {
        let asset = AVURLAsset(url: URL(fileURLWithPath: path))
        guard let track = asset.tracks(withMediaType: .video).first, let reader = try? AVAssetReader(asset: asset) else {
            Log.shared.line("cannot open \(path)"); return false
        }
        let out = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()])
        out.alwaysCopiesSampleData = false
        reader.add(out)
        guard reader.startReading() else { Log.shared.line("reader failed"); return false }
        fileReader = reader
        let fps = Double(track.nominalFrameRate)
        let step = max(1, Int((fps / 30.0).rounded()))   // 60 fps file -> every 2nd frame
        let size = track.naturalSize
        sourceSize = "\(Int(size.width))x\(Int(size.height))"
        Log.shared.line("file: \((path as NSString).lastPathComponent) \(sourceSize) \(fps) fps, feeding every \(step) frame(s) interp: \(opts.interpolate)")
        var tb: CMTimebase?
        CMTimebaseCreateWithSourceClock(allocator: nil, sourceClock: hostClock, timebaseOut: &tb)
        if let tb { CMTimebaseSetTime(tb, time: hostNow()); CMTimebaseSetRate(tb, rate: 1); layer.controlTimebase = tb }
        layer.videoGravity = .resizeAspect
        sourceW = Int(size.width); sourceH = Int(size.height)
        prepareInterpolators(width: sourceW, height: sourceH)
        let interval = Double(step) / fps
        let feeder = DispatchQueue(label: "file.decode", qos: .userInitiated)
        let ready = DispatchSemaphore(value: 0)
        var pending: [CVPixelBuffer] = []
        let lock = NSLock()
        feeder.async {   // decode ahead, keep a small backlog
            var n = 0
            var more = true
            while more {
                autoreleasepool {
                    guard let sb = out.copyNextSampleBuffer() else { more = false; return }
                    n += 1
                    guard n % step == 0, let pb = CMSampleBufferGetImageBuffer(sb) else { return }
                    while true { lock.lock(); let c = pending.count; lock.unlock(); if c < 3 { break }; usleep(2000) }
                    lock.lock(); pending.append(pb); lock.unlock()
                    ready.signal()
                }
            }
            usleep(400_000)   // let the last frames reach the screen
            DispatchQueue.main.async { NSApp.terminate(nil) }
        }
        queue.async {
            var next = CMTimeGetSeconds(hostNow()) + 0.5
            while true {
                ready.wait()
                lock.lock(); let pb = pending.removeFirst(); lock.unlock()
                let wait = next - CMTimeGetSeconds(hostNow())
                if wait > 0 { usleep(useconds_t(wait * 1_000_000)) }
                else if wait < -0.1 { next = CMTimeGetSeconds(hostNow()) }   // after a stall, resume the cadence instead of bursting
                autoreleasepool {
                    let now = hostNow()
                    self.handle(pb, pts: now, arrival: now)   // a device with zero delivery delay
                }
                next += interval
            }
        }
        return true
    }

    /// Create the interpolation sessions before the first frame, so start-up does not stall the stream.
    private func prepareInterpolators(width: Int, height: Int) {
        sourceW = width; sourceH = height
        ensureInterpolators()
    }

    /// Make sure the sessions needed by the current mode exist.
    private var rifeEngines: [Mode: RifeEngine] = [:]
    private var rifeFailed = false
    private var rifeDebugCount = 0
    private var longGaps = 0, earlyShown = 0
    private lazy var stillProtector = StillProtector()
    private var lastProtectMs = 0.0
    private var dumpSeen = 0
    private var rifePrevBGRA: CVPixelBuffer?
    private func ensureInterpolators() {
        if let tier = mode.rifeTier, rifeEngines[mode] == nil, !rifeFailed, sourceW > 0 {
            if let e = RifeEngine(width: sourceW, height: sourceH, tier: tier) {
                rifeEngines[mode] = e
                Log.shared.line("rife mode: \(sourceW)x\(sourceH) whole frame on the GPU, tier \(tier.rawValue)")
            } else { rifeFailed = true; Log.shared.line("rife mode unavailable, showing originals only") }
        }
        if mode == .tiled && tiled == nil && !tiledFailed {
            let grid = opts.grid ?? (opts.tiles == 9 ? Grid.k3 : Grid.k2)
            tiled = TiledInterpolator(grid)
            if tiled == nil { tiledFailed = true; Log.shared.line("tile mode unavailable, using single pass") }
            else { Log.shared.line("tile mode: \(grid.cols)x\(grid.rows) of \(grid.tileW)x\(grid.tileH) -> \(grid.workW)x\(grid.workH), originals \(opts.origNative ? "native" : "work size")") }
        }
        if (mode == .single || (mode == .tiled && tiledFailed)) && interp == nil && sourceW > 0 {
            interp = Interpolator(sourceWidth: sourceW, sourceHeight: sourceH)
            if let i = interp { Log.shared.line("interp size: \(i.width)x\(i.height)") } else { Log.shared.line("interpolation unavailable, passing frames through") }
        }
    }

    /// True when frames should simply be shown as they arrive.
    private var passing: Bool {
        if mode == .pass { return true }
        if rifeEngines[mode] != nil { return false }
        if mode == .tiled && tiled != nil && !tiledFailed { return false }
        return interp == nil
    }

    /// Hand a frame to the display layer. Presentation times never go backwards or collide.
    @discardableResult
    private func enqueue(_ pb: CVPixelBuffer, at time: CMTime) -> CMTime {
        var t = time
        if lastEnqueued.isValid {
            let floor = CMTimeAdd(lastEnqueued, CMTime(value: 4, timescale: 1000))
            if t < floor { t = floor }
        }
        if layer.status == .failed { layer.flush() }
        if let sb = makeSample(pb, pts: t) { layer.enqueue(sb) }
        lastEnqueued = t
        return t
    }

    // --- 60 fps capture carrying a 30 fps game: every picture arrives twice. ---
    private var lastRaw: CVPixelBuffer?
    private var recentDiffs: [Double] = []      // frame-to-frame differences of the last two seconds
    private var lastSeenPTS = CMTime.invalid, lastKeptPTS = CMTime.invalid, lastKeptWasRepeat = false
    private(set) var contentIs60 = false

    /// Sparse luma comparison: true when this frame is the same picture as the previous one.
    private func differenceFromPrevious(_ pb: CVPixelBuffer) -> Double {
        defer { lastRaw = pb }
        guard let prev = lastRaw, CVPixelBufferGetWidth(prev) == CVPixelBufferGetWidth(pb),
              CVPixelBufferGetHeight(prev) == CVPixelBufferGetHeight(pb) else { return 99 }
        CVPixelBufferLockBaseAddress(pb, .readOnly); CVPixelBufferLockBaseAddress(prev, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly); CVPixelBufferUnlockBaseAddress(prev, .readOnly) }
        guard let a = CVPixelBufferGetBaseAddressOfPlane(pb, 0), let b = CVPixelBufferGetBaseAddressOfPlane(prev, 0) else { return 99 }
        let w = CVPixelBufferGetWidthOfPlane(pb, 0), h = CVPixelBufferGetHeightOfPlane(pb, 0)
        let ra = CVPixelBufferGetBytesPerRowOfPlane(pb, 0), rb = CVPixelBufferGetBytesPerRowOfPlane(prev, 0)
        let pa = a.assumingMemoryBound(to: UInt8.self), pbb = b.assumingMemoryBound(to: UInt8.self)
        var sum = 0, n = 0
        var y = 8
        while y < h {
            var x = 8
            while x < w { sum += abs(Int(pa[y * ra + x]) - Int(pbb[y * rb + x])); n += 1; x += 16 }
            y += 16
        }
        let d = n > 0 ? Double(sum) / Double(n) : 99
        diffLog.append(d); if diffLog.count >= 120 {
            let v = diffLog.sorted(); diffLog.removeAll()
            Log.shared.line(String(format: "   frame-to-frame difference over 2 s: p10 %.3f p25 %.3f p50 %.3f p75 %.3f p90 %.3f", v[12], v[30], v[60], v[90], v[108]))
        }
        return d
    }
    private var diffLog: [Double] = []
    private let seenLock = NSLock()
    private var capIndex = 0, pickParity = 0, parityVotes = 0
    private var contentUsed = false, tentative = false, repeatRun = 0
    private var parityChange = [0.0, 0.0]         // recent picture change seen on even / odd capture frames
    private(set) var paritySwitches = 0
    private var seenCount = 0

    /// Runs on the light capture queue for every delivered frame; decides whether the frame is processed.
    /// Keeping this off the processing queue matters: a 60 fps device must never wait for a generated frame.
    private func acceptFrame(_ pb: CVPixelBuffer, pts: CMTime) -> Bool {
        if opts.fps >= 50 {
            // Frames can go missing while a generated frame is being computed, so only neighbours
            // one 60 Hz step apart say anything about repeats.
            let seenGap = lastSeenPTS.isValid ? ms(CMTimeSubtract(pts, lastSeenPTS)) : -1
            lastSeenPTS = pts
            seenLock.lock(); seenCount += 1; seenLock.unlock()
            if seenGap > 60 || (seenGap < 0 && seenGap != -1) { Log.shared.line(String(format: "capture gap: %.0f ms between delivered frames", seenGap)) }
            let d = differenceFromPrevious(pb)
            let neighbour = seenGap > 5 && seenGap < 22          // previous frame really was the one before
            var isRepeat = false, known = false
            if neighbour {
                recentDiffs.append(d); if recentDiffs.count > 120 { recentDiffs.removeFirst() }
                // A repeat differs from its original by nothing, or by a little noise (4K capture).
                // Anything clearly below the typical change of the last two seconds counts as one.
                // Reference = upper quartile: with half the frames being repeats the median sits right
                // on the edge between the two groups and would make the verdict flip constantly.
                let ref = recentDiffs.sorted()[recentDiffs.count * 3 / 4]
                let limit = max(0.9, 0.6 * ref)        // repeats measured at up to ~40% of a real change (noisy 4K scenes)
                isRepeat = d < limit; known = true
                if recentDiffs.count >= 90 {
                    let share = Double(recentDiffs.filter { $0 < limit }.count) / Double(recentDiffs.count)
                    // Wide gap between the two verdicts so the mode cannot flip back and forth.
                    let now60 = share < 0.12 ? true : (share > 0.30 ? false : contentIs60)
                    if now60 != contentIs60 {
                        contentIs60 = now60
                        Log.shared.line(String(format: "game content is ~%d fps (repeated frames %.0f%%)", now60 ? 60 : 30, share * 100))
                        if mode != .pass && opts.autoPass { queue.async { self.resetTiming() }; let t = now60 ? "游戏已是 60 帧，自动直通" : "游戏是 30 帧，补到 60"; DispatchQueue.main.async { self.onStatus?(t) } }
                    }
                }
            } else { recentDiffs.removeAll() }
            if mode != .pass && !(opts.autoPass && contentIs60) {
                // Use the first copy of each picture (it is 16.7 ms fresher). While the picture stands
                // still nothing tells the copies apart, so simply keep every other frame.
                let keptGap = lastKeptPTS.isValid ? ms(CMTimeSubtract(pts, lastKeptPTS)) : 1000
                var keep: Bool
                if opts.legacyPick {
                    if keptGap < 0 || keptGap > 60 { keep = true }
                    else if known && !isRepeat { keep = keptGap >= 25 || lastKeptWasRepeat }
                    else if known { keep = keptGap >= 31 }                 // a repeat: only on the 30 fps cadence (still picture, or its original was missed)
                    else { keep = keptGap >= 25 }
                } else {
                    // Judging every frame on its own ("is this a repeat?") goes wrong in dark or slow scenes, and each
                    // wrong verdict shifts the 30 fps cadence by one capture frame: two pictures 16.7 ms apart, then
                    // none for 50 ms (seen in a real game as "in 30.4-31.2 fps" and as flicker on the glass).
                    // Instead: number the capture frames, take every other one, and move to the other half only
                    // when the evidence has pointed there for a while.
                    capIndex += seenGap > 5 ? max(1, Int((seenGap / 16.6667).rounded())) : 1
                    let par = capIndex & 1
                    if neighbour { parityChange[par] = parityChange[par] * 0.9 + d; parityChange[1 - par] *= 0.9 }
                    if opts.votePick {
                        let other = 1 - pickParity
                        if parityChange[other] > 1.6 * parityChange[pickParity] + 0.3 { parityVotes += 1 } else { parityVotes = 0 }
                        if parityVotes >= 10 { pickParity = other; parityVotes = 0; paritySwitches += 1 }
                        keep = par == pickParity || keptGap < 0 || keptGap > 60
                    } else {
                        // Third version. With 60 fps capture a repeated picture is bit-identical to the one before
                        // (difference exactly 0), so "repeat" needs no guessing: the earlier limits (0.9, or a share
                        // of the recent average) called slow real pictures repeats. Stay on one half; when the frame
                        // due now repeats a picture already used (the console held it for three capture frames),
                        // skip it and take the very next frame if that one is new: one 50 ms step, nothing lost,
                        // instead of a repeated picture, several stale ones and a 50 ms hole.
                        let repeated = neighbour && d < 0.003
                        repeatRun = repeated ? repeatRun + 1 : 0
                        if !repeated { contentUsed = false }
                        if keptGap < 0 || keptGap > 60 { keep = true; tentative = false }
                        else if tentative {
                            tentative = false
                            if repeated { pickParity = 1 - pickParity; keep = false }     // nothing new after all: back to the old half
                            else { keep = true; paritySwitches += 1 }
                        }
                        else if par == pickParity {
                            // repeatRun > 2: the picture stands still (or the game stalled); just keep the cadence.
                            if repeated && contentUsed && repeatRun <= 2 { pickParity = 1 - par; tentative = true; keep = false } else { keep = true }
                        }
                        else { keep = false }
                    }
                    if keep && keptGap >= 0 && keptGap < 25 { keep = false }      // never two pictures one capture frame apart
                    if keep { contentUsed = true }
                }
                if !keep { return false }
                lastKeptPTS = pts
                lastKeptWasRepeat = known && isRepeat
            }
        }
        return true
    }

    private func handle(_ pb: CVPixelBuffer, pts: CMTime, arrival: CMTime) {
        applyPendingMode()
        frames += 1
        deliveryMs.append(ms(CMTimeSubtract(arrival, pts)))

        // A source that already runs at ~60 fps needs no generated frames on a 60 Hz screen.
        let srcGap = prevPTS.isValid ? ms(CMTimeSubtract(pts, prevPTS)) : -1
        vsLock.lock(); let screenPeriod = vsPeriod; vsLock.unlock()
        let fastSource = mode != .pass && screenPeriod > 0.015
            && (opts.fps >= 50 ? (opts.autoPass && contentIs60) : (gapHistory.count >= 20 && nominalHalfMs < 11))
        if fastSource && !fastSourceNoted {
            fastSourceNoted = true
            Log.shared.line("source is already ~60 fps on a 60 Hz screen: passing through")
            DispatchQueue.main.async { self.onStatus?("信号已是 60 帧，自动直通") }
        }
        if passing || fastSource {
            if srcGap > 5 && srcGap < 60 {
                gapHistory.append(srcGap)
                if gapHistory.count > 60 { gapHistory.removeFirst(gapHistory.count - 60) }
                nominalHalfMs = min(20, max(7, percentile(gapHistory, 0.5) / 2))
            }
            prevPTS = pts
            audio?.setDelay(0)                          // picture is not held, so neither is the sound
            // Pass-through: no waiting at all, show the frame as soon as it arrives.
            if layer.status == .failed { layer.flush() }
            if let sb = makeSample(pb, pts: arrival) {
                if let arr = CMSampleBufferGetSampleAttachmentsArray(sb, createIfNecessary: true), CFArrayGetCount(arr) > 0 {
                    let d = unsafeBitCast(CFArrayGetValueAtIndex(arr, 0), to: CFMutableDictionary.self)
                    CFDictionarySetValue(d, Unmanaged.passUnretained(kCMSampleAttachmentKey_DisplayImmediately).toOpaque(),
                                         Unmanaged.passUnretained(kCFBooleanTrue).toOpaque())
                }
                layer.enqueue(sb)
            }
            procMs.append(ms(CMTimeSubtract(hostNow(), arrival)))
            report()
            return
        }

        // Source cadence: only bridge frames that are really consecutive.
        let gapMs = prevPTS.isValid ? ms(CMTimeSubtract(pts, prevPTS)) : -1
        let consecutive = gapMs > 5 && gapMs < (rifeEngines[mode] != nil && opts.multiMid ? 76 : 60)   // GPU mode also bridges a picture held for four refreshes
        if consecutive {
            gapHistory.append(gapMs)
            if gapHistory.count > 60 { gapHistory.removeFirst(gapHistory.count - 60) }
            nominalHalfMs = min(20, max(7, percentile(gapHistory, 0.5) / 2))   // smoothed, so jitter cannot reorder frames
        }
        let half = CMTime(seconds: nominalHalfMs / 1000.0, preferredTimescale: 1_000_000_000)

        // How long to hold frames: follow the slowest recent frames; rise at once, come down slowly.
        // The screen compositor needs the GPU around every refresh; a GPU interpolation running just then makes it
        // miss refreshes. Optionally start the job at a fixed point of the refresh cycle.
        if opts.gpuPhaseMs >= 0, rifeEngines[mode] != nil {
            vsLock.lock(); let vt0 = vsTarget, vp0 = vsPeriod; vsLock.unlock()
            if vp0 > 0.004 {
                var ph = (CMTimeGetSeconds(hostNow()) - vt0).truncatingRemainder(dividingBy: vp0); if ph < 0 { ph += vp0 }
                var wait = opts.gpuPhaseMs / 1000.0 - ph; if wait < -0.0015 { wait += vp0 }
                if wait > 0 { Thread.sleep(forTimeInterval: wait) }
            }
        }
        let schedNow = hostNow()
        let follow = opts.followArrival && opts.offsetMs <= 0
        if opts.offsetMs > 0 { liveOffsetMs = opts.offsetMs }
        else if readyHistory.count >= 20 {
            // Big (4K / BGRA) frames reaching the display layer only ~2 ms before their time often miss that refresh
            // (measured on the glass: bursts of shared refreshes whenever the hold sat on its lower step). 6 ms was clean.
            let big = rifeEngines[mode] != nil || (sourceW > 2600 && !(mode == .tiled && !opts.origNative))
            let want = percentile(readyHistory, 0.98) + (opts.marginMs >= 0 ? opts.marginMs : (big ? 8.0 : 2.0))
            if !offsetBootstrapped { liveOffsetMs = want; offsetBootstrapped = true }
            else if want > liveOffsetMs { liveOffsetMs = want }
            else if liveOffsetMs - want > 2 { liveOffsetMs -= max(0.05, (liveOffsetMs - want) * 0.01) }   // far above: come down in seconds, not minutes
        } else if liveOffsetMs == 0 { liveOffsetMs = follow ? 40 : 60 }
        // Follow mode: the capture card's delivery delay drifts in a slow sawtooth (its 60.000 fps clock against the
        // console's 59.94). Each frame's own delay is already known here, so only the processing time is estimated.
        effOffsetMs = follow ? min(100, max(0, ms(CMTimeSubtract(schedNow, pts)))) + liveOffsetMs : liveOffsetMs   // clamp: the first frames can carry wild timestamps
        let offset = CMTime(seconds: effOffsetMs / 1000.0, preferredTimescale: 1_000_000_000)
        var midTime = CMTimeAdd(pts, offset)          // generated frame
        var origTime = CMTimeAdd(midTime, half)        // this original, half a source interval later
        var hold = effOffsetMs + nominalHalfMs
        vsLock.lock(); let vt = vsTarget, vp = vsPeriod; vsLock.unlock()
        if opts.vsync && vp > 0.004 {
            // Put each frame on a definite screen refresh. Timestamp jitter of a millisecond or two can
            // then no longer flip a frame between two neighbouring refreshes.
            let p = CMTimeGetSeconds(pts)
            // Measured on the glass (ScreenCaptureKit frame status): 4K or BGRA frames handed over 3 ms before a refresh
            // often miss it and two frames then share one refresh (looks like flicker, or like no interpolation at all).
            // A sweep (1...15 ms, three 4K modes) found only 5 ms clean: 0 shared refreshes in 51 s per mode, while 3 ms
            // and 7-15 ms all showed bursts. Small NV12 frames are fine with 3 ms.
            let bigFrames = rifeEngines[mode] != nil || (sourceW > 2600 && !(mode == .tiled && !opts.origNative))
            let lead = (opts.leadMs >= 0 ? opts.leadMs : (bigFrames ? 5.0 : 3.0)) / 1000.0
            let desired = p + effOffsetMs / 1000.0 + lead                      // earliest refresh this frame can make
            var slot = vt + ((desired - vt) / vp).rounded(.up) * vp
            if consecutive && prevMidSlot > 0 {
                let steps = max(1, (gapMs / 1000.0 / vp).rounded())
                var expected = prevMidSlot + steps * vp
                expected = vt + ((expected - vt) / vp).rounded() * vp          // back onto the current grid
                // Keep the established phase while it is still reachable and not needlessly late.
                // The hold steps by one refresh as the card's delivery delay drifts. Right at a step the per-frame jitter
                // used to flip it back and forth for seconds; every flip puts two frames on one refresh or leaves one
                // empty (seen on the glass as bursts of flicker). Hence a dead band: go up only when really unreachable
                // (no tolerance: letting the headroom shrink by 3 ms brought the bursts back at the top of every delivery cycle), come down only with a clear 5 ms beyond one refresh to spare.
                if expected >= desired && expected <= desired + vp + (opts.steady ? 0.012 : 0.005) { slot = expected }
            }
            prevMidSlot = slot
            let halfSlots = max(1, (nominalHalfMs / 1000.0 / vp).rounded())
            midTime = CMTime(seconds: slot - lead, preferredTimescale: 1_000_000_000)
            origTime = CMTime(seconds: slot + halfSlots * vp - lead, preferredTimescale: 1_000_000_000)
            hold = (slot + halfSlots * vp - p) * 1000.0
        }
        holdMs.append(hold)
        // The picture's hold now steps up and down with the card's delivery sawtooth (every ~8 s). Each change of the
        // sound's delay is an audible skip, so the sound keeps to the upper step: at most one refresh behind the picture.
        audioHoldHistory.append(hold)
        if audioHoldHistory.count > 600 { audioHoldHistory.removeFirst(audioHoldHistory.count - 600) }
        audio?.setDelay(percentile(audioHoldHistory, 0.95))

        var mid: CVPixelBuffer?
        var earlyMid: CVPixelBuffer?                   // GPU mode, long gap: the 1/3 picture, one refresh before `mid`
        var shown: CVPixelBuffer = pb                  // what represents this original on screen
        var work: CVPixelBuffer?
        let t0 = hostNow()
        if let rife = rifeEngines[mode] {
            if let bgra = rife.convert(pb) {
                shown = bgra                                   // originals and generated frames in one format
                defer { rifePrevBGRA = bgra }
                if !consecutive { rife.reset() }
                // The game does not hold 30 fps evenly (GTA: more than one picture per second stays up for three
                // refreshes instead of two). One in-between picture then leaves a refresh with nothing new; two of
                // them, at 1/3 and 2/3, fill the 50 ms evenly.
                let refreshes = (consecutive && vp > 0.004) ? Int((gapMs / (vp * 1000)).rounded()) : 2
                var pushed: CVPixelBuffer?
                if (refreshes == 3 || refreshes == 4) && opts.multiMid {
                    longGaps += 1
                    // The last two refreshes before the new picture: 1/3 and 2/3 of the way, or 2/4 and 3/4.
                    // The careful tier takes ~25 ms per picture at 4K: two of them overran the 33 ms budget and the delay
                    // climbed to 150 ms. There: one picture only, but at the right fraction (2/3 or 3/4 instead of 1/2).
                    let affordTwo = mode == .rifeFast
                    let steps: [Float] = affordTwo ? [Float(refreshes - 2) / Float(refreshes), Float(refreshes - 1) / Float(refreshes)] : [Float(refreshes - 1) / Float(refreshes)]
                    let two = rife.push(bgra, timesteps: steps)
                    if two.count == 1 { pushed = two[0] }
                    if two.count == 2 {
                        if opts.protectStill, let p = rifePrevBGRA, stillProtector?.apply(previous: p, current: bgra, mid: two[0]) != true { rife.protectStill(previous: p, current: bgra, mid: two[0]) }
                        copyColor(from: bgra, to: two[0]); earlyMid = two[0]; pushed = two[1]
                    }
                } else { pushed = rife.push(bgra) }
                if let m = pushed, consecutive {
                    if opts.protectStill, let p = rifePrevBGRA {
                        let t0 = CACurrentMediaTime()
                        if opts.protectOnCPU || stillProtector?.apply(previous: p, current: bgra, mid: m) != true { rife.protectStill(previous: p, current: bgra, mid: m) }
                        lastProtectMs = (CACurrentMediaTime() - t0) * 1000
                    }
                    copyColor(from: bgra, to: m); mid = m
                    rifeDebugCount += 1
                    if rifeDebugCount % 30 == 0 {
                        let a = meanBGRA(bgra), b = meanBGRA(m)
                        Log.shared.line(String(format: "   rife check: orig BGRA %.1f %.1f %.1f %.1f | mid %.1f %.1f %.1f %.1f | fmt %@ / %@ | %dx%d / %dx%d", a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3],
                                               fourCC(CVPixelBufferGetPixelFormatType(bgra)), fourCC(CVPixelBufferGetPixelFormatType(m)),
                                               CVPixelBufferGetWidth(bgra), CVPixelBufferGetHeight(bgra), CVPixelBufferGetWidth(m), CVPixelBufferGetHeight(m)))
                        if let p = rifePrevBGRA {
                            let pc = compareBGRA(p, bgra), mc = compareBGRA(m, bgra), mp = compareBGRA(m, p)
                            Log.shared.line(String(format: "   rife motion: |prev-cur| %.2f  |mid-cur| %.2f  |mid-prev| %.2f   detail: orig %.2f mid %.2f", pc.diff, mc.diff, mp.diff, mc.gradB, mc.gradA))
                        }
                        Log.shared.line("   rife tags: orig \(CVBufferCopyAttachments(bgra, .shouldPropagate).map { ($0 as NSDictionary).allKeys.count } ?? -1) keys, mid \(CVBufferCopyAttachments(m, .shouldPropagate).map { ($0 as NSDictionary).allKeys.count } ?? -1) keys")
                    }
                }
                else if consecutive { failed += 1 }
            } else { failed += 1; rife.reset() }
        } else if mode == .tiled, let tiled, !tiledFailed {
            if let w = tiled.work(from: pb), let tiles = tiled.crop(w) {
                work = w
                if !opts.origNative { shown = w }
                if consecutive, let prev = prevTiles {
                    let t0 = hostNow()
                    mid = tiled.midpoint(previous: prev, current: tiles, previousPTS: prevPTS, currentPTS: pts, like: w)
                    let took = ms(CMTimeSubtract(hostNow(), t0))
                    if took > 100 { Log.shared.line(String(format: "slow interp: %.0f ms (frame gap %.1f ms)", took, ms(CMTimeSubtract(pts, prevPTS)))) }
                    if mid == nil { failed += 1 }
                }
                prevTiles = tiles
            } else { failed += 1; prevTiles = nil }
        } else if let interp {
            if let scaled = interp.prepare(pb) {
                if consecutive, let prev = prevScaled {
                    mid = interp.midpoint(previous: prev, current: scaled, previousPTS: prevPTS, currentPTS: pts)
                    if mid == nil { failed += 1 }
                }
                prevScaled = scaled
            } else { failed += 1; prevScaled = nil }
        }
        prevPTS = pts

        if !opts.dump.isEmpty, let mid {
            dumpSeen += 1
            if dumpSeen > opts.dumpAfter && dumpSeen <= opts.dumpAfter + 6 {
                let k = dumpSeen - opts.dumpAfter, m = mid, o = shown, dir = opts.dump
                dumpQueue.async {
                    savePNG(m, String(format: "%@/%02d_a_mid.png", dir, k)); savePNG(o, String(format: "%@/%02d_b_orig.png", dir, k))
                    Log.shared.line("dumped pair \(k)")
                }
            }
        }
        if let mid {
            let ready = hostNow()
            interpMs.append(ms(CMTimeSubtract(ready, t0)))
            readyMs.append(ms(CMTimeSubtract(ready, pts)))
            warmup += 1
            if warmup > 10 {                               // the first frames after a start or mode switch are slow
                readyHistory.append(follow ? ms(CMTimeSubtract(ready, schedNow)) : readyMs.last!)
                if readyHistory.count > 180 { readyHistory.removeFirst(readyHistory.count - 180) }
            }
            let lateBy = ms(CMTimeSubtract(ready, midTime))
            if lateBy > 0 { lateMids += 1 }
            if lateBy > 8 { droppedMids += 1 }          // too late to be useful: show only the original
            else {
                if let e = earlyMid, vp > 0.004 {
                    let t = CMTimeSubtract(midTime, CMTime(seconds: vp, preferredTimescale: 1_000_000_000))
                    if ms(CMTimeSubtract(ready, t)) <= 2 { enqueue(e, at: t); earlyShown += 1 }      // only if it can still make its refresh
                }
                enqueue(mid, at: midTime); mids += 1
            }
        }
        if hostNow() > origTime { lateOrig += 1 }
        enqueue(shown, at: origTime)
        procMs.append(ms(CMTimeSubtract(hostNow(), arrival)))

        // Quality statistics, after the frames are on their way.
        if let tiled, let work {
            let wl = meanLuma(work)
            if let mid {
                lumaSteps.append(abs(meanLuma(mid) - (wl + prevWorkLuma) / 2))
                seamDiffs.append(tiled.lastSeamDiff); textureDiffs.append(tiled.lastTextureDiff)
            }
            prevWorkLuma = wl
        }
        report()
    }

    private func report() {
        let now = hostNow()
        let span = CMTimeGetSeconds(CMTimeSubtract(now, lastReport))
        guard span >= 1 else { return }
        var status = ""
        if passing || fastSourceNoted {
            status = String(format: "%@｜%@ 入 %.0f 帧｜程序处理 %.1f 毫秒", mode == .pass ? Mode.pass.title : "信号已 60 帧，自动直通", sourceSize, Double(frames) / span, percentile(procMs, 0.95))
            Log.shared.line(String(format: "src %@ in %.1f fps | pass-through | device delivery p50 %.1f p95 %.1f ms | our processing p95 %.2f ms (arrival -> display layer; excludes vsync and compositor)",
                                   sourceSize, Double(frames) / span, percentile(deliveryMs, 0.5), percentile(deliveryMs, 0.95), percentile(procMs, 0.95)))
        } else {
            // Scheduled hold from capture timestamp to the original's display time. Not included: delay inside the
            // capture device before the timestamp, waiting for the next display refresh, and the system compositor.
            let hold = percentile(holdMs, 0.5)
            seenLock.lock(); let seen = seenCount; seenCount = 0; seenLock.unlock()
            let src = opts.fps >= 50 ? String(format: "采集 %.0f 帧（其中新画面 %.0f）", Double(seen) / span, Double(frames) / span) : String(format: "采集 %.0f 帧", Double(frames) / span)
            status = String(format: "%@｜%@ %@｜补帧后上屏 %.0f 帧｜等待 %.0f 毫秒｜晚到 %d 丢弃 %d", mode.title, sourceSize, src, Double(frames + mids) / span, hold, lateMids, droppedMids)
            Log.shared.line(String(format: "src %@ in %.1f fps | out %.1f fps | mid %d late %d dropped %d fail %d | orig late %d | delivery p50 %.1f p95 %.1f ms | ready p95 %.1f ms | interp p95 %.1f ms | offset %.1f ms | hold to refresh p50 %.1f p95 %.1f ms (%@) | beyond delivery %.1f ms",
                                   sourceSize, Double(frames) / span, Double(frames + mids) / span, mids, lateMids, droppedMids, failed, lateOrig,
                                   percentile(deliveryMs, 0.5), percentile(deliveryMs, 0.95), percentile(readyMs, 0.95), percentile(interpMs, 0.95),
                                   effOffsetMs, hold, percentile(holdMs, 0.95), opts.vsync && vsPeriod > 0.004 ? "on refresh grid" : "not aligned",
                                   hold - percentile(deliveryMs, 0.5)))
        }
        if opts.fps >= 50 && !opts.legacyPick { Log.shared.line("   picked half: \(pickParity), switches so far \(paritySwitches) | long gaps (50-67 ms) \(longGaps), filled with two pictures \(earlyShown)") }
        if let r = rifeEngines[mode] { Log.shared.line(String(format: "   stages: convert %.1f engine %.1f protect %.1f ms", r.lastConvertMs, r.lastEngineMs, lastProtectMs)) }
        if !seamDiffs.isEmpty {
            if let tiled { Log.shared.line(String(format: "   stages: scale %.1f cut %.1f engine %.1f stitch %.1f ms", tiled.lastScaleMs, tiled.lastCropMs, tiled.lastEngineMs, tiled.lastStitchMs)) }
            Log.shared.line(String(format: "   seam: tiles differ %.2f in overlap (p95 %.2f) vs natural detail %.2f (0-255 scale)",
                                   seamDiffs.reduce(0, +) / Double(seamDiffs.count), percentile(seamDiffs, 0.95),
                                   textureDiffs.reduce(0, +) / Double(textureDiffs.count)))
            Log.shared.line(String(format: "   brightness: generated frame differs from its neighbours by %.2f (p95 %.2f) of 255",
                                   lumaSteps.reduce(0, +) / Double(max(1, lumaSteps.count)), percentile(lumaSteps, 0.95)))
            seamDiffs.removeAll(); textureDiffs.removeAll(); lumaSteps.removeAll()
        }
        if let audio {
            let (n, re, fl) = audio.takeCounters()
            Log.shared.line(String(format: "   audio: %d buffers, delayed %.0f ms, clock resyncs %d, renderer failures %d", n, audio.currentDelay(), re, fl) + " | " + audio.dbg)
        }
        if !status.isEmpty { DispatchQueue.main.async { self.onStatus?(status) } }
        holdMs.removeAll()
        frames = 0; mids = 0; lateMids = 0; lateOrig = 0; failed = 0; droppedMids = 0
        deliveryMs.removeAll(); readyMs.removeAll(); interpMs.removeAll(); procMs.removeAll()
        lastReport = now
    }
}

// MARK: - App

/// A borderless window does not take key presses unless it says it can.
final class KeyWindow: NSWindow {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    let pipeline = Pipeline()
    var displayLink: CADisplayLink?
    let hud = NSTextField(labelWithString: "")
    var hudPinned = false
    var hudHide: DispatchWorkItem?

    /// Show a line of status. Pinned (key H) it stays; otherwise it fades after a few seconds.
    func showStatus(_ text: String, force: Bool = false) {
        guard hudPinned || force else { return }
        hud.stringValue = "  " + text + "    [1 直通  2 低延迟补帧  3 约2K补帧  H 状态  Q 退出]  "
        hud.sizeToFit()
        hud.isHidden = false
        hudHide?.cancel()
        if !hudPinned {
            let w = DispatchWorkItem { [weak self] in self?.hud.isHidden = true }
            hudHide = w
            DispatchQueue.main.asyncAfter(deadline: .now() + 4, execute: w)
        }
    }

    @objc func refreshTick(_ link: CADisplayLink) {
        pipeline.updateVsync(target: link.targetTimestamp, period: link.targetTimestamp - link.timestamp)
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        AVCaptureDevice.requestAccess(for: .video) { granted in
            guard granted else { Log.shared.line("camera permission denied"); DispatchQueue.main.async { NSApp.terminate(nil) }; return }
            let go = { DispatchQueue.main.async { self.begin() } }
            if opts.audio.isEmpty || !opts.file.isEmpty { go(); return }
            AVCaptureDevice.requestAccess(for: .audio) { ok in
                if !ok { Log.shared.line("microphone permission denied: no sound from the capture device") }
                go()
            }
        }
    }

    func begin() {
        let screens = NSScreen.screens
        let named = opts.screenName.isEmpty ? nil : screens.first { $0.localizedName.localizedCaseInsensitiveContains(opts.screenName) }
        // No screen named: take the external one with the largest panel (a television before a desk monitor),
        // and the built-in screen only when nothing else is connected.
        func displayID(_ s: NSScreen) -> CGDirectDisplayID { (s.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? CGDirectDisplayID) ?? 0 }
        func diagonal(_ s: NSScreen) -> Double { let mm = CGDisplayScreenSize(displayID(s)); return Double((mm.width * mm.width + mm.height * mm.height).squareRoot()) }
        let external = screens.filter { CGDisplayIsBuiltin(displayID($0)) == 0 }.max { diagonal($0) < diagonal($1) }
        let automatic = opts.screenExplicit && screens.indices.contains(opts.screen) ? screens[opts.screen] : (external ?? NSScreen.main ?? screens[0])
        let screen = named ?? automatic
        Log.shared.line(String(format: "screen: %@ (%.0f mm diagonal, %.0fx%.0f px, %@)", screen.localizedName, diagonal(screen),
                               screen.frame.width * screen.backingScaleFactor, screen.frame.height * screen.backingScaleFactor,
                               named != nil ? "named" : (opts.screenExplicit ? "by index" : "chosen automatically")))
        window = KeyWindow(contentRect: screen.frame, styleMask: [.borderless], backing: .buffered, defer: false, screen: screen)
        window.backgroundColor = .black
        window.title = "CaptureInterp"
        let view = NSView(frame: screen.frame)
        view.wantsLayer = true
        view.layer = CALayer()
        view.layer?.backgroundColor = NSColor.black.cgColor
        // Render at the panel's real pixels (5K on the Studio Display), not at point size.
        view.layer?.contentsScale = screen.backingScaleFactor
        pipeline.layer.contentsScale = screen.backingScaleFactor
        pipeline.layer.frame = view.bounds
        pipeline.layer.autoresizingMask = [.layerWidthSizable, .layerHeightSizable]
        view.layer?.addSublayer(pipeline.layer)
        window.contentView = view
        window.setFrame(screen.frame, display: true)
        // Above the menu bar, so the top strip of the picture is never covered when another app is in front.
        window.level = NSWindow.Level(rawValue: NSWindow.Level.mainMenu.rawValue + 1)
        window.collectionBehavior = [.canJoinAllSpaces, .stationary, .ignoresCycle]
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        hud.font = .monospacedSystemFont(ofSize: 14, weight: .medium)
        hud.textColor = .white
        hud.drawsBackground = true
        hud.backgroundColor = NSColor.black.withAlphaComponent(0.6)
        hud.isHidden = true
        hud.setFrameOrigin(NSPoint(x: 16, y: 16))
        view.addSubview(hud)
        pipeline.onStatus = { [weak self] text in
            guard let self else { return }
            let important = text.hasPrefix("模式") || text.contains("断开") || text.contains("自动直通")
            self.showStatus(text, force: important)
        }
        NSEvent.addLocalMonitorForEvents(matching: .keyDown) { e in
            switch (e.charactersIgnoringModifiers ?? "").lowercased() {
            case "q": NSApp.terminate(nil); return nil
            case "1": self.pipeline.setMode(.pass); return nil
            case "2": self.pipeline.setMode(.single); return nil
            case "3": self.pipeline.setMode(.tiled); return nil
            case "4": self.pipeline.setMode(.rifeFast); return nil
            case "5": self.pipeline.setMode(.rifeBalanced); return nil
            case "h":
                self.hudPinned.toggle()
                if self.hudPinned { self.showStatus("状态常显：" + self.pipeline.mode.title) } else { self.hud.isHidden = true }
                return nil
            default: break
            }
            if e.keyCode == 53 { NSApp.terminate(nil); return nil }
            return e
        }
        let link = screen.displayLink(target: self, selector: #selector(refreshTick(_:)))
        link.add(to: .main, forMode: .common)
        displayLink = link
        guard opts.file.isEmpty ? pipeline.start() : pipeline.startFile(opts.file) else { NSApp.terminate(nil); return }
        showStatus("模式：" + pipeline.mode.title, force: true)
        if opts.seconds > 0 {
            DispatchQueue.main.asyncAfter(deadline: .now() + opts.seconds) { NSApp.terminate(nil) }
        }
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
