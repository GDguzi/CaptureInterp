import Foundation
import CoreVideo
import Metal

/// Helpers for binding CVPixelBuffer / MTLTexture to MPSGraphTensorData.
public enum PixelBufferConvert {

    /// Creates an MTLTexture view of a CVPixelBuffer (BGRA8). Caller retains the buffer.
    public static func makeTexture(from pixelBuffer: CVPixelBuffer,
                                    textureCache: CVMetalTextureCache) throws -> MTLTexture {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        var cvTex: CVMetalTexture?
        let err = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault,
            textureCache,
            pixelBuffer,
            nil,
            .bgra8Unorm,
            width, height, 0,
            &cvTex
        )
        guard err == kCVReturnSuccess, let cvTex,
              let tex = CVMetalTextureGetTexture(cvTex) else {
            throw NSError(domain: "PixelBufferConvert", code: Int(err),
                          userInfo: [NSLocalizedDescriptionKey: "CVMetalTextureCacheCreateTextureFromImage failed"])
        }
        return tex
    }

    /// Creates zero-copy Y (.r8Unorm) and CbCr (.rg8Unorm) MTLTexture views of an NV12 (420v/420f) CVPixelBuffer.
    public static func makeNV12Textures(from pixelBuffer: CVPixelBuffer,
                                        textureCache: CVMetalTextureCache) throws -> (y: MTLTexture, uv: MTLTexture, fullRange: Bool) {
        let fmt = CVPixelBufferGetPixelFormatType(pixelBuffer)
        let isFull = (fmt == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)
        let yW = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)
        let yH = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)
        let uvW = CVPixelBufferGetWidthOfPlane(pixelBuffer, 1)
        let uvH = CVPixelBufferGetHeightOfPlane(pixelBuffer, 1)

        var cvY: CVMetalTexture?, cvUV: CVMetalTexture?
        let e0 = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, textureCache, pixelBuffer, nil, .r8Unorm, yW, yH, 0, &cvY)
        let e1 = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, textureCache, pixelBuffer, nil, .rg8Unorm, uvW, uvH, 1, &cvUV)
        guard e0 == kCVReturnSuccess, e1 == kCVReturnSuccess,
              let cvY, let cvUV,
              let yTex = CVMetalTextureGetTexture(cvY),
              let uvTex = CVMetalTextureGetTexture(cvUV) else {
            throw NSError(domain: "PixelBufferConvert", code: Int(e0 != kCVReturnSuccess ? e0 : e1),
                          userInfo: [NSLocalizedDescriptionKey: "NV12 CVMetalTextureCacheCreateTextureFromImage failed"])
        }
        return (yTex, uvTex, isFull)
    }

    /// Allocates an output CVPixelBuffer (BGRA8) of the given size, Metal-compatible.
    public static func makeOutputPixelBuffer(width: Int, height: Int) throws -> CVPixelBuffer {
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
        ]
        var buffer: CVPixelBuffer?
        let err = CVPixelBufferCreate(kCFAllocatorDefault,
                                      width, height,
                                      kCVPixelFormatType_32BGRA,
                                      attrs as CFDictionary,
                                      &buffer)
        guard err == kCVReturnSuccess, let buffer else {
            throw NSError(domain: "PixelBufferConvert", code: Int(err),
                          userInfo: [NSLocalizedDescriptionKey: "CVPixelBufferCreate failed"])
        }
        return buffer
    }
}
