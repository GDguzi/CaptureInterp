#include <metal_stdlib>
using namespace metal;

// 复现 half 加法的中点远离零舍入，避免跨后端的中点舍入差异。
inline half rifeAddHalf(half a, half b, uint tiesAway) {
    float sum = float(a) + float(b);
    if (tiesAway == 0) return half(sum);
    uint bits = as_type<uint>(sum);
    ushort sign = ushort((bits >> 16) & 0x8000u);
    uint mantissa = bits & 0x7fffffu;
    int exponent = int((bits >> 23) & 0xffu) - 127 + 15;
    ushort result;
    if (exponent >= 31) {
        result = ushort(sign | 0x7c00u | (((bits & 0x7f800000u) == 0x7f800000u && mantissa != 0) ? 0x200u : 0u));
    } else if (exponent <= 0) {
        if (exponent < -10) result = sign;
        else {
            uint shift = uint(14 - exponent);
            result = ushort(sign | ushort(((mantissa | 0x800000u) + (1u << (shift - 1))) >> shift));
        }
    } else {
        result = ushort(sign | ushort((uint(exponent) << 10) + ((mantissa + 0x1000u) >> 13)));
    }
    return as_type<half>(result);
}

kernel void rifeAccumulateFlowMaskRNA(
    device const half4* previousFlow [[buffer(0)]],
    device const half4* deltaFlow [[buffer(1)]],
    device half4* outputFlow [[buffer(2)]],
    device const half* previousMask [[buffer(3)]],
    device const half* deltaMask [[buffer(4)]],
    device half* outputMask [[buffer(5)]],
    constant uint2& dimensions [[buffer(6)]],
    constant uint2& rounding [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dimensions.x || gid.y >= dimensions.y) return;
    uint index = gid.y * dimensions.x + gid.x;
    half4 a = previousFlow[index], b = deltaFlow[index];
    outputFlow[index] = half4(rifeAddHalf(a.x,b.x,rounding.x), rifeAddHalf(a.y,b.y,rounding.x),
                             rifeAddHalf(a.z,b.z,rounding.x), rifeAddHalf(a.w,b.w,rounding.x));
    outputMask[index] = rifeAddHalf(previousMask[index], deltaMask[index], rounding.y);
}

/// Backward warp: output[x, y] = source(x + flow.x, y + flow.y) with bilinear filtering.
/// Flow is in pixel units. Boundary is clamp-to-edge (matches PyTorch grid_sample padding_mode='border').
///
/// Source and output are RGBA half-precision textures; flow is RG half-precision (only .rg used).
/// All four output lanes pass through unchanged, so this kernel handles two cases interchangeably:
///   - 3-channel image warps: caller passes RGBA texture with alpha as junk; alpha lane is ignored downstream.
///   - 4-channel feature warps (v4.25 encoder features): caller passes RGBA texture with all 4 lanes meaningful.
/// No kernel-side change is needed to switch between cases — only the texture's data semantics differ at runtime.
kernel void rifeBackwardWarp(
    texture2d<half, access::sample> source [[texture(0)]],
    texture2d<half, access::sample> flow   [[texture(1)]],
    texture2d<half, access::write>  output [[texture(2)]],
    uint2 gid                              [[thread_position_in_grid]]
) {
    const uint W = output.get_width();
    const uint H = output.get_height();
    if (gid.x >= W || gid.y >= H) return;

    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::linear);

    // flow.read returns half4; we only use rg.
    half4 fRead = flow.read(gid);
    float2 d = float2(fRead.rg);

    // Pixel-center convention: integer pixel (x, y) is sampled at coordinate (x + 0.5, y + 0.5).
    float2 srcCoord = float2(gid) + float2(0.5, 0.5) + d;

    half4 c = source.sample(s, srcCoord);
    output.write(c, gid);
}

/// Same as rifeBackwardWarp but writes to a packed RGB MTLBuffer (NHWC, drops alpha)
/// instead of an RGBA texture. Used for the FINAL full-res warps so we can feed
/// rifeBlendAndPack directly without an intermediate RGBA→RGB pack kernel.
///
/// Saves one full-res RGBA read + RGB write (~32 MB bandwidth at 4K) per warp call.
kernel void rifeBackwardWarpToBuffer(
    texture2d<half, access::sample> source [[texture(0)]],
    texture2d<half, access::sample> flow   [[texture(1)]],
    device half *output                    [[buffer(0)]],
    constant uint2 &outDim                 [[buffer(1)]],
    uint2 gid                              [[thread_position_in_grid]]
) {
    if (gid.x >= outDim.x || gid.y >= outDim.y) return;
    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::linear);
    half4 fRead = flow.read(gid);
    float2 d = float2(fRead.rg);
    float2 srcCoord = float2(gid) + float2(0.5, 0.5) + d;
    half4 c = source.sample(s, srcCoord);
    uint base = (gid.y * outDim.x + gid.x) * 3;
    output[base + 0] = c.r;
    output[base + 1] = c.g;
    output[base + 2] = c.b;
}

/// BGRA8 texture → packed fp16 RGB buffer (NHWC, drops alpha, divides by 255).
kernel void rifeBGRAToRGB(
    texture2d<half, access::read> input  [[texture(0)]],
    device half *output                  [[buffer(0)]],
    uint2 gid                            [[thread_position_in_grid]]
) {
    uint W = input.get_width(), H = input.get_height();
    if (gid.x >= W || gid.y >= H) return;
    half4 c = input.read(gid);
    uint outIdx = (gid.y * W + gid.x) * 3;
    output[outIdx + 0] = c.r;
    output[outIdx + 1] = c.g;
    output[outIdx + 2] = c.b;
}

/// Fused BGRA8 texture → packed fp16 RGB buffer at internal (downsampled) resolution.
///
/// Replaces the two-pass (bgraToRGB → MPSGraph bilinear downsample) path used by the
/// balanced tier so 4K input traverses memory once instead of twice. The hardware linear
/// sampler matches MPSGraph's bilinear resize (centerResult: true, alignCorners: false)
/// up to fp16 rounding — empirically PSNR drift is < 0.1 dB.
///
/// gid spans the *output* (downsampled) grid. invScale = 1.0 / internalScale (e.g. 2.0
/// for balanced). Output buffer layout: [1, outH, outW, 3] NHWC fp16.
kernel void rifeBGRADownsampleToRGB(
    texture2d<half, access::sample> input  [[texture(0)]],
    device half *output                    [[buffer(0)]],
    constant float &invScale               [[buffer(1)]],
    constant uint2 &outDim                 [[buffer(2)]],
    uint2 gid                              [[thread_position_in_grid]]
) {
    if (gid.x >= outDim.x || gid.y >= outDim.y) return;
    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::linear);
    float2 center = (float2(gid) + float2(0.5f)) * invScale;
    half3 rgb;
    if (invScale <= 2.01f) {
        // invScale == 2: center lands on exact 2x2 texel corner -> single bilinear tap averages all 4 pixels.
        rgb = input.sample(s, center).rgb;
    } else {
        // invScale == 3 (off = 2/3) -> exact 3x3 9-pixel uniform box average in 4 bilinear taps;
        // invScale == 4 (off = 1.0) -> exact 4x4 16-pixel uniform box average in 4 bilinear taps.
        float off = (invScale - 1.0f) * (1.0f / 3.0f);
        float3 acc = float3(input.sample(s, center + float2(-off, -off)).rgb)
                   + float3(input.sample(s, center + float2( off, -off)).rgb)
                   + float3(input.sample(s, center + float2(-off,  off)).rgb)
                   + float3(input.sample(s, center + float2( off,  off)).rgb);
        rgb = half3(acc * 0.25f);
    }
    uint outIdx = (gid.y * outDim.x + gid.x) * 3;
    output[outIdx + 0] = rgb.r;
    output[outIdx + 1] = rgb.g;
    output[outIdx + 2] = rgb.b;
}

/// Packed fp16 RGB buffer (NHWC) → BGRA8 texture (alpha=1).
kernel void rifeRGBToBGRA(
    device const half *input              [[buffer(0)]],
    texture2d<half, access::write> output [[texture(0)]],
    uint2 gid                             [[thread_position_in_grid]]
) {
    uint W = output.get_width(), H = output.get_height();
    if (gid.x >= W || gid.y >= H) return;
    uint inIdx = (gid.y * W + gid.x) * 3;
    half4 c;
    c.r = input[inIdx + 0];
    c.g = input[inIdx + 1];
    c.b = input[inIdx + 2];
    c.a = 1.0h;
    output.write(c, gid);
}

/// Fused final blend + RGB→BGRA pack.
///
/// out = w0 * sigmoid(mask) + w1 * (1 - sigmoid(mask))   then write as BGRA8 (alpha=1).
///
/// Replaces the legacy (MPSGraph blend executable + rifeRGBToBGRA) pair with a single
/// full-res pass. Math is identical (sigmoid + linear blend). At 4K this saves one full-res
/// RGB read/write (~24 MB bandwidth) and one MPSGraph encode.
kernel void rifeBlendAndPack(
    device const half *warped0  [[buffer(0)]],
    device const half *warped1  [[buffer(1)]],
    device const half *mask     [[buffer(2)]],   // pre-sigmoid logits
    texture2d<half, access::write> output [[texture(0)]],
    uint2 gid                   [[thread_position_in_grid]]
) {
    uint W = output.get_width(), H = output.get_height();
    if (gid.x >= W || gid.y >= H) return;

    uint flatIdx = gid.y * W + gid.x;
    uint rgbBase = flatIdx * 3;

    half3 w0 = half3(warped0[rgbBase + 0], warped0[rgbBase + 1], warped0[rgbBase + 2]);
    half3 w1 = half3(warped1[rgbBase + 0], warped1[rgbBase + 1], warped1[rgbBase + 2]);

    half m = mask[flatIdx];
    // sigmoid(x) = 1 / (1 + exp(-x))
    half mSig = 1.0h / (1.0h + exp(-m));

    half3 blended = w0 * mSig + w1 * (1.0h - mSig);

    output.write(half4(blended.r, blended.g, blended.b, 1.0h), gid);
}

/// Same as rifeBlendAndPack but bilinear-samples the mask from an internal-res buffer
/// instead of reading at the output (full-res) grid. Used for balanced tier so we can
/// skip the upsampleMaskExecutable MPSGraph encode entirely. The bilinear math matches
/// MPSGraph.resize(centerResult: true, alignCorners: false).
kernel void rifeBlendUpsampleMaskAndPack(
    device const half *warped0       [[buffer(0)]],
    device const half *warped1       [[buffer(1)]],
    device const half *maskInternal  [[buffer(2)]],   // pre-sigmoid logits, [1, iH, iW, 1]
    constant uint2 &internalDim      [[buffer(3)]],   // (iW, iH)
    texture2d<half, access::write> output [[texture(0)]],
    uint2 gid                        [[thread_position_in_grid]]
) {
    uint W = output.get_width(), H = output.get_height();
    if (gid.x >= W || gid.y >= H) return;

    uint flatIdx = gid.y * W + gid.x;
    uint rgbBase = flatIdx * 3;

    half3 w0 = half3(warped0[rgbBase + 0], warped0[rgbBase + 1], warped0[rgbBase + 2]);
    half3 w1 = half3(warped1[rgbBase + 0], warped1[rgbBase + 1], warped1[rgbBase + 2]);

    int iW = int(internalDim.x);
    int iH = int(internalDim.y);
    float u = (float(gid.x) + 0.5) * float(iW) / float(W) - 0.5;
    float v = (float(gid.y) + 0.5) * float(iH) / float(H) - 0.5;
    int u0 = int(floor(u));
    int v0 = int(floor(v));
    float du = u - float(u0);
    float dv = v - float(v0);
    int u1 = min(u0 + 1, iW - 1);
    int v1 = min(v0 + 1, iH - 1);
    u0 = max(u0, 0);
    v0 = max(v0, 0);

    float m00 = float(maskInternal[v0 * iW + u0]);
    float m01 = float(maskInternal[v0 * iW + u1]);
    float m10 = float(maskInternal[v1 * iW + u0]);
    float m11 = float(maskInternal[v1 * iW + u1]);
    float m = mix(mix(m00, m01, du), mix(m10, m11, du), dv);

    half mh = half(m);
    half mSig = 1.0h / (1.0h + exp(-mh));

    half3 blended = w0 * mSig + w1 * (1.0h - mSig);
    output.write(half4(blended.r, blended.g, blended.b, 1.0h), gid);
}

// 裁剪只限制输出区域，张量索引和 mask 重采样仍使用 padded 网格。
kernel void rifeBlendUpsampleMaskAndPackCropped(
    device const half *warped0       [[buffer(0)]],
    device const half *warped1       [[buffer(1)]],
    device const half *maskInternal  [[buffer(2)]],   // pre-sigmoid logits, [1, iH, iW, 1]
    constant uint2 &internalDim      [[buffer(3)]],   // (iW, iH)
    constant uint2 &sourceDim [[buffer(4)]],
    texture2d<half, access::write> output [[texture(0)]],
    uint2 gid                        [[thread_position_in_grid]]
) {
    uint W = sourceDim.x, H = sourceDim.y;
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;

    uint flatIdx = gid.y * W + gid.x;
    uint rgbBase = flatIdx * 3;

    half3 w0 = half3(warped0[rgbBase + 0], warped0[rgbBase + 1], warped0[rgbBase + 2]);
    half3 w1 = half3(warped1[rgbBase + 0], warped1[rgbBase + 1], warped1[rgbBase + 2]);

    int iW = int(internalDim.x);
    int iH = int(internalDim.y);
    float u = (float(gid.x) + 0.5) * float(iW) / float(W) - 0.5;
    float v = (float(gid.y) + 0.5) * float(iH) / float(H) - 0.5;
    int u0 = int(floor(u));
    int v0 = int(floor(v));
    float du = u - float(u0);
    float dv = v - float(v0);
    int u1 = min(u0 + 1, iW - 1);
    int v1 = min(v0 + 1, iH - 1);
    u0 = max(u0, 0);
    v0 = max(v0, 0);

    float m00 = float(maskInternal[v0 * iW + u0]);
    float m01 = float(maskInternal[v0 * iW + u1]);
    float m10 = float(maskInternal[v1 * iW + u0]);
    float m11 = float(maskInternal[v1 * iW + u1]);
    float m = mix(mix(m00, m01, du), mix(m10, m11, du), dv);

    half mh = half(m);
    half mSig = 1.0h / (1.0h + exp(-mh));

    half3 blended = w0 * mSig + w1 * (1.0h - mSig);
    output.write(half4(blended.r, blended.g, blended.b, 1.0h), gid);
}

/// Splits a 4-channel fp16 NHWC flow buffer into two RG fp16 textures.
/// Input: device const half *flow (W*H*4 elements, NHWC interleaved).
/// Outputs: rg16Float textures, [0:2] for flow0, [2:4] for flow1.
kernel void rifeFlowSplit(
    device const half *flow                  [[buffer(0)]],
    texture2d<half, access::write> flow0     [[texture(0)]],
    texture2d<half, access::write> flow1     [[texture(1)]],
    uint2 gid                                [[thread_position_in_grid]]
) {
    uint W = flow0.get_width(), H = flow0.get_height();
    if (gid.x >= W || gid.y >= H) return;
    uint base = (gid.y * W + gid.x) * 4;
    flow0.write(half4(flow[base + 0], flow[base + 1], 0, 0), gid);
    flow1.write(half4(flow[base + 2], flow[base + 3], 0, 0), gid);
}

/// Fused bilinear upsample + flow split for the FINAL flow tensor (balanced tier only).
///
/// Reads the internal-resolution flow buffer [1, iH, iW, 4], bilinear-resamples it onto
/// the full-resolution grid, multiplies displacement values by `valueScale` (1/internalScale
/// so flow tracks the new resolution), and writes channels [0:2] / [2:4] to two RG fp16
/// textures. This replaces the (upsampleFlowExecutable + rifeFlowSplit) pair — saves one
/// MPSGraph encoder switch and one full-res buffer round-trip.
///
/// Bilinear math matches MPSGraph.resize(centerResult: true, alignCorners: false): output
/// pixel (gx, gy) maps to source coord (gx + 0.5)*iW/W - 0.5 etc. Boundary handling clamps
/// to [0, dim-1] (matches MPSGraph default). Drift vs the MPSGraph executable is fp16
/// rounding only — empirically PSNR delta is < 0.05 dB.
kernel void rifeFlowUpsampleSplit(
    device const half *flowInternal               [[buffer(0)]],
    constant uint2 &internalDim                   [[buffer(1)]],
    constant float &valueScale                    [[buffer(2)]],
    texture2d<half, access::write> flow0Full      [[texture(0)]],
    texture2d<half, access::write> flow1Full      [[texture(1)]],
    uint2 gid                                     [[thread_position_in_grid]]
) {
    uint W = flow0Full.get_width(), H = flow0Full.get_height();
    if (gid.x >= W || gid.y >= H) return;

    int iW = int(internalDim.x);
    int iH = int(internalDim.y);

    // Map output pixel center to source space, then subtract 0.5 to get source pixel index.
    // (Matches MPSGraph centerResult:true, alignCorners:false.)
    float u = (float(gid.x) + 0.5) * float(iW) / float(W) - 0.5;
    float v = (float(gid.y) + 0.5) * float(iH) / float(H) - 0.5;

    int u0 = int(floor(u));
    int v0 = int(floor(v));
    float du = u - float(u0);
    float dv = v - float(v0);
    int u1 = min(u0 + 1, iW - 1);
    int v1 = min(v0 + 1, iH - 1);
    u0 = max(u0, 0);
    v0 = max(v0, 0);

    uint base00 = uint(v0 * iW + u0) * 4u;
    uint base01 = uint(v0 * iW + u1) * 4u;
    uint base10 = uint(v1 * iW + u0) * 4u;
    uint base11 = uint(v1 * iW + u1) * 4u;
    float4 v00 = float4(float(flowInternal[base00 + 0]), float(flowInternal[base00 + 1]),
                        float(flowInternal[base00 + 2]), float(flowInternal[base00 + 3]));
    float4 v01 = float4(float(flowInternal[base01 + 0]), float(flowInternal[base01 + 1]),
                        float(flowInternal[base01 + 2]), float(flowInternal[base01 + 3]));
    float4 v10 = float4(float(flowInternal[base10 + 0]), float(flowInternal[base10 + 1]),
                        float(flowInternal[base10 + 2]), float(flowInternal[base10 + 3]));
    float4 v11 = float4(float(flowInternal[base11 + 0]), float(flowInternal[base11 + 1]),
                        float(flowInternal[base11 + 2]), float(flowInternal[base11 + 3]));
    float4 t = mix(mix(v00, v01, du), mix(v10, v11, du), dv);
    t *= valueScale;

    half4 th = half4(t);
    flow0Full.write(half4(th.x, th.y, 0, 0), gid);
    flow1Full.write(half4(th.z, th.w, 0, 0), gid);
}

/// Expands a packed fp16 RGB buffer (NHWC) → RGBA16Float texture (alpha=1).
/// Used to make warp-source textures from MPSGraph-side RGB tensors without a CPU round-trip.
kernel void rifeRGBBufToRGBATex(
    device const half *input              [[buffer(0)]],
    texture2d<half, access::write> output [[texture(0)]],
    uint2 gid                             [[thread_position_in_grid]]
) {
    uint W = output.get_width(), H = output.get_height();
    if (gid.x >= W || gid.y >= H) return;
    uint inIdx = (gid.y * W + gid.x) * 3;
    half4 c;
    c.r = input[inIdx + 0];
    c.g = input[inIdx + 1];
    c.b = input[inIdx + 2];
    c.a = 1.0h;
    output.write(c, gid);
}

/// Expands a packed fp16 4-channel buffer (NHWC) → RGBA16Float texture.
/// Used to make warp-source textures from v4.25 encoder feature buffers without a CPU round-trip.
/// Unlike rifeRGBBufToRGBATex, all four channels are meaningful — alpha is not forced to 1.
kernel void rife4ChBufToRGBATex(
    device const half *input              [[buffer(0)]],
    texture2d<half, access::write> output [[texture(0)]],
    uint2 gid                             [[thread_position_in_grid]]
) {
    uint W = output.get_width(), H = output.get_height();
    if (gid.x >= W || gid.y >= H) return;
    uint inIdx = (gid.y * W + gid.x) * 4;
    output.write(half4(input[inIdx], input[inIdx+1], input[inIdx+2], input[inIdx+3]), gid);
}

/// Packs an RGBA16Float texture into a fp16 4-channel MTLBuffer (NHWC), preserving all 4 channels.
/// Used to feed v4.25 warped feature textures back to MPSGraph as 4-ch tensors without a CPU round-trip.
kernel void rifeRGBATexTo4ChBuf(
    texture2d<half, access::read> input [[texture(0)]],
    device half *output                  [[buffer(0)]],
    uint2 gid                            [[thread_position_in_grid]]
) {
    uint W = input.get_width(), H = input.get_height();
    if (gid.x >= W || gid.y >= H) return;
    half4 c = input.read(gid);
    uint outIdx = (gid.y * W + gid.x) * 4;
    output[outIdx + 0] = c.r;
    output[outIdx + 1] = c.g;
    output[outIdx + 2] = c.b;
    output[outIdx + 3] = c.a;
}

/// Drops alpha from an RGBA16Float texture and packs into an fp16 RGB MTLBuffer (NHWC).
/// Used to feed warp results back to MPSGraph as RGB tensors without a CPU round-trip.
kernel void rifeRGBATexToRGBBuf(
    texture2d<half, access::read> input [[texture(0)]],
    device half *output                  [[buffer(0)]],
    uint2 gid                            [[thread_position_in_grid]]
) {
    uint W = input.get_width(), H = input.get_height();
    if (gid.x >= W || gid.y >= H) return;
    half4 c = input.read(gid);
    uint outIdx = (gid.y * W + gid.x) * 3;
    output[outIdx + 0] = c.r;
    output[outIdx + 1] = c.g;
    output[outIdx + 2] = c.b;
}

inline float rifeBt709Luma(float3 c) {
    return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
}

struct CatmullSample {
    float3 cr;
    float3 lin;
    float3 lo;
    float3 hi;
};

/// 5-bilinear-tap Catmull-Rom bicubic sampler with strict local 2x2 neighborhood min/max anti-ringing clamp.
/// Preserves sub-pixel 4K micro-texture during motion warps with zero white/black overshoot halos.
inline CatmullSample sampleCatmullRom5Tap(
    texture2d<half, access::sample> tex,
    sampler linSampler,
    float2 coord
) {
    int texW = int(tex.get_width()), texH = int(tex.get_height());
    if (!isfinite(coord.x) || !isfinite(coord.y)) {
        coord = float2(0.5f, 0.5f);
    }
    float2 samplePos = coord - 0.5f;
    float2 texPos1 = floor(samplePos);
    float2 f = samplePos - texPos1;

    float2 w0 = f * (-0.5f + f * (1.0f - 0.5f * f));
    float2 w1 = 1.0f + f * f * (-2.5f + 1.5f * f);
    float2 w2 = f * (0.5f + f * (2.0f - 1.5f * f));
    float2 w3 = f * f * (-0.5f + 0.5f * f);

    float2 w12 = w1 + w2;
    float2 offset12 = w2 / w12;

    float2 tc0  = texPos1 - 0.5f;
    float2 tc3  = texPos1 + 2.5f;
    float2 tc12 = texPos1 + 0.5f + offset12;

    float cw12    = w12.x * w12.y;
    float cwTop   = w12.x * w0.y;
    float cwLeft  = w0.x  * w12.y;
    float cwRight = w3.x  * w12.y;
    float cwBot   = w12.x * w3.y;
    float wSum    = cw12 + cwTop + cwLeft + cwRight + cwBot;

    float3 cMid   = float3(tex.sample(linSampler, tc12).rgb);
    float3 cTop   = float3(tex.sample(linSampler, float2(tc12.x, tc0.y)).rgb);
    float3 cLeft  = float3(tex.sample(linSampler, float2(tc0.x, tc12.y)).rgb);
    float3 cRight = float3(tex.sample(linSampler, float2(tc3.x, tc12.y)).rgb);
    float3 cBot   = float3(tex.sample(linSampler, float2(tc12.x, tc3.y)).rgb);
    float3 crRaw  = (cMid * cw12 + cTop * cwTop + cLeft * cwLeft + cRight * cwRight + cBot * cwBot) / wSum;

    int x0 = clamp(int(texPos1.x),     0, texW - 1);
    int x1 = clamp(int(texPos1.x) + 1, 0, texW - 1);
    int y0 = clamp(int(texPos1.y),     0, texH - 1);
    int y1 = clamp(int(texPos1.y) + 1, 0, texH - 1);

    float3 p00 = float3(tex.read(uint2(uint(x0), uint(y0))).rgb);
    float3 p10 = float3(tex.read(uint2(uint(x1), uint(y0))).rgb);
    float3 p01 = float3(tex.read(uint2(uint(x0), uint(y1))).rgb);
    float3 p11 = float3(tex.read(uint2(uint(x1), uint(y1))).rgb);

    float3 lo = min(min(p00, p10), min(p01, p11));
    float3 hi = max(max(p00, p10), max(p01, p11));
    float3 lin = mix(mix(p00, p10, f.x), mix(p01, p11, f.x), f.y);

    CatmullSample out;
    out.cr  = clamp(crRaw, lo, hi);
    out.lin = lin;
    out.lo  = lo;
    out.hi  = hi;
    return out;
}

/// Single-pass fused 2x2 edge-guided flow/mask upsample + backward warp + blend + optional UI/subtitle
/// still-pixel protection + cropped BGRA writeout.
kernel void rifeEdgeGuidedWarpBlendCropped(
    texture2d<half, access::sample> src0          [[texture(0)]],
    texture2d<half, access::sample> src1          [[texture(1)]],
    device const half *flowInternal               [[buffer(0)]],
    device const half *maskInternal               [[buffer(1)]],
    constant uint2 &internalDim                   [[buffer(2)]],
    constant float &valueScale                    [[buffer(3)]],
    constant uint2 &sourceDim                     [[buffer(4)]],
    constant uint  &protectStillFlag              [[buffer(5)]],
    texture2d<half, access::write> output         [[texture(2)]],
    uint2 gid                                     [[thread_position_in_grid]]
) {
    uint outW = output.get_width(), outH = output.get_height();
    if (gid.x >= outW || gid.y >= outH) return;

    float stillBlend = 0.0f;
    half3 curExact = half3(0.0h);
    if (protectStillFlag != 0u && src0.get_width() >= outW && src0.get_height() >= outH &&
        src1.get_width() >= outW && src1.get_height() >= outH) {
        const float tol = 3.5f / 255.0f;
        float3 curC = float3(src1.read(gid).rgb);
        float3 d0 = abs(float3(src0.read(gid).rgb) - curC);
        if (max(d0.r, max(d0.g, d0.b)) <= tol) {
            int w = int(outW), h = int(outH);
            int stillCount = 0;
            float minY = 1.0f, maxY = 0.0f;
            for (int dy = -1; dy <= 1; dy++) {
                uint py = uint(clamp(int(gid.y) + dy, 0, h - 1));
                for (int dx = -2; dx <= 2; dx++) {
                    uint2 p = uint2(uint(clamp(int(gid.x) + dx, 0, w - 1)), py);
                    float3 cp = float3(src1.read(p).rgb);
                    float3 dp = abs(float3(src0.read(p).rgb) - cp);
                    if (max(dp.r, max(dp.g, dp.b)) <= tol) stillCount++;
                    float y = rifeBt709Luma(cp);
                    minY = min(minY, y);
                    maxY = max(maxY, y);
                }
            }
            if (stillCount >= 13) {
                const int2 arms[4] = { int2(-5, 0), int2(5, 0), int2(0, -5), int2(0, 5) };
                for (int k = 0; k < 4; k++) {
                    uint2 p = uint2(uint(clamp(int(gid.x) + arms[k].x, 0, w - 1)),
                                    uint(clamp(int(gid.y) + arms[k].y, 0, h - 1)));
                    float y = rifeBt709Luma(float3(src1.read(p).rgb));
                    minY = min(minY, y);
                    maxY = max(maxY, y);
                }
                float contrast = maxY - minY;
                if (contrast >= (28.0f / 255.0f)) {
                    stillBlend = clamp((contrast - 28.0f / 255.0f) / (20.0f / 255.0f), 0.0f, 1.0f);
                    curExact = half3(curC);
                    if (stillBlend >= 0.999f) {
                        output.write(half4(curExact, 1.0h), gid);
                        return;
                    }
                }
            }
        }
    }

    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::linear);
    uint W = sourceDim.x, H = sourceDim.y;
    int iW = int(internalDim.x), iH = int(internalDim.y);

    float u = (float(gid.x) + 0.5f) * float(iW) / float(W) - 0.5f;
    float v = (float(gid.y) + 0.5f) * float(iH) / float(H) - 0.5f;
    int u0 = int(floor(u)), v0 = int(floor(v));
    float du = u - float(u0), dv = v - float(v0);
    int u1 = min(u0 + 1, iW - 1), v1 = min(v0 + 1, iH - 1);
    u0 = max(u0, 0); v0 = max(v0, 0);

    uint idx00 = uint(v0 * iW + u0), idx01 = uint(v0 * iW + u1);
    uint idx10 = uint(v1 * iW + u0), idx11 = uint(v1 * iW + u1);
    uint b00 = idx00 * 4u, b01 = idx01 * 4u, b10 = idx10 * 4u, b11 = idx11 * 4u;

    float4 f00 = float4(float(flowInternal[b00+0]), float(flowInternal[b00+1]), float(flowInternal[b00+2]), float(flowInternal[b00+3])) * valueScale;
    float4 f01 = float4(float(flowInternal[b01+0]), float(flowInternal[b01+1]), float(flowInternal[b01+2]), float(flowInternal[b01+3])) * valueScale;
    float4 f10 = float4(float(flowInternal[b10+0]), float(flowInternal[b10+1]), float(flowInternal[b10+2]), float(flowInternal[b10+3])) * valueScale;
    float4 f11 = float4(float(flowInternal[b11+0]), float(flowInternal[b11+1]), float(flowInternal[b11+2]), float(flowInternal[b11+3])) * valueScale;

    float m00 = float(maskInternal[idx00]), m01 = float(maskInternal[idx01]);
    float m10 = float(maskInternal[idx10]), m11 = float(maskInternal[idx11]);

    float sw00 = (1.0f - du) * (1.0f - dv), sw01 = du * (1.0f - dv);
    float sw10 = (1.0f - du) * dv,          sw11 = du * dv;

    float4 fFinal = f00 * sw00 + f01 * sw01 + f10 * sw10 + f11 * sw11;
    float  mFinal = m00 * sw00 + m01 * sw01 + m10 * sw10 + m11 * sw11;

    float2 center = float2(gid) + float2(0.5f, 0.5f);

    float4 dDiagA = abs(f00 - f11), dDiagB = abs(f01 - f10);
    float spread = max(max(dDiagA.x + dDiagA.y, dDiagA.z + dDiagA.w),
                       max(dDiagB.x + dDiagB.y, dDiagB.z + dDiagB.w));

    if (spread >= 3.0f) {
        float4 candsF[4] = { f00, f01, f10, f11 };
        float  candsM[4] = { m00, m01, m10, m11 };
        float  spatW[4]  = { sw00 + 0.04f, sw01 + 0.04f, sw10 + 0.04f, sw11 + 0.04f };
        float  photoErr[4];
        float  minErr = 1e9f;
        for (int k = 0; k < 4; k++) {
            half3 c0 = src0.sample(s, center + candsF[k].xy).rgb;
            half3 c1 = src1.sample(s, center + candsF[k].zw).rgb;
            float e = float(abs(c0.r - c1.r) + abs(c0.g - c1.g) + abs(c0.b - c1.b));
            photoErr[k] = e;
            minErr = min(minErr, e);
        }
        float sumW = 0.0f;
        float4 accF = float4(0.0f);
        float  accM = 0.0f;
        for (int k = 0; k < 4; k++) {
            float pw = exp(-clamp((photoErr[k] - minErr) * 14.0f, 0.0f, 12.0f));
            float w = spatW[k] * pw;
            accF += candsF[k] * w;
            accM += candsM[k] * w;
            sumW += w;
        }
        if (sumW > 1e-6f) {
            fFinal = accF / sumW;
            mFinal = accM / sumW;
        }
    }

    CatmullSample s0 = sampleCatmullRom5Tap(src0, s, center + fFinal.xy);
    CatmullSample s1 = sampleCatmullRom5Tap(src1, s, center + fFinal.zw);
    float m = 1.0f / (1.0f + exp(-mFinal));
    float3 blendedF = s0.cr * m + s1.cr * (1.0f - m);

    // Recover fine micro-texture when sub-pixel flow phase difference partially cancels grass/brick ripples
    float3 hf0 = s0.cr - s0.lin;
    float3 hf1 = s1.cr - s1.lin;
    if (dot(hf0, hf1) < 0.0f) {
        float3 hfPick = (dot(hf0, hf0) >= dot(hf1, hf1)) ? hf0 : hf1;
        blendedF = clamp(blendedF + hfPick * 0.45f, min(s0.lo, s1.lo), max(s0.hi, s1.hi));
    }

    half3 blended = half3(blendedF);
    if (stillBlend > 0.0f) {
        blended = mix(blended, curExact, half(stillBlend));
    }
    output.write(half4(blended.r, blended.g, blended.b, 1.0h), gid);
}

inline float nv12NormY(float yRaw, uint fullRange) {
    return (fullRange != 0u) ? yRaw : clamp(yRaw * (255.0f / 219.0f) - (16.0f / 219.0f), 0.0f, 1.0f);
}

/// Zero-copy NV12 (bi-planar 8-bit 4:2:0, ITU-R BT.709 video-range or full-range, Left chroma siting)
/// to BGRA8 texture conversion + optional Contrast-Adaptive Sharpening (CAS) for 4K->5K panel scaling.
kernel void rifeNV12ToBGRA(
    texture2d<float, access::read>   yTex       [[texture(0)]],
    texture2d<float, access::sample> uvTex      [[texture(1)]],
    texture2d<half,  access::write>  outBGRA    [[texture(2)]],
    constant uint  &fullRange                   [[buffer(0)]],
    constant float &casStrength                 [[buffer(1)]],
    uint2 gid                                   [[thread_position_in_grid]]
) {
    uint W = outBGRA.get_width(), H = outBGRA.get_height();
    if (gid.x >= W || gid.y >= H) return;

    constexpr sampler chromaSampler(coord::pixel, address::clamp_to_edge, filter::linear);
    // MPEG-2 / H.264 / BT.709 4:2:0 Left chroma siting: horizontally co-sited at even x, vertically centered
    float2 uvCoord = float2(0.5f * float(gid.x) + 0.5f, 0.5f * float(gid.y) + 0.25f);
    float y = nv12NormY(yTex.read(gid).r, fullRange);

    if (casStrength > 0.001f) {
        float yN  = nv12NormY(yTex.read(uint2(gid.x, max(gid.y, 1u) - 1u)).r, fullRange);
        float yS  = nv12NormY(yTex.read(uint2(gid.x, min(gid.y + 1u, H - 1u))).r, fullRange);
        float yW  = nv12NormY(yTex.read(uint2(max(gid.x, 1u) - 1u, gid.y)).r, fullRange);
        float yEa = nv12NormY(yTex.read(uint2(min(gid.x + 1u, W - 1u), gid.y)).r, fullRange);
        float mnY = min(y, min(min(yN, yS), min(yW, yEa)));
        float mxY = max(y, max(max(yN, yS), max(yW, yEa)));
        float amp = sqrt(clamp(min(mnY, 1.0f - mxY) / max(mxY, 1e-4f), 0.0f, 1.0f));
        float wLobe = -amp / mix(8.0f, 4.5f, clamp(casStrength, 0.0f, 1.0f));
        y = clamp((y + (yN + yS + yW + yEa) * wLobe) / (1.0f + 4.0f * wLobe), mnY, mxY);
    }

    float2 uvRaw = uvTex.sample(chromaSampler, uvCoord).rg;
    float cb, cr;
    if (fullRange != 0u) {
        cb = uvRaw.x - (128.0f / 255.0f);
        cr = uvRaw.y - (128.0f / 255.0f);
    } else {
        cb = uvRaw.x * (255.0f / 224.0f) - (128.0f / 224.0f);
        cr = uvRaw.y * (255.0f / 224.0f) - (128.0f / 224.0f);
    }

    // ITU-R BT.709 YCbCr -> RGB
    float3 rgb;
    rgb.r = y + 1.5748f * cr;
    rgb.g = y - 0.187324f * cb - 0.468124f * cr;
    rgb.b = y + 1.8556f * cb;
    rgb = clamp(rgb, 0.0f, 1.0f);
    outBGRA.write(half4(half3(rgb), 1.0h), gid);
}
