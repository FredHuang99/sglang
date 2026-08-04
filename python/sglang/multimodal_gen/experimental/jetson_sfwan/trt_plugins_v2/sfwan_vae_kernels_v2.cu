// SPDX-License-Identifier: Apache-2.0

#include "sfwan_vae_kernels_v2.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace
{

__device__ __forceinline__ int64_t linearOffset(int32_t n, int32_t c,
    int32_t d, int32_t h, int32_t w, int32_t channels, int32_t depth,
    int32_t height, int32_t width)
{
    return (((static_cast<int64_t>(n) * channels + c) * depth + d) * height
               + h)
        * width
        + w;
}

__device__ __forceinline__ int64_t cdhw32Offset(int32_t n, int32_t c,
    int32_t d, int32_t h, int32_t w, int32_t channels, int32_t depth,
    int32_t height, int32_t width)
{
    int32_t const channelBlocks = (channels + 31) / 32;
    return (((((static_cast<int64_t>(n) * channelBlocks + c / 32) * depth + d)
                   * height
               + h)
                  * width
              + w)
                * 32)
        + c % 32;
}

__device__ __forceinline__ int8_t quantizeSigned(float value, float invScale)
{
    int32_t result = __float2int_rn(value * invScale);
    result = result < -127 ? -127 : (result > 127 ? 127 : result);
    return static_cast<int8_t>(result);
}

__device__ __forceinline__ float sourceValue(int32_t mode, void const* input0,
    void const* residual, float consumeScale, int32_t n, int32_t c,
    int32_t d, int32_t h, int32_t w, int32_t channels, int32_t depth,
    int32_t height, int32_t width)
{
    if (mode == 0)
    {
        auto const* values = static_cast<half const*>(input0);
        return __half2float(values[linearOffset(
            n, c, d, h, w, channels, depth, height, width)]);
    }
    auto const* values = static_cast<int8_t const*>(input0);
    float result = static_cast<float>(values[cdhw32Offset(
                       n, c, d, h, w, channels, depth, height, width)])
        * consumeScale;
    if (mode == 2)
    {
        auto const* residualValues = static_cast<half const*>(residual);
        result += __half2float(residualValues[linearOffset(
            n, c, d, h, w, channels, depth, height, width)]);
    }
    return result;
}

__global__ void copyCacheHistoryKernel(int8_t const* cache, int8_t* packed,
    int8_t* cacheOutput, int32_t nSize, int32_t channels,
    int32_t cacheDepth, int32_t currentDepth, int32_t height, int32_t width,
    int32_t packedDepth, int32_t packedHeight, int32_t packedWidth,
    int32_t depthFront, int32_t heightTop, int32_t widthLeft,
    int32_t cacheOutputDepth)
{
    int64_t const count
        = static_cast<int64_t>(nSize) * channels * cacheDepth * height * width;
    int32_t const retainedFrom
        = cacheDepth + currentDepth - cacheOutputDepth;
    for (int64_t index
         = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x)
    {
        int64_t value = index;
        int32_t const w = value % width;
        value /= width;
        int32_t const h = value % height;
        value /= height;
        int32_t const d = value % cacheDepth;
        value /= cacheDepth;
        int32_t const c = value % channels;
        int32_t const n = value / channels;
        int8_t const q = cache[cdhw32Offset(
            n, c, d, h, w, channels, cacheDepth, height, width)];
        packed[cdhw32Offset(n, c, depthFront + d, heightTop + h,
            widthLeft + w, channels, packedDepth, packedHeight, packedWidth)]
            = q;
        if (d >= retainedFrom && d - retainedFrom < cacheOutputDepth)
        {
            cacheOutput[cdhw32Offset(n, c, d - retainedFrom, h, w,
                channels, cacheOutputDepth, height, width)] = q;
        }
    }
}

__global__ void normSiluQuantPackKernel(int32_t mode, void const* input0,
    half const* residual, half const* gamma, int8_t* packed,
    int8_t* cacheOutput, half* residualOutput, float consumeScale,
    float invProduceScale, int32_t nSize, int32_t channels,
    int32_t currentDepth, int32_t height, int32_t width,
    int32_t cacheDepth, int32_t cacheOutputDepth, int32_t packedDepth,
    int32_t packedHeight, int32_t packedWidth, int32_t depthFront,
    int32_t heightTop, int32_t widthLeft)
{
    int32_t constexpr kWarpsPerBlock = 4;
    int32_t const lane = threadIdx.x & 31;
    int32_t const warp = threadIdx.x >> 5;
    int64_t const site
        = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    int64_t const siteCount
        = static_cast<int64_t>(nSize) * currentDepth * height * width;
    if (site >= siteCount)
    {
        return;
    }
    int64_t value = site;
    int32_t const w = value % width;
    value /= width;
    int32_t const h = value % height;
    value /= height;
    int32_t const d = value % currentDepth;
    int32_t const n = value / currentDepth;

    float sumSquares = 0.0F;
    for (int32_t c = lane; c < channels; c += 32)
    {
        float const x = sourceValue(mode, input0, residual, consumeScale, n,
            c, d, h, w, channels, currentDepth, height, width);
        sumSquares += x * x;
        if (residualOutput != nullptr)
        {
            residualOutput[linearOffset(n, c, d, h, w, channels,
                currentDepth, height, width)] = __float2half_rn(x);
        }
    }
    unsigned int const mask = __activemask();
    for (int32_t offset = 16; offset > 0; offset /= 2)
    {
        sumSquares += __shfl_down_sync(mask, sumSquares, offset);
    }
    float const totalSquares = __shfl_sync(mask, sumSquares, 0);
    float const factor = sqrtf(static_cast<float>(channels))
        / fmaxf(sqrtf(totalSquares), 1.0e-12F);
    int32_t const cacheRetainedFrom
        = cacheDepth + currentDepth - cacheOutputDepth;
    for (int32_t c = lane; c < channels; c += 32)
    {
        float normalized = sourceValue(mode, input0, residual, consumeScale,
                               n, c, d, h, w, channels, currentDepth, height,
                               width)
            * factor * __half2float(gamma[c]);
        float const silu = normalized / (1.0F + expf(-normalized));
        int8_t const q = quantizeSigned(silu, invProduceScale);
        packed[cdhw32Offset(n, c, depthFront + cacheDepth + d,
            heightTop + h, widthLeft + w, channels, packedDepth,
            packedHeight, packedWidth)] = q;
        int32_t const sourceDepth = cacheDepth + d;
        if (sourceDepth >= cacheRetainedFrom)
        {
            int32_t const outputDepth = sourceDepth - cacheRetainedFrom;
            if (outputDepth < cacheOutputDepth)
            {
                cacheOutput[cdhw32Offset(n, c, outputDepth, h, w, channels,
                    cacheOutputDepth, height, width)] = q;
            }
        }
    }
}

__global__ void residualTailKernel(int8_t const* input,
    half const* residual, half* output, float scale, int32_t nSize,
    int32_t channels, int32_t depth, int32_t height, int32_t width)
{
    int64_t const count
        = static_cast<int64_t>(nSize) * channels * depth * height * width;
    for (int64_t index
         = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x)
    {
        int64_t value = index;
        int32_t const w = value % width;
        value /= width;
        int32_t const h = value % height;
        value /= height;
        int32_t const d = value % depth;
        value /= depth;
        int32_t const c = value % channels;
        int32_t const n = value / channels;
        float const result
            = static_cast<float>(input[cdhw32Offset(
                  n, c, d, h, w, channels, depth, height, width)])
                * scale
            + __half2float(residual[index]);
        output[index] = __float2half_rn(result);
    }
}

__global__ void migrateCacheKernel(half const* source, int8_t* destination,
    float invScale, int32_t nSize, int32_t channels, int32_t depth,
    int32_t height, int32_t width)
{
    int64_t const count
        = static_cast<int64_t>(nSize) * channels * depth * height * width;
    for (int64_t index
         = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x)
    {
        int64_t value = index;
        int32_t const w = value % width;
        value /= width;
        int32_t const h = value % height;
        value /= height;
        int32_t const d = value % depth;
        value /= depth;
        int32_t const c = value % channels;
        int32_t const n = value / channels;
        destination[cdhw32Offset(
            n, c, d, h, w, channels, depth, height, width)]
            = quantizeSigned(__half2float(source[index]), invScale);
    }
}

int32_t launchCount(int64_t count)
{
    return static_cast<int32_t>(
        std::min<int64_t>((count + 255) / 256, 65535));
}

bool positiveShape(int32_t const* shape)
{
    if (shape == nullptr)
    {
        return false;
    }
    for (int32_t index = 0; index < 5; ++index)
    {
        if (shape[index] <= 0)
        {
            return false;
        }
    }
    return true;
}

} // namespace

extern "C" int32_t sfwanV2LaunchBoundary(int32_t mode, void const* input0,
    void const* input1, void const* input2, void const* input3, void* output0,
    void* output1, void* output2, float consumeScale, float produceScale,
    int32_t const* currentShape, int32_t const* cacheShape,
    int32_t const* packedShape, int32_t const* pads, cudaStream_t stream)
{
    if (input0 == nullptr || output0 == nullptr || currentShape == nullptr
        || consumeScale <= 0.0F || !positiveShape(currentShape)
        || mode < 0 || mode > 3)
    {
        return -1;
    }
    int32_t const n = currentShape[0];
    int32_t const c = currentShape[1];
    int32_t const d = currentShape[2];
    int32_t const h = currentShape[3];
    int32_t const w = currentShape[4];
    if (c % 32 != 0)
    {
        return -1;
    }
    if (mode == 3)
    {
        if (input1 == nullptr)
        {
            return -1;
        }
        int64_t const count = static_cast<int64_t>(n) * c * d * h * w;
        residualTailKernel<<<launchCount(count), 256, 0, stream>>>(
            static_cast<int8_t const*>(input0),
            static_cast<half const*>(input1), static_cast<half*>(output0),
            consumeScale, n, c, d, h, w);
        return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
    }
    if (produceScale <= 0.0F || input1 == nullptr || input2 == nullptr
        || output1 == nullptr || cacheShape == nullptr
        || packedShape == nullptr || pads == nullptr
        || !positiveShape(cacheShape) || !positiveShape(packedShape))
    {
        return -1;
    }
    half const* gamma = mode == 2 ? static_cast<half const*>(input2)
                                  : static_cast<half const*>(input1);
    int8_t const* cache = mode == 2 ? static_cast<int8_t const*>(input3)
                                    : static_cast<int8_t const*>(input2);
    half const* residual
        = mode == 2 ? static_cast<half const*>(input1) : nullptr;
    half* residualOutput = mode == 2 ? static_cast<half*>(output0) : nullptr;
    int8_t* packed = mode == 2 ? static_cast<int8_t*>(output1)
                               : static_cast<int8_t*>(output0);
    int8_t* cacheOutput = mode == 2 ? static_cast<int8_t*>(output2)
                                    : static_cast<int8_t*>(output1);
    if (gamma == nullptr || cache == nullptr || packed == nullptr
        || cacheOutput == nullptr || (mode == 2 && residual == nullptr))
    {
        return -1;
    }
    int32_t const cacheD = cacheShape[2];
    int32_t const cacheOutD = cacheShape[2];
    int32_t const packedD = packedShape[2];
    int32_t const packedH = packedShape[3];
    int32_t const packedW = packedShape[4];
    int32_t const depthFront = pads[2];
    int32_t const heightTop = pads[3];
    int32_t const widthLeft = pads[4];
    if (cacheShape[0] != n || cacheShape[1] != c || cacheShape[3] != h
        || cacheShape[4] != w || packedShape[0] != n
        || packedShape[1] != c
        || packedD != depthFront + cacheD + d + pads[7]
        || packedH != pads[3] + h + pads[8]
        || packedW != pads[4] + w + pads[9])
    {
        return -1;
    }
    size_t const packedBytes
        = static_cast<size_t>(n) * c * packedD * packedH * packedW;
    cudaError_t status = cudaMemsetAsync(packed, 0, packedBytes, stream);
    if (status != cudaSuccess)
    {
        return -1;
    }
    int64_t const cacheCount
        = static_cast<int64_t>(n) * c * cacheD * h * w;
    copyCacheHistoryKernel<<<launchCount(cacheCount), 256, 0, stream>>>(cache,
        packed, cacheOutput, n, c, cacheD, d, h, w, packedD, packedH,
        packedW, depthFront, heightTop, widthLeft, cacheOutD);
    if (cudaPeekAtLastError() != cudaSuccess)
    {
        return -1;
    }
    int64_t const sites = static_cast<int64_t>(n) * d * h * w;
    int32_t const blocks = static_cast<int32_t>((sites + 3) / 4);
    normSiluQuantPackKernel<<<blocks, 128, 0, stream>>>(mode, input0,
        residual, gamma, packed, cacheOutput, residualOutput, consumeScale,
        1.0F / produceScale, n, c, d, h, w, cacheD, cacheOutD, packedD,
        packedH, packedW, depthFront, heightTop, widthLeft);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int32_t sfwanFusionV2MigrateCache(void const* sourceHalf,
    void* destinationInt8, float scale, int32_t const* shape,
    cudaStream_t stream)
{
    if (sourceHalf == nullptr || destinationInt8 == nullptr || scale <= 0.0F
        || !positiveShape(shape) || shape[1] % 32 != 0)
    {
        return -1;
    }
    int64_t const count = static_cast<int64_t>(shape[0]) * shape[1] * shape[2]
        * shape[3] * shape[4];
    migrateCacheKernel<<<launchCount(count), 256, 0, stream>>>(
        static_cast<half const*>(sourceHalf),
        static_cast<int8_t*>(destinationInt8), 1.0F / scale, shape[0],
        shape[1], shape[2], shape[3], shape[4]);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}
