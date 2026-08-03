// SPDX-License-Identifier: Apache-2.0

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace
{

__device__ __forceinline__ int64_t linearOffset(
    int32_t n, int32_t c, int32_t d, int32_t h, int32_t w, int32_t channels, int32_t depth,
    int32_t height, int32_t width)
{
    return (((static_cast<int64_t>(n) * channels + c) * depth + d) * height + h) * width + w;
}

__device__ __forceinline__ int64_t cdhw32Offset(
    int32_t n, int32_t c, int32_t d, int32_t h, int32_t w, int32_t channels, int32_t depth,
    int32_t height, int32_t width)
{
    int32_t const channelBlocks = (channels + 31) / 32;
    return (((((static_cast<int64_t>(n) * channelBlocks + c / 32) * depth + d) * height + h) * width + w)
                * 32)
        + c % 32;
}

__device__ __forceinline__ int8_t quantizeSigned(float value, float invScale)
{
    int32_t quantized = __float2int_rn(value * invScale);
    quantized = quantized < -127 ? -127 : (quantized > 127 ? 127 : quantized);
    return static_cast<int8_t>(quantized);
}

__global__ void packQuantKernel(half const* current, half const* cache, int8_t* output, half* cacheOutput,
    float invScale,
    int32_t nSize, int32_t channels, int32_t currentDepth, int32_t cacheDepth, int32_t inputHeight,
    int32_t inputWidth, int32_t outputDepth, int32_t outputHeight, int32_t outputWidth, int32_t depthFront,
    int32_t heightTop, int32_t widthLeft, int32_t cacheOutputDepth)
{
    int32_t const paddedChannels = ((channels + 31) / 32) * 32;
    int64_t const packCount
        = static_cast<int64_t>(nSize) * paddedChannels * outputDepth * outputHeight * outputWidth;
    int64_t const cacheCount = cacheOutput == nullptr
        ? 0
        : static_cast<int64_t>(nSize) * channels * cacheOutputDepth * inputHeight * inputWidth;
    int64_t const count = packCount > cacheCount ? packCount : cacheCount;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x)
    {
        if (index < packCount)
        {
            int64_t value = index;
            int32_t const w = value % outputWidth;
            value /= outputWidth;
            int32_t const h = value % outputHeight;
            value /= outputHeight;
            int32_t const d = value % outputDepth;
            value /= outputDepth;
            int32_t const c = value % paddedChannels;
            int32_t const n = value / paddedChannels;

            float sourceValue = 0.0F;
            int32_t const sourceW = w - widthLeft;
            int32_t const sourceH = h - heightTop;
            int32_t const sourceD = d - depthFront;
            if (c < channels && sourceW >= 0 && sourceW < inputWidth && sourceH >= 0
                && sourceH < inputHeight && sourceD >= 0 && sourceD < cacheDepth + currentDepth)
            {
                if (sourceD < cacheDepth)
                {
                    sourceValue = __half2float(cache[linearOffset(
                        n, c, sourceD, sourceH, sourceW, channels, cacheDepth, inputHeight, inputWidth)]);
                }
                else
                {
                    sourceValue = __half2float(current[linearOffset(n, c, sourceD - cacheDepth, sourceH,
                        sourceW, channels, currentDepth, inputHeight, inputWidth)]);
                }
            }
            output[cdhw32Offset(n, c, d, h, w, paddedChannels, outputDepth, outputHeight, outputWidth)]
                = quantizeSigned(sourceValue, invScale);
        }
        if (index < cacheCount)
        {
            int64_t value = index;
            int32_t const w = value % inputWidth;
            value /= inputWidth;
            int32_t const h = value % inputHeight;
            value /= inputHeight;
            int32_t const d = value % cacheOutputDepth;
            value /= cacheOutputDepth;
            int32_t const c = value % channels;
            int32_t const n = value / channels;
            int32_t const sourceD = cacheDepth + currentDepth - cacheOutputDepth + d;
            if (sourceD < cacheDepth)
            {
                cacheOutput[index]
                    = cache[linearOffset(n, c, sourceD, h, w, channels, cacheDepth, inputHeight, inputWidth)];
            }
            else
            {
                cacheOutput[index] = current[linearOffset(n, c, sourceD - cacheDepth, h, w, channels,
                    currentDepth, inputHeight, inputWidth)];
            }
        }
    }
}

__global__ void cacheUpdateKernel(half const* current, half const* cache, half* output, int32_t nSize,
    int32_t channels, int32_t currentDepth, int32_t cacheDepth, int32_t height, int32_t width,
    int32_t outputDepth)
{
    int64_t const count = static_cast<int64_t>(nSize) * channels * outputDepth * height * width;
    int32_t const totalDepth = currentDepth + cacheDepth;
    int32_t const firstSourceDepth = totalDepth - outputDepth;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x)
    {
        int64_t value = index;
        int32_t const w = value % width;
        value /= width;
        int32_t const h = value % height;
        value /= height;
        int32_t const d = value % outputDepth;
        value /= outputDepth;
        int32_t const c = value % channels;
        int32_t const n = value / channels;
        int32_t const sourceD = firstSourceDepth + d;
        half result;
        if (sourceD < cacheDepth)
        {
            result = cache[linearOffset(n, c, sourceD, h, w, channels, cacheDepth, height, width)];
        }
        else
        {
            result = current[linearOffset(
                n, c, sourceD - cacheDepth, h, w, channels, currentDepth, height, width)];
        }
        output[index] = result;
    }
}

__global__ void residualEpilogueKernel(int8_t const* input, half const* residual, half* output, float scale,
    int32_t nSize, int32_t channels, int32_t depth, int32_t height, int32_t width)
{
    int64_t const count = static_cast<int64_t>(nSize) * channels * depth * height * width;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
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
        int64_t const packed = cdhw32Offset(n, c, d, h, w, channels, depth, height, width);
        float const result = static_cast<float>(input[packed]) * scale + __half2float(residual[index]);
        output[index] = __float2half_rn(result);
    }
}

__global__ void normSiluEpilogueKernel(int8_t const* input, half const* gamma, half* output, float scale,
    int32_t nSize, int32_t channels, int32_t depth, int32_t height, int32_t width)
{
    int32_t const position = blockIdx.x;
    int32_t const w = position % width;
    int32_t value = position / width;
    int32_t const h = value % height;
    value /= height;
    int32_t const d = value % depth;
    int32_t const n = value / depth;

    float localSquares = 0.0F;
    for (int32_t c = threadIdx.x; c < channels; c += blockDim.x)
    {
        float const x = static_cast<float>(input[cdhw32Offset(n, c, d, h, w, channels, depth, height, width)])
            * scale;
        localSquares += x * x;
    }
    extern __shared__ float reduction[];
    reduction[threadIdx.x] = localSquares;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (threadIdx.x < stride)
        {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    // torch.nn.functional.normalize uses max(L2 norm, eps). Multiplication by
    // sqrt(C) then gives the Wan channel-first RMS-normalization convention.
    float const factor = sqrtf(static_cast<float>(channels)) / fmaxf(sqrtf(reduction[0]), 1.0e-12F);
    for (int32_t c = threadIdx.x; c < channels; c += blockDim.x)
    {
        int64_t const packed = cdhw32Offset(n, c, d, h, w, channels, depth, height, width);
        float normalized = static_cast<float>(input[packed]) * scale * factor * __half2float(gamma[c]);
        float const silu = normalized / (1.0F + expf(-normalized));
        output[linearOffset(n, c, d, h, w, channels, depth, height, width)] = __float2half_rn(silu);
    }
}

int32_t launchCount(int64_t elements)
{
    return static_cast<int32_t>(std::min<int64_t>((elements + 255) / 256, 65535));
}

} // namespace

extern "C" int32_t sfwanLaunchPackQuant(half const* current, half const* cache, int8_t* output,
    half* cacheOutput, float scale, int32_t const* currentShape, int32_t const* cacheShape,
    int32_t const* outputShape, int32_t const* cacheOutputShape, int32_t const* pads, cudaStream_t stream)
{
    if (current == nullptr || output == nullptr || scale <= 0.0F)
    {
        return -1;
    }
    int32_t const n = currentShape[0];
    int32_t const c = currentShape[1];
    int32_t const currentD = currentShape[2];
    int32_t const h = currentShape[3];
    int32_t const w = currentShape[4];
    int32_t const cacheD = cache == nullptr ? 0 : cacheShape[2];
    int32_t const outD = outputShape[2];
    int32_t const outH = outputShape[3];
    int32_t const outW = outputShape[4];
    int32_t const depthBack = pads[7];
    int32_t const depthFront = outD - currentD - cacheD - depthBack;
    int32_t const heightTop = pads[3];
    int32_t const widthLeft = pads[4];
    int32_t const cacheOutputD = cacheOutput == nullptr ? 0 : cacheOutputShape[2];
    if (n <= 0 || c <= 0 || currentD <= 0 || h <= 0 || w <= 0 || depthFront < 0 || depthBack < 0
        || cacheOutputD < 0 || cacheOutputD > cacheD + currentD)
    {
        return -1;
    }
    int64_t const packCount = static_cast<int64_t>(n) * ((c + 31) / 32 * 32) * outD * outH * outW;
    int64_t const cacheCount = static_cast<int64_t>(n) * c * cacheOutputD * h * w;
    packQuantKernel<<<launchCount(std::max(packCount, cacheCount)), 256, 0, stream>>>(current, cache, output,
        cacheOutput, 1.0F / scale, n, c, currentD, cacheD, h, w, outD, outH, outW, depthFront, heightTop,
        widthLeft, cacheOutputD);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int32_t sfwanLaunchCacheUpdate(half const* current, half const* cache, half* output,
    int32_t const* currentShape, int32_t const* cacheShape, int32_t const* outputShape, cudaStream_t stream)
{
    if (current == nullptr || output == nullptr)
    {
        return -1;
    }
    int32_t const n = currentShape[0];
    int32_t const c = currentShape[1];
    int32_t const currentD = currentShape[2];
    int32_t const h = currentShape[3];
    int32_t const w = currentShape[4];
    int32_t const cacheD = cache == nullptr ? 0 : cacheShape[2];
    int32_t const outputD = outputShape[2];
    if (outputD <= 0 || outputD > cacheD + currentD)
    {
        return -1;
    }
    int64_t const count = static_cast<int64_t>(n) * c * outputD * h * w;
    cacheUpdateKernel<<<launchCount(count), 256, 0, stream>>>(
        current, cache, output, n, c, currentD, cacheD, h, w, outputD);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int32_t sfwanLaunchEpilogue(int32_t mode, int8_t const* input, half const* auxiliary, half* output,
    float scale, int32_t const* shape, cudaStream_t stream)
{
    if (input == nullptr || auxiliary == nullptr || output == nullptr || scale <= 0.0F)
    {
        return -1;
    }
    int32_t const n = shape[0];
    int32_t const c = shape[1];
    int32_t const d = shape[2];
    int32_t const h = shape[3];
    int32_t const w = shape[4];
    if (mode == 0)
    {
        int32_t const positions = n * d * h * w;
        normSiluEpilogueKernel<<<positions, 256, 256 * sizeof(float), stream>>>(
            input, auxiliary, output, scale, n, c, d, h, w);
    }
    else if (mode == 1)
    {
        int64_t const count = static_cast<int64_t>(n) * c * d * h * w;
        residualEpilogueKernel<<<launchCount(count), 256, 0, stream>>>(
            input, auxiliary, output, scale, n, c, d, h, w);
    }
    else
    {
        return -1;
    }
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}
