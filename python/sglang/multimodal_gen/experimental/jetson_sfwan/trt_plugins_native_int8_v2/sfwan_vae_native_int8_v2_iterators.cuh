// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfwan_vae_native_int8_v2_epilogue.cuh"

#include <mma.h>

namespace sfwan_v2
{

namespace wmma = nvcuda::wmma;

constexpr int32_t kWarpsPerBlock = 8;
constexpr int32_t kWarpSize = 32;
constexpr int32_t kMmaM = 16;
constexpr int32_t kMmaN = 16;
constexpr int32_t kMmaK = 16;
constexpr size_t kWarpABytes = kMmaM * kMmaK;
constexpr size_t kWarpBBytes = kMmaK * kMmaN;
constexpr size_t kWarpAccumulatorBytes
    = kMmaM * kMmaN * sizeof(int32_t);
constexpr size_t kWarpScratchBytes
    = kWarpABytes + kWarpBBytes + kWarpAccumulatorBytes;
constexpr size_t kBlockScratchBytes = kWarpsPerBlock * kWarpScratchBytes;

__global__ void entryNormQuantCompactKernel(void const* input,
    int8_t const* cache, half const* gamma, int8_t* current,
    int8_t* cacheOutput, bool inputIsInt8, bool hasCache, float inputScale,
    float outputScale, int32_t nSize, int32_t channels, int32_t depth,
    int32_t height, int32_t width, int32_t cacheDepth,
    int32_t cacheOutputDepth)
{
    int32_t constexpr warps = 4;
    int32_t const lane = threadIdx.x & 31;
    int32_t const warp = threadIdx.x >> 5;
    int64_t const site = static_cast<int64_t>(blockIdx.x) * warps + warp;
    int64_t const sites
        = static_cast<int64_t>(nSize) * depth * height * width;
    if (site >= sites)
    {
        return;
    }
    int64_t value = site;
    int32_t const w = value % width;
    value /= width;
    int32_t const h = value % height;
    value /= height;
    int32_t const d = value % depth;
    int32_t const n = value / depth;
    float sum = 0.0F;
    for (int32_t c = lane; c < channels; c += 32)
    {
        float const x = readBlockValue(input, inputIsInt8, inputScale, n,
            c, d, h, w, channels, depth, height, width);
        sum += x * x;
    }
    unsigned int const mask = __activemask();
    for (int32_t offset = 16; offset; offset >>= 1)
    {
        sum += __shfl_down_sync(mask, sum, offset);
    }
    float const factor = sqrtf(static_cast<float>(channels))
        / fmaxf(sqrtf(__shfl_sync(mask, sum, 0)), 1.0e-12F);
    for (int32_t c = lane; c < channels; c += 32)
    {
        float normalized = readBlockValue(input, inputIsInt8, inputScale,
                               n, c, d, h, w, channels, depth, height, width)
            * factor * __half2float(gamma[c]);
        int8_t const q = quantizeSigned(
            normalized / (1.0F + expf(-normalized)), outputScale);
        current[cdhw32Offset(
            n, c, d, h, w, channels, depth, height, width)] = q;
        int32_t const combinedDepth = cacheDepth + depth;
        for (int32_t outputIndex = 0; outputIndex < cacheOutputDepth;
             ++outputIndex)
        {
            int32_t const source
                = combinedDepth - cacheOutputDepth + outputIndex;
            if (source < cacheDepth)
            {
                if (d == 0)
                {
                    int8_t history = 0;
                    if (hasCache && source >= 0)
                    {
                        history = cache[cdhw32Offset(n, c, source, h, w,
                            channels, cacheDepth, height, width)];
                    }
                    cacheOutput[cdhw32Offset(n, c, outputIndex, h, w,
                        channels, cacheOutputDepth, height, width)] = history;
                }
            }
            else if (source - cacheDepth == d)
            {
                cacheOutput[cdhw32Offset(n, c, outputIndex, h, w,
                    channels, cacheOutputDepth, height, width)] = q;
            }
        }
    }
}

__global__ void accumulatorNormQuantCompactKernel(
    int32_t const* accumulator, float const* weightScale,
    float const* bias, int8_t const* cache, half const* gamma,
    int8_t* current, int8_t* cacheOutput, bool hasCache,
    float activationScale, float outputScale, int32_t nSize,
    int32_t channels, int32_t depth, int32_t height, int32_t width,
    int32_t cacheDepth, int32_t cacheOutputDepth)
{
    int32_t constexpr warps = 4;
    int32_t const lane = threadIdx.x & 31;
    int32_t const warp = threadIdx.x >> 5;
    int64_t const site = static_cast<int64_t>(blockIdx.x) * warps + warp;
    int64_t const sites
        = static_cast<int64_t>(nSize) * depth * height * width;
    if (site >= sites)
    {
        return;
    }
    int64_t value = site;
    int32_t const w = value % width;
    value /= width;
    int32_t const h = value % height;
    value /= height;
    int32_t const d = value % depth;
    int32_t const n = value / depth;
    float sum = 0.0F;
    for (int32_t c = lane; c < channels; c += 32)
    {
        int64_t const index
            = nhwcOffset(n * depth + d, h, w, c, height, width, channels);
        float const x = static_cast<float>(accumulator[index])
                * activationScale * weightScale[c]
            + bias[c];
        sum += x * x;
    }
    unsigned int const mask = __activemask();
    for (int32_t offset = 16; offset; offset >>= 1)
    {
        sum += __shfl_down_sync(mask, sum, offset);
    }
    float const factor = sqrtf(static_cast<float>(channels))
        / fmaxf(sqrtf(__shfl_sync(mask, sum, 0)), 1.0e-12F);
    for (int32_t c = lane; c < channels; c += 32)
    {
        int64_t const index
            = nhwcOffset(n * depth + d, h, w, c, height, width, channels);
        float normalized
            = (static_cast<float>(accumulator[index]) * activationScale
                  * weightScale[c]
                + bias[c])
            * factor * __half2float(gamma[c]);
        int8_t const q = quantizeSigned(
            normalized / (1.0F + expf(-normalized)), outputScale);
        current[cdhw32Offset(
            n, c, d, h, w, channels, depth, height, width)] = q;
        int32_t const combinedDepth = cacheDepth + depth;
        for (int32_t outputIndex = 0; outputIndex < cacheOutputDepth;
             ++outputIndex)
        {
            int32_t const source
                = combinedDepth - cacheOutputDepth + outputIndex;
            if (source < cacheDepth)
            {
                if (d == 0)
                {
                    int8_t history = 0;
                    if (hasCache && source >= 0)
                    {
                        history = cache[cdhw32Offset(n, c, source, h, w,
                            channels, cacheDepth, height, width)];
                    }
                    cacheOutput[cdhw32Offset(n, c, outputIndex, h, w,
                        channels, cacheOutputDepth, height, width)] = history;
                }
            }
            else if (source - cacheDepth == d)
            {
                cacheOutput[cdhw32Offset(n, c, outputIndex, h, w,
                    channels, cacheOutputDepth, height, width)] = q;
            }
        }
    }
}

template <bool DirectCausal, bool FusedResidual>
__global__ void directConvKernel(int8_t const* activation,
    int8_t const* cache, bool hasCache, int8_t const* weight,
    float const* weightScale, float const* bias, void const* shortcut,
    void* output, int32_t* accumulatorOutput, bool shortcutIsInt8,
    bool outputIsInt8, float activationScale, float shortcutScale,
    float outputScale, int32_t nSize, int32_t inputChannels,
    int32_t inputDepth, int32_t inputHeight, int32_t inputWidth,
    int32_t cacheDepth, int32_t outputChannels, int32_t outputDepth,
    int32_t outputHeight, int32_t outputWidth, int32_t kernelH,
    int32_t kernelW, int32_t padH, int32_t padW, int32_t strideH,
    int32_t strideW, int32_t dilationH, int32_t dilationW)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    int32_t const warp = threadIdx.x >> 5;
    int32_t const lane = threadIdx.x & 31;
    int32_t const mCount = nSize * outputDepth * outputHeight * outputWidth;
    int32_t const mTiles = (mCount + kMmaM - 1) / kMmaM;
    int32_t const nTiles = (outputChannels + kMmaN - 1) / kMmaN;
    int32_t const task = blockIdx.x * kWarpsPerBlock + warp;
    if (task >= mTiles * nTiles)
    {
        return;
    }
    int32_t const mTile = task / nTiles;
    int32_t const nTile = task % nTiles;
    extern __shared__ __align__(16) unsigned char raw[];
    unsigned char* warpRaw = raw + warp * kWarpScratchBytes;
    int8_t* tileA = reinterpret_cast<int8_t*>(warpRaw);
    int8_t* tileB = reinterpret_cast<int8_t*>(warpRaw + kWarpABytes);
    int32_t* tileC = reinterpret_cast<int32_t*>(
        warpRaw + kWarpABytes + kWarpBBytes);
    wmma::fragment<wmma::accumulator, kMmaM, kMmaN, kMmaK, int32_t>
        fragmentC;
    wmma::fill_fragment(fragmentC, 0);
    int32_t const foldedChannels = inputChannels * 3;
    int32_t const kCount = kernelH * kernelW * foldedChannels;
    int32_t const kTiles = (kCount + kMmaK - 1) / kMmaK;
    for (int32_t kTile = 0; kTile < kTiles; ++kTile)
    {
        for (int32_t index = lane; index < kMmaM * kMmaK; index += 32)
        {
            int32_t const localM = index / kMmaK;
            int32_t const localK = index % kMmaK;
            int32_t const m = mTile * kMmaM + localM;
            int32_t const k = kTile * kMmaK + localK;
            int8_t value = 0;
            if (m < mCount && k < kCount)
            {
                int32_t coordinate = m;
                int32_t const ow = coordinate % outputWidth;
                coordinate /= outputWidth;
                int32_t const oh = coordinate % outputHeight;
                coordinate /= outputHeight;
                int32_t const od = coordinate % outputDepth;
                int32_t const n = coordinate / outputDepth;
                int32_t const folded = k % foldedChannels;
                int32_t const spatial = k / foldedChannels;
                int32_t const kw = spatial % kernelW;
                int32_t const kh = spatial / kernelW;
                int32_t const history = folded / inputChannels;
                int32_t const c = folded % inputChannels;
                int32_t const ih = oh * strideH - padH + kh * dilationH;
                int32_t const iw = ow * strideW - padW + kw * dilationW;
                if (ih >= 0 && ih < inputHeight && iw >= 0
                    && iw < inputWidth)
                {
                    if constexpr (DirectCausal)
                    {
                        value = sfwanV2ReadDirectCausal(activation, cache,
                            hasCache, n, c, od, ih, iw, history,
                            inputChannels, inputDepth, inputHeight,
                            inputWidth, cacheDepth);
                    }
                    else
                    {
                        value = activation[nhwcOffset(n * outputDepth + od,
                            ih, iw, folded, inputHeight, inputWidth,
                            foldedChannels)];
                    }
                }
            }
            tileA[index] = value;
        }
        for (int32_t index = lane; index < kMmaK * kMmaN; index += 32)
        {
            int32_t const localK = index / kMmaN;
            int32_t const localN = index % kMmaN;
            int32_t const k = kTile * kMmaK + localK;
            int32_t const channel = nTile * kMmaN + localN;
            tileB[index] = (k < kCount && channel < outputChannels)
                ? weight[static_cast<int64_t>(channel) * kCount + k]
                : 0;
        }
        __syncwarp();
        wmma::fragment<wmma::matrix_a, kMmaM, kMmaN, kMmaK,
            signed char, wmma::row_major>
            fragmentA;
        wmma::fragment<wmma::matrix_b, kMmaM, kMmaN, kMmaK,
            signed char, wmma::row_major>
            fragmentB;
        wmma::load_matrix_sync(fragmentA,
            reinterpret_cast<signed char const*>(tileA), kMmaK);
        wmma::load_matrix_sync(fragmentB,
            reinterpret_cast<signed char const*>(tileB), kMmaN);
        wmma::mma_sync(fragmentC, fragmentA, fragmentB, fragmentC);
        __syncwarp();
    }
    wmma::store_matrix_sync(
        tileC, fragmentC, kMmaN, wmma::mem_row_major);
    __syncwarp();
    for (int32_t index = lane; index < kMmaM * kMmaN; index += 32)
    {
        int32_t const localM = index / kMmaN;
        int32_t const localN = index % kMmaN;
        int32_t const m = mTile * kMmaM + localM;
        int32_t const c = nTile * kMmaN + localN;
        if (m >= mCount || c >= outputChannels)
        {
            continue;
        }
        int32_t coordinate = m;
        int32_t const w = coordinate % outputWidth;
        coordinate /= outputWidth;
        int32_t const h = coordinate % outputHeight;
        coordinate /= outputHeight;
        int32_t const d = coordinate % outputDepth;
        int32_t const n = coordinate / outputDepth;
        if constexpr (FusedResidual)
        {
            sfwanV2StoreResidualValue(tileC[index], activationScale,
                weightScale, bias, shortcut, output, shortcutIsInt8,
                outputIsInt8, shortcutScale, outputScale, n, c, d, h, w,
                outputChannels, outputDepth, outputHeight, outputWidth);
        }
        else
        {
            accumulatorOutput[nhwcOffset(n * outputDepth + d, h, w, c,
                outputHeight, outputWidth, outputChannels)] = tileC[index];
        }
    }
#endif
}

inline int32_t launchDirectConvAccumulator(int8_t const* current,
    int8_t const* cache, bool hasCache, int8_t const* weight,
    int32_t* accumulator, int32_t n, int32_t inputChannels,
    int32_t inputDepth, int32_t inputHeight, int32_t inputWidth,
    int32_t cacheDepth, int32_t outputChannels, int32_t outputDepth,
    int32_t outputHeight, int32_t outputWidth, int32_t kernelH,
    int32_t kernelW, int32_t padH, int32_t padW, int32_t strideH,
    int32_t strideW, int32_t dilationH, int32_t dilationW,
    cudaStream_t stream)
{
    int32_t const m = n * outputDepth * outputHeight * outputWidth;
    int32_t const tasks = ((m + 15) / 16)
        * ((outputChannels + 15) / 16);
    directConvKernel<true, false>
        <<<static_cast<int32_t>((tasks + kWarpsPerBlock - 1)
               / kWarpsPerBlock),
            kWarpsPerBlock * kWarpSize, kBlockScratchBytes, stream>>>(
            current, cache, hasCache, weight, nullptr, nullptr, nullptr,
            nullptr, accumulator, false, false, 1.0F, 1.0F, 1.0F, n,
            inputChannels, inputDepth, inputHeight, inputWidth, cacheDepth,
            outputChannels, outputDepth, outputHeight, outputWidth, kernelH,
            kernelW, padH, padW, strideH, strideW, dilationH, dilationW);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}

inline int32_t launchFusedResidualConv(bool directCausal,
    int8_t const* activation, int8_t const* cache, bool hasCache,
    int8_t const* weight, float const* weightScale, float const* bias,
    void const* shortcut, void* output, bool shortcutIsInt8,
    bool outputIsInt8, float activationScale, float shortcutScale,
    float outputScale, int32_t n, int32_t inputChannels,
    int32_t inputDepth, int32_t inputHeight, int32_t inputWidth,
    int32_t cacheDepth, int32_t outputChannels, int32_t outputDepth,
    int32_t outputHeight, int32_t outputWidth, int32_t kernelH,
    int32_t kernelW, int32_t padH, int32_t padW, int32_t strideH,
    int32_t strideW, int32_t dilationH, int32_t dilationW,
    cudaStream_t stream)
{
    int32_t const m = n * outputDepth * outputHeight * outputWidth;
    int32_t const tasks = ((m + 15) / 16)
        * ((outputChannels + 15) / 16);
    dim3 const grid(static_cast<uint32_t>(
        (tasks + kWarpsPerBlock - 1) / kWarpsPerBlock));
    dim3 const block(kWarpsPerBlock * kWarpSize);
    if (directCausal)
    {
        directConvKernel<true, true><<<grid, block, kBlockScratchBytes,
            stream>>>(activation, cache, hasCache, weight, weightScale,
            bias, shortcut, output, nullptr, shortcutIsInt8, outputIsInt8,
            activationScale, shortcutScale, outputScale, n, inputChannels,
            inputDepth, inputHeight, inputWidth, cacheDepth, outputChannels,
            outputDepth, outputHeight, outputWidth, kernelH, kernelW, padH,
            padW, strideH, strideW, dilationH, dilationW);
    }
    else
    {
        directConvKernel<false, true><<<grid, block, kBlockScratchBytes,
            stream>>>(activation, nullptr, false, weight, weightScale, bias,
            shortcut, output, nullptr, shortcutIsInt8, outputIsInt8,
            activationScale, shortcutScale, outputScale, n, inputChannels,
            inputDepth, inputHeight, inputWidth, 0, outputChannels,
            outputDepth, outputHeight, outputWidth, kernelH, kernelW, padH,
            padW, strideH, strideW, dilationH, dilationW);
    }
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}

__device__ __forceinline__ void storePersistentCacheValue(
    int8_t q, int8_t const* cache, int8_t* output, bool hasCache,
    int32_t c, int32_t h, int32_t w, int32_t channels, int32_t height,
    int32_t width, int32_t cacheDepth, int32_t outputDepth)
{
    int32_t const combinedDepth = cacheDepth + 1;
    for (int32_t outputIndex = 0; outputIndex < outputDepth; ++outputIndex)
    {
        int32_t const source = combinedDepth - outputDepth + outputIndex;
        int8_t value = q;
        if (source < cacheDepth)
        {
            value = (hasCache && source >= 0)
                ? cache[cdhw32Offset(0, c, source, h, w, channels,
                      cacheDepth, height, width)]
                : 0;
        }
        output[cdhw32Offset(0, c, outputIndex, h, w, channels,
            outputDepth, height, width)] = value;
    }
}

template <int TileH, int TileW>
__global__ void persistentResidualBlockKernel(
    SfWanNativeInt8BlockConfig config, void const* blockInput,
    void const* shortcut, int8_t const* cache1, int8_t const* cache2,
    half const* gamma1, half const* gamma2, int8_t const* weight1,
    float const* weightScale1, float const* bias1, int8_t const* weight2,
    float const* weightScale2, float const* bias2, void* blockOutput,
    int8_t* cache1Output, int8_t* cache2Output)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    int32_t const lane = threadIdx.x & 31;
    int32_t const warp = threadIdx.x >> 5;
    int32_t const c1 = config.inputShape[1];
    int32_t const k1 = config.weight1Shape[0];
    int32_t const k2 = config.weight2Shape[0];
    int32_t const h = config.outputShape[3];
    int32_t const w = config.outputShape[4];
    int32_t const pad1H = config.conv1Params[0];
    int32_t const pad1W = config.conv1Params[1];
    int32_t const pad2H = config.conv2Params[0];
    int32_t const pad2W = config.conv2Params[1];
    int32_t const inputPatchH = TileH + 2 * (pad1H + pad2H);
    int32_t const inputPatchW = TileW + 2 * (pad1W + pad2W);
    int32_t const midPatchH = TileH + 2 * pad2H;
    int32_t const midPatchW = TileW + 2 * pad2W;
    int32_t const inputPatchM = inputPatchH * inputPatchW;
    int32_t const midPatchM = midPatchH * midPatchW;
    extern __shared__ __align__(16) unsigned char raw[];
    size_t cursor = 0;
    int8_t* entryQ = reinterpret_cast<int8_t*>(raw + cursor);
    cursor += static_cast<size_t>(inputPatchM) * c1;
    cursor = (cursor + 15) & ~size_t(15);
    half* conv1Value = reinterpret_cast<half*>(raw + cursor);
    cursor += static_cast<size_t>(midPatchM) * k1 * sizeof(half);
    cursor = (cursor + 15) & ~size_t(15);
    int8_t* midQ = reinterpret_cast<int8_t*>(raw + cursor);
    cursor += static_cast<size_t>(midPatchM) * k1;
    cursor = (cursor + 15) & ~size_t(15);
    unsigned char* warpScratch = raw + cursor + warp * kWarpScratchBytes;
    int8_t* tileA = reinterpret_cast<int8_t*>(warpScratch);
    int8_t* tileB
        = reinterpret_cast<int8_t*>(warpScratch + kWarpABytes);
    int32_t* tileC = reinterpret_cast<int32_t*>(
        warpScratch + kWarpABytes + kWarpBBytes);
    int32_t const tileOriginH = blockIdx.y * TileH;
    int32_t const tileOriginW = blockIdx.x * TileW;
    int32_t const inputOriginH = tileOriginH - pad1H - pad2H;
    int32_t const inputOriginW = tileOriginW - pad1W - pad2W;
    int32_t const midOriginH = tileOriginH - pad2H;
    int32_t const midOriginW = tileOriginW - pad2W;

    // Producer 1: one warp owns a spatial position and keeps only the scalar
    // RMS statistic in shuffle traffic. Quantized values go directly to the
    // shared tile consumed by Conv1; central values also advance cache1.
    for (int32_t site = warp; site < inputPatchM; site += kWarpsPerBlock)
    {
        int32_t const localH = site / inputPatchW;
        int32_t const localW = site % inputPatchW;
        int32_t const globalH = inputOriginH + localH;
        int32_t const globalW = inputOriginW + localW;
        bool const valid = globalH >= 0 && globalH < config.inputShape[3]
            && globalW >= 0 && globalW < config.inputShape[4];
        float sum = 0.0F;
        if (valid)
        {
            for (int32_t c = lane; c < c1; c += 32)
            {
                float const value = readBlockValue(blockInput,
                    config.inputIsInt8 != 0, config.inputScale, 0, c, 0,
                    globalH, globalW, c1, 1, config.inputShape[3],
                    config.inputShape[4]);
                sum += value * value;
            }
        }
        unsigned int const mask = __activemask();
        for (int32_t offset = 16; offset; offset >>= 1)
        {
            sum += __shfl_down_sync(mask, sum, offset);
        }
        float const factor = valid
            ? sqrtf(static_cast<float>(c1))
                / fmaxf(sqrtf(__shfl_sync(mask, sum, 0)), 1.0e-12F)
            : 0.0F;
        for (int32_t c = lane; c < c1; c += 32)
        {
            int8_t q = 0;
            if (valid)
            {
                float normalized = readBlockValue(blockInput,
                                       config.inputIsInt8 != 0,
                                       config.inputScale, 0, c, 0, globalH,
                                       globalW, c1, 1,
                                       config.inputShape[3],
                                       config.inputShape[4])
                    * factor * __half2float(gamma1[c]);
                q = quantizeSigned(
                    normalized / (1.0F + expf(-normalized)),
                    config.conv1InputScale);
            }
            entryQ[static_cast<int64_t>(site) * c1 + c] = q;
            if (valid && globalH >= tileOriginH
                && globalH < tileOriginH + TileH
                && globalW >= tileOriginW
                && globalW < tileOriginW + TileW)
            {
                int32_t const cacheDepth = config.hasCache1
                    ? config.cache1InputShape[2]
                    : 0;
                storePersistentCacheValue(q, cache1, cache1Output,
                    config.hasCache1 != 0, c, globalH, globalW, c1,
                    config.inputShape[3], config.inputShape[4], cacheDepth,
                    config.cache1OutputShape[2]);
            }
        }
    }
    __syncthreads();

    int32_t const conv1K
        = config.weight1Shape[1] * config.weight1Shape[2] * c1 * 3;
    int32_t const conv1Tasks = ((midPatchM + 15) / 16)
        * ((k1 + 15) / 16);
    for (int32_t task = warp; task < conv1Tasks;
         task += kWarpsPerBlock)
    {
        int32_t const nTiles = (k1 + 15) / 16;
        int32_t const mTile = task / nTiles;
        int32_t const nTile = task % nTiles;
        wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> acc;
        wmma::fill_fragment(acc, 0);
        for (int32_t kBase = 0; kBase < conv1K; kBase += 16)
        {
            for (int32_t index = lane; index < 256; index += 32)
            {
                int32_t const localM = index / 16;
                int32_t const localK = index % 16;
                int32_t const m = mTile * 16 + localM;
                int32_t const k = kBase + localK;
                int8_t value = 0;
                if (m < midPatchM && k < conv1K)
                {
                    int32_t const mh = m / midPatchW;
                    int32_t const mw = m % midPatchW;
                    int32_t const folded = k % (c1 * 3);
                    int32_t const spatial = k / (c1 * 3);
                    int32_t const kh = spatial / config.weight1Shape[2];
                    int32_t const kw = spatial % config.weight1Shape[2];
                    int32_t const history = folded / c1;
                    int32_t const c = folded % c1;
                    int32_t const globalH = midOriginH + mh;
                    int32_t const globalW = midOriginW + mw;
                    int32_t const ih = globalH * config.conv1Params[2]
                        - pad1H + kh * config.conv1Params[4];
                    int32_t const iw = globalW * config.conv1Params[3]
                        - pad1W + kw * config.conv1Params[5];
                    if (ih >= 0 && ih < config.inputShape[3] && iw >= 0
                        && iw < config.inputShape[4])
                    {
                        int32_t const source
                            = (config.hasCache1
                                      ? config.cache1InputShape[2]
                                      : 0)
                            - 2 + history;
                        if (source < (config.hasCache1
                                         ? config.cache1InputShape[2]
                                         : 0))
                        {
                            if (config.hasCache1 && source >= 0)
                            {
                                value = cache1[cdhw32Offset(0, c, source,
                                    ih, iw, c1,
                                    config.cache1InputShape[2],
                                    config.inputShape[3],
                                    config.inputShape[4])];
                            }
                        }
                        else
                        {
                            int32_t const ph = ih - inputOriginH;
                            int32_t const pw = iw - inputOriginW;
                            if (ph >= 0 && ph < inputPatchH && pw >= 0
                                && pw < inputPatchW)
                            {
                                value = entryQ[(static_cast<int64_t>(ph)
                                                    * inputPatchW
                                                + pw)
                                        * c1
                                    + c];
                            }
                        }
                    }
                }
                tileA[index] = value;
                int32_t const bk = kBase + index / 16;
                int32_t const bn = nTile * 16 + index % 16;
                tileB[index] = (bk < conv1K && bn < k1)
                    ? weight1[static_cast<int64_t>(bn) * conv1K + bk]
                    : 0;
            }
            __syncwarp();
            wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char,
                wmma::row_major>
                a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char,
                wmma::row_major>
                b;
            wmma::load_matrix_sync(a,
                reinterpret_cast<signed char const*>(tileA), 16);
            wmma::load_matrix_sync(b,
                reinterpret_cast<signed char const*>(tileB), 16);
            wmma::mma_sync(acc, a, b, acc);
            __syncwarp();
        }
        wmma::store_matrix_sync(tileC, acc, 16, wmma::mem_row_major);
        __syncwarp();
        for (int32_t index = lane; index < 256; index += 32)
        {
            int32_t const m = mTile * 16 + index / 16;
            int32_t const c = nTile * 16 + index % 16;
            if (m < midPatchM && c < k1)
            {
                float const value = static_cast<float>(tileC[index])
                        * config.conv1InputScale * weightScale1[c]
                    + bias1[c];
                conv1Value[static_cast<int64_t>(m) * k1 + c]
                    = __float2half_rn(value);
            }
        }
    }
    __syncthreads();

    for (int32_t site = warp; site < midPatchM; site += kWarpsPerBlock)
    {
        float sum = 0.0F;
        for (int32_t c = lane; c < k1; c += 32)
        {
            float const value = __half2float(
                conv1Value[static_cast<int64_t>(site) * k1 + c]);
            sum += value * value;
        }
        unsigned int const mask = __activemask();
        for (int32_t offset = 16; offset; offset >>= 1)
        {
            sum += __shfl_down_sync(mask, sum, offset);
        }
        float const factor = sqrtf(static_cast<float>(k1))
            / fmaxf(sqrtf(__shfl_sync(mask, sum, 0)), 1.0e-12F);
        int32_t const mh = site / midPatchW;
        int32_t const mw = site % midPatchW;
        int32_t const globalH = midOriginH + mh;
        int32_t const globalW = midOriginW + mw;
        bool const valid = globalH >= 0 && globalH < h && globalW >= 0
            && globalW < w;
        for (int32_t c = lane; c < k1; c += 32)
        {
            float normalized = __half2float(
                                   conv1Value[static_cast<int64_t>(site)
                                           * k1
                                       + c])
                * factor * __half2float(gamma2[c]);
            int8_t const q = valid
                ? quantizeSigned(normalized / (1.0F + expf(-normalized)),
                      config.conv2InputScale)
                : 0;
            midQ[static_cast<int64_t>(site) * k1 + c] = q;
            if (valid && globalH >= tileOriginH
                && globalH < tileOriginH + TileH
                && globalW >= tileOriginW
                && globalW < tileOriginW + TileW)
            {
                int32_t const cacheDepth = config.hasCache2
                    ? config.cache2InputShape[2]
                    : 0;
                storePersistentCacheValue(q, cache2, cache2Output,
                    config.hasCache2 != 0, c, globalH, globalW, k1, h, w,
                    cacheDepth, config.cache2OutputShape[2]);
            }
        }
    }
    __syncthreads();

    int32_t const remainingH = h - tileOriginH;
    int32_t const remainingW = w - tileOriginW;
    int32_t const centralH
        = remainingH > 0 ? (remainingH < TileH ? remainingH : TileH) : 0;
    int32_t const centralW
        = remainingW > 0 ? (remainingW < TileW ? remainingW : TileW) : 0;
    int32_t const centralM = centralH * centralW;
    int32_t const conv2K
        = config.weight2Shape[1] * config.weight2Shape[2] * k1 * 3;
    int32_t const conv2Tasks
        = ((centralM + 15) / 16) * ((k2 + 15) / 16);
    for (int32_t task = warp; task < conv2Tasks;
         task += kWarpsPerBlock)
    {
        int32_t const nTiles = (k2 + 15) / 16;
        int32_t const mTile = task / nTiles;
        int32_t const nTile = task % nTiles;
        wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> acc;
        wmma::fill_fragment(acc, 0);
        for (int32_t kBase = 0; kBase < conv2K; kBase += 16)
        {
            for (int32_t index = lane; index < 256; index += 32)
            {
                int32_t const localM = index / 16;
                int32_t const localK = index % 16;
                int32_t const m = mTile * 16 + localM;
                int32_t const k = kBase + localK;
                int8_t value = 0;
                if (m < centralM && k < conv2K)
                {
                    int32_t const oh = tileOriginH + m / centralW;
                    int32_t const ow = tileOriginW + m % centralW;
                    int32_t const folded = k % (k1 * 3);
                    int32_t const spatial = k / (k1 * 3);
                    int32_t const kh = spatial / config.weight2Shape[2];
                    int32_t const kw = spatial % config.weight2Shape[2];
                    int32_t const history = folded / k1;
                    int32_t const c = folded % k1;
                    int32_t const ih = oh * config.conv2Params[2] - pad2H
                        + kh * config.conv2Params[4];
                    int32_t const iw = ow * config.conv2Params[3] - pad2W
                        + kw * config.conv2Params[5];
                    if (ih >= 0 && ih < h && iw >= 0 && iw < w)
                    {
                        int32_t const cacheDepth = config.hasCache2
                            ? config.cache2InputShape[2]
                            : 0;
                        int32_t const source = cacheDepth - 2 + history;
                        if (source < cacheDepth)
                        {
                            if (config.hasCache2 && source >= 0)
                            {
                                value = cache2[cdhw32Offset(0, c, source,
                                    ih, iw, k1, cacheDepth, h, w)];
                            }
                        }
                        else
                        {
                            int32_t const ph = ih - midOriginH;
                            int32_t const pw = iw - midOriginW;
                            if (ph >= 0 && ph < midPatchH && pw >= 0
                                && pw < midPatchW)
                            {
                                value = midQ[(static_cast<int64_t>(ph)
                                                  * midPatchW
                                              + pw)
                                        * k1
                                    + c];
                            }
                        }
                    }
                }
                tileA[index] = value;
                int32_t const bk = kBase + index / 16;
                int32_t const bn = nTile * 16 + index % 16;
                tileB[index] = (bk < conv2K && bn < k2)
                    ? weight2[static_cast<int64_t>(bn) * conv2K + bk]
                    : 0;
            }
            __syncwarp();
            wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char,
                wmma::row_major>
                a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char,
                wmma::row_major>
                b;
            wmma::load_matrix_sync(a,
                reinterpret_cast<signed char const*>(tileA), 16);
            wmma::load_matrix_sync(b,
                reinterpret_cast<signed char const*>(tileB), 16);
            wmma::mma_sync(acc, a, b, acc);
            __syncwarp();
        }
        wmma::store_matrix_sync(tileC, acc, 16, wmma::mem_row_major);
        __syncwarp();
        for (int32_t index = lane; index < 256; index += 32)
        {
            int32_t const m = mTile * 16 + index / 16;
            int32_t const c = nTile * 16 + index % 16;
            if (m < centralM && c < k2)
            {
                int32_t const oh = tileOriginH + m / centralW;
                int32_t const ow = tileOriginW + m % centralW;
                sfwanV2StoreResidualValue(tileC[index],
                    config.conv2InputScale, weightScale2, bias2, shortcut,
                    blockOutput, config.shortcutIsInt8 != 0,
                    config.outputIsInt8 != 0, config.inputScale,
                    config.outputScale, 0, c, 0, oh, ow, k2, 1, h, w);
            }
        }
    }
#endif
}

inline size_t persistentSharedBytes(
    SfWanNativeInt8BlockConfig const& config, int32_t tileH, int32_t tileW)
{
    int32_t const c1 = config.inputShape[1];
    int32_t const k1 = config.weight1Shape[0];
    int32_t const inputH
        = tileH + 2 * (config.conv1Params[0] + config.conv2Params[0]);
    int32_t const inputW
        = tileW + 2 * (config.conv1Params[1] + config.conv2Params[1]);
    int32_t const midH = tileH + 2 * config.conv2Params[0];
    int32_t const midW = tileW + 2 * config.conv2Params[1];
    size_t result = static_cast<size_t>(inputH) * inputW * c1;
    result = (result + 15) & ~size_t(15);
    result += static_cast<size_t>(midH) * midW * k1 * sizeof(half);
    result = (result + 15) & ~size_t(15);
    result += static_cast<size_t>(midH) * midW * k1;
    result = (result + 15) & ~size_t(15);
    return result + kBlockScratchBytes;
}

inline int32_t selectPersistentTile(SfWanNativeInt8BlockConfig const& config,
    int32_t& tileH, int32_t& tileW, size_t& sharedBytes)
{
    int32_t device{};
    int32_t maximum{};
    if (cudaGetDevice(&device) != cudaSuccess
        || cudaDeviceGetAttribute(&maximum,
               cudaDevAttrMaxSharedMemoryPerBlockOptin, device)
            != cudaSuccess)
    {
        return -1;
    }
    int32_t const candidates[][2] = {{8, 16}, {8, 8}, {4, 8}};
    for (auto const& candidate : candidates)
    {
        size_t const bytes
            = persistentSharedBytes(config, candidate[0], candidate[1]);
        if (bytes <= static_cast<size_t>(maximum))
        {
            tileH = candidate[0];
            tileW = candidate[1];
            sharedBytes = bytes;
            return 0;
        }
    }
    return -1;
}

inline int32_t launchPersistentResidualBlock(
    SfWanNativeInt8BlockConfig const& config, void const* blockInput,
    void const* shortcut, int8_t const* cache1, int8_t const* cache2,
    half const* gamma1, half const* gamma2, int8_t const* weight1,
    float const* weightScale1, float const* bias1, int8_t const* weight2,
    float const* weightScale2, float const* bias2, void* blockOutput,
    int8_t* cache1Output, int8_t* cache2Output, cudaStream_t stream)
{
    if (config.inputShape[0] != 1 || config.inputShape[2] != 1
        || config.outputShape[0] != 1 || config.outputShape[2] != 1
        || config.inputShape[3] != config.outputShape[3]
        || config.inputShape[4] != config.outputShape[4]
        || config.conv1Params[2] != 1 || config.conv1Params[3] != 1
        || config.conv2Params[2] != 1 || config.conv2Params[3] != 1)
    {
        return -1;
    }
    int32_t tileH{}, tileW{};
    size_t sharedBytes{};
    if (selectPersistentTile(config, tileH, tileW, sharedBytes) != 0)
    {
        return -1;
    }
    dim3 const grid(
        static_cast<uint32_t>((config.outputShape[4] + tileW - 1) / tileW),
        static_cast<uint32_t>((config.outputShape[3] + tileH - 1) / tileH));
    dim3 const block(kWarpsPerBlock * kWarpSize);
#define SFWAN_LAUNCH_PERSISTENT(TH, TW)                                      \
    do                                                                       \
    {                                                                        \
        auto kernel = persistentResidualBlockKernel<TH, TW>;                 \
        if (cudaFuncSetAttribute(kernel,                                     \
                cudaFuncAttributeMaxDynamicSharedMemorySize,                 \
                static_cast<int32_t>(sharedBytes))                           \
            != cudaSuccess)                                                  \
        {                                                                    \
            return -1;                                                       \
        }                                                                    \
        kernel<<<grid, block, sharedBytes, stream>>>(config, blockInput,      \
            shortcut, cache1, cache2, gamma1, gamma2, weight1,               \
            weightScale1, bias1, weight2, weightScale2, bias2, blockOutput,  \
            cache1Output, cache2Output);                                     \
    } while (false)
    if (tileH == 8 && tileW == 16)
    {
        SFWAN_LAUNCH_PERSISTENT(8, 16);
    }
    else if (tileH == 8 && tileW == 8)
    {
        SFWAN_LAUNCH_PERSISTENT(8, 8);
    }
    else
    {
        SFWAN_LAUNCH_PERSISTENT(4, 8);
    }
#undef SFWAN_LAUNCH_PERSISTENT
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}

} // namespace sfwan_v2
