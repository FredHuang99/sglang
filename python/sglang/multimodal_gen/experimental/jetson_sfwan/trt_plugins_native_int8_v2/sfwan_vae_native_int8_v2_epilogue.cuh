// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_fp16.h>

// Register/shared-memory epilogue used by the V2 WMMA convolution.  The
// accumulator never has a global-memory address: each warp stages one 16x16
// fragment in shared memory and immediately applies channel-owned scale,
// bias, residual and output conversion.
__device__ __forceinline__ void sfwanV2StoreResidualValue(
    int32_t accumulator, float activationScale, float const* weightScale,
    float const* bias, void const* shortcut, void* output,
    bool shortcutIsInt8, bool outputIsInt8, float shortcutScale,
    float outputScale, int32_t n, int32_t c, int32_t d, int32_t h,
    int32_t w, int32_t channels, int32_t depth, int32_t height,
    int32_t width)
{
    float result = static_cast<float>(accumulator) * activationScale
            * weightScale[c]
        + bias[c];
    if (shortcutIsInt8)
    {
        result += static_cast<float>(static_cast<int8_t const*>(shortcut)
                [cdhw32Offset(n, c, d, h, w, channels, depth, height,
                    width)])
            * shortcutScale;
    }
    else
    {
        result += __half2float(static_cast<half const*>(shortcut)
                [linearOffset(n, c, d, h, w, channels, depth, height,
                    width)]);
    }
    if (outputIsInt8)
    {
        static_cast<int8_t*>(output)[cdhw32Offset(
            n, c, d, h, w, channels, depth, height, width)]
            = quantizeSigned(result, outputScale);
    }
    else
    {
        static_cast<half*>(output)[linearOffset(
            n, c, d, h, w, channels, depth, height, width)]
            = __float2half_rn(result);
    }
}

__device__ __forceinline__ int8_t sfwanV2ReadDirectCausal(
    int8_t const* current, int8_t const* cache, bool hasCache,
    int32_t n, int32_t c, int32_t outputDepth, int32_t h, int32_t w,
    int32_t history, int32_t channels, int32_t depth, int32_t height,
    int32_t width, int32_t cacheDepth)
{
    int32_t const source = cacheDepth + outputDepth - 2 + history;
    if (source < cacheDepth)
    {
        if (!hasCache || source < 0)
        {
            return 0;
        }
        return cache[cdhw32Offset(n, c, source, h, w, channels,
            cacheDepth, height, width)];
    }
    int32_t const currentDepth = source - cacheDepth;
    if (currentDepth < 0 || currentDepth >= depth)
    {
        return 0;
    }
    return current[cdhw32Offset(n, c, currentDepth, h, w, channels,
        depth, height, width)];
}
