// SPDX-License-Identifier: Apache-2.0

#include "sfwan_vae_native_int8_kernels.h"

#include <cstdint>

#include <cutlass/conv/convolution.h>

// CUTLASS 57e3cfb still ships the legacy ImplicitGemmConvolution device
// wrapper, but convolution.h no longer provides the workspace-size helper
// referenced by that wrapper.  The native SFWan kernels use serial split-K
// with one slice, so this helper is not exercised at runtime; it must still
// be visible while the wrapper template is parsed.  Keep the complete legacy
// semantics here so the compatibility shim is also correct if workspace-size
// validation is performed for another convolution operator.
namespace cutlass::conv
{

template <typename ProblemSize>
inline std::int64_t sfwan_implicit_gemm_tensor_c_size(
    Operator convolutionOperator, ProblemSize const& problemSize)
{
    if (convolutionOperator == Operator::kFprop)
    {
        return static_cast<std::int64_t>(problemSize.N) * problemSize.P
            * problemSize.Q * problemSize.K;
    }
    if (convolutionOperator == Operator::kDgrad
        || convolutionOperator == Operator::kDeconv)
    {
        return static_cast<std::int64_t>(problemSize.N) * problemSize.H
            * problemSize.W * problemSize.C;
    }
    return static_cast<std::int64_t>(problemSize.K) * problemSize.R
        * problemSize.S * problemSize.C;
}

} // namespace cutlass::conv

#define implicit_gemm_tensor_c_size sfwan_implicit_gemm_tensor_c_size
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#undef implicit_gemm_tensor_c_size
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/conv/threadblock/threadblock_swizzle.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/layout/tensor.h>

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <vector>

#ifndef SFWAN_CUTLASS_COMMIT
#error "SFWAN_CUTLASS_COMMIT must bind the native plugin to one CUTLASS ABI"
#endif

namespace
{

#ifdef SFWAN_NATIVE_INT8_V2
constexpr int32_t kBaseTileCount = 6;
constexpr int32_t kTileCount = 18;
#else
constexpr int32_t kTileCount = 6;
#endif
constexpr int32_t kProfileStageCount = 7;
constexpr int32_t kMaxProfileIds = 2048;
constexpr size_t kCutlassWorkspaceReserve = 64U * 1024U * 1024U;

struct ProfileRecord
{
    std::array<float, kProfileStageCount> totals{};
    int64_t calls{};
};

std::atomic<int32_t> gProfiling{0};
std::array<ProfileRecord, kMaxProfileIds> gProfiles{};
std::mutex gProfileMutex;

size_t alignUp(size_t value, size_t alignment = 256)
{
    return (value + alignment - 1) / alignment * alignment;
}

bool zeroOrPositiveShape(int32_t const* shape, bool allowZero)
{
    if (shape == nullptr)
    {
        return false;
    }
    for (int32_t index = 0; index < 5; ++index)
    {
        if (shape[index] < 0 || (!allowZero && shape[index] == 0))
        {
            return false;
        }
    }
    return true;
}

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

__device__ __forceinline__ int64_t nhwcOffset(int32_t n, int32_t h,
    int32_t w, int32_t c, int32_t height, int32_t width, int32_t channels)
{
    return ((static_cast<int64_t>(n) * height + h) * width + w) * channels
        + c;
}

__device__ __forceinline__ int8_t quantizeSigned(float value, float scale)
{
    int32_t result = __float2int_rn(value / scale);
    result = result < -127 ? -127 : (result > 127 ? 127 : result);
    return static_cast<int8_t>(result);
}

__device__ __forceinline__ float readBlockValue(void const* input,
    bool isInt8, float scale, int32_t n, int32_t c, int32_t d, int32_t h,
    int32_t w, int32_t channels, int32_t depth, int32_t height,
    int32_t width)
{
    if (isInt8)
    {
        return static_cast<float>(static_cast<int8_t const*>(input)[cdhw32Offset(
                   n, c, d, h, w, channels, depth, height, width)])
            * scale;
    }
    return __half2float(static_cast<half const*>(input)[linearOffset(
        n, c, d, h, w, channels, depth, height, width)]);
}

__global__ void entryNormQuantWindowKernel(void const* input,
    int8_t const* cache, half const* gamma, int8_t* temporalWindow,
    int8_t* cacheOutput, bool inputIsInt8, bool hasCache, float inputScale,
    float outputScale, int32_t nSize, int32_t channels, int32_t depth,
    int32_t height, int32_t width, int32_t cacheDepth,
    int32_t cacheOutputDepth)
{
    int32_t constexpr kWarpsPerBlock = 4;
    int32_t const lane = threadIdx.x & 31;
    int32_t const warp = threadIdx.x >> 5;
    int64_t const site
        = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    int64_t const siteCount
        = static_cast<int64_t>(nSize) * depth * height * width;
    if (site >= siteCount)
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

    float sumSquares = 0.0F;
    for (int32_t c = lane; c < channels; c += 32)
    {
        float const x = readBlockValue(input, inputIsInt8, inputScale, n, c,
            d, h, w, channels, depth, height, width);
        sumSquares += x * x;
    }
    unsigned int const mask = __activemask();
    for (int32_t offset = 16; offset > 0; offset /= 2)
    {
        sumSquares += __shfl_down_sync(mask, sumSquares, offset);
    }
    float const totalSquares = __shfl_sync(mask, sumSquares, 0);
    float const factor = sqrtf(static_cast<float>(channels))
        / fmaxf(sqrtf(totalSquares), 1.0e-12F);
    int32_t const foldedChannels = channels * 3;
    for (int32_t c = lane; c < channels; c += 32)
    {
        if (d == 0)
        {
            for (int32_t outputDepth = 0; outputDepth < depth; ++outputDepth)
            {
                for (int32_t history = 0; history < 3; ++history)
                {
                    int32_t const source
                        = cacheDepth + outputDepth - 2 + history;
                    if (source >= cacheDepth)
                    {
                        continue;
                    }
                    int8_t q = 0;
                    if (hasCache && source >= 0)
                    {
                        q = cache[cdhw32Offset(n, c, source, h, w,
                            channels, cacheDepth, height, width)];
                    }
                    temporalWindow[nhwcOffset(n * depth + outputDepth, h, w,
                        history * channels + c, height, width,
                        foldedChannels)] = q;
                }
            }
        }
        float normalized = readBlockValue(input, inputIsInt8, inputScale, n,
                               c, d, h, w, channels, depth, height, width)
            * factor * __half2float(gamma[c]);
        float const silu = normalized / (1.0F + expf(-normalized));
        int8_t const q = quantizeSigned(silu, outputScale);
        for (int32_t outputDepth = d;
             outputDepth < depth && outputDepth <= d + 2; ++outputDepth)
        {
            int32_t const history = 2 - (outputDepth - d);
            temporalWindow[nhwcOffset(n * depth + outputDepth, h, w,
                history * channels + c, height, width, foldedChannels)] = q;
        }
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
                cacheOutput[cdhw32Offset(n, c, outputIndex, h, w, channels,
                    cacheOutputDepth, height, width)] = q;
            }
        }
    }
}

__global__ void accumulatorNormQuantWindowKernel(int32_t const* accumulator,
    float const* weightScale, float const* bias, int8_t const* cache,
    half const* gamma, int8_t* temporalWindow, int8_t* cacheOutput,
    bool hasCache, float activationScale, float outputScale, int32_t nSize,
    int32_t channels, int32_t depth, int32_t height, int32_t width,
    int32_t cacheDepth, int32_t cacheOutputDepth)
{
    int32_t constexpr kWarpsPerBlock = 4;
    int32_t const lane = threadIdx.x & 31;
    int32_t const warp = threadIdx.x >> 5;
    int64_t const site
        = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    int64_t const siteCount
        = static_cast<int64_t>(nSize) * depth * height * width;
    if (site >= siteCount)
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
    float sumSquares = 0.0F;
    for (int32_t c = lane; c < channels; c += 32)
    {
        int64_t const index = nhwcOffset(
            n * depth + d, h, w, c, height, width, channels);
        float const x = static_cast<float>(accumulator[index])
                * activationScale * weightScale[c]
            + bias[c];
        sumSquares += x * x;
    }
    unsigned int const mask = __activemask();
    for (int32_t offset = 16; offset > 0; offset /= 2)
    {
        sumSquares += __shfl_down_sync(mask, sumSquares, offset);
    }
    float const totalSquares = __shfl_sync(mask, sumSquares, 0);
    float const factor = sqrtf(static_cast<float>(channels))
        / fmaxf(sqrtf(totalSquares), 1.0e-12F);
    int32_t const foldedChannels = channels * 3;
    for (int32_t c = lane; c < channels; c += 32)
    {
        if (d == 0)
        {
            for (int32_t outputDepth = 0; outputDepth < depth; ++outputDepth)
            {
                for (int32_t history = 0; history < 3; ++history)
                {
                    int32_t const source
                        = cacheDepth + outputDepth - 2 + history;
                    if (source >= cacheDepth)
                    {
                        continue;
                    }
                    int8_t q = 0;
                    if (hasCache && source >= 0)
                    {
                        q = cache[cdhw32Offset(n, c, source, h, w,
                            channels, cacheDepth, height, width)];
                    }
                    temporalWindow[nhwcOffset(n * depth + outputDepth, h, w,
                        history * channels + c, height, width,
                        foldedChannels)] = q;
                }
            }
        }
        int64_t const index = nhwcOffset(
            n * depth + d, h, w, c, height, width, channels);
        float normalized = (static_cast<float>(accumulator[index])
                               * activationScale * weightScale[c]
                               + bias[c])
            * factor * __half2float(gamma[c]);
        float const silu = normalized / (1.0F + expf(-normalized));
        int8_t const q = quantizeSigned(silu, outputScale);
        for (int32_t outputDepth = d;
             outputDepth < depth && outputDepth <= d + 2; ++outputDepth)
        {
            int32_t const history = 2 - (outputDepth - d);
            temporalWindow[nhwcOffset(n * depth + outputDepth, h, w,
                history * channels + c, height, width, foldedChannels)] = q;
        }
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
                cacheOutput[cdhw32Offset(n, c, outputIndex, h, w, channels,
                    cacheOutputDepth, height, width)] = q;
            }
        }
    }
}

__global__ void residualEpilogueKernel(int32_t const* accumulator,
    float const* weightScale, float const* bias, void const* shortcut,
    void* output, bool shortcutIsInt8, bool outputIsInt8,
    float activationScale, float shortcutScale, float outputScale,
    int32_t nSize, int32_t channels, int32_t depth, int32_t height,
    int32_t width)
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
        int64_t const accumulatorIndex = nhwcOffset(
            n * depth + d, h, w, c, height, width, channels);
        float result = static_cast<float>(accumulator[accumulatorIndex])
                * activationScale * weightScale[c]
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
            static_cast<int8_t*>(output)[cdhw32Offset(n, c, d, h, w,
                channels, depth, height, width)]
                = quantizeSigned(result, outputScale);
        }
        else
        {
            static_cast<half*>(output)[linearOffset(n, c, d, h, w, channels,
                depth, height, width)] = __float2half_rn(result);
        }
    }
}

#ifdef SFWAN_NATIVE_INT8_V2
#include "../trt_plugins_native_int8_v2/sfwan_vae_native_int8_v2_iterators.cuh"
#endif

int32_t launchCount(int64_t count)
{
    return static_cast<int32_t>(
        std::min<int64_t>((count + 255) / 256, 65535));
}

template <typename ThreadblockShape, typename WarpShape, int Stages>
cutlass::Status runConv(int8_t const* activation, int8_t const* weight,
    int32_t* output, void* cutlassWorkspace, int32_t n, int32_t h,
    int32_t w, int32_t c, int32_t k, int32_t r, int32_t s, int32_t p,
    int32_t q, int32_t padH, int32_t padW, int32_t strideH,
    int32_t strideW, int32_t dilationH, int32_t dilationW,
    cudaStream_t stream)
{
    using ElementInput = int8_t;
    using ElementOutput = int32_t;
    using ElementAccumulator = int32_t;
    using Layout = cutlass::layout::TensorNHWC;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
    using Epilogue = cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 4, ElementAccumulator, ElementAccumulator>;
    using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
    using Kernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
        ElementInput, Layout, ElementInput, Layout, ElementOutput, Layout,
        ElementAccumulator, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, ThreadblockShape, WarpShape, InstructionShape,
        Epilogue, Swizzle, Stages, cutlass::arch::OpMultiplyAddSaturate,
        cutlass::conv::IteratorAlgorithm::kOptimized,
        cutlass::conv::StrideSupport::kUnity>::Kernel;
    using Conv = cutlass::conv::device::ImplicitGemmConvolution<Kernel>;
    using InputRef = cutlass::TensorRef<ElementInput, Layout>;
    using OutputRef = cutlass::TensorRef<ElementOutput, Layout>;
    cutlass::conv::Conv2dProblemSize problem(
        cutlass::Tensor4DCoord(n, h, w, c),
        cutlass::Tensor4DCoord(k, r, s, c),
        cutlass::Tensor4DCoord(padH, padH, padW, padW),
        cutlass::MatrixCoord(strideH, strideW),
        cutlass::MatrixCoord(dilationH, dilationW),
        cutlass::Tensor4DCoord(n, p, q, k),
        cutlass::conv::Mode::kCrossCorrelation, 1);
    // The legacy CUTLASS Conv2d wrapper exposes mutable TensorRef inputs even
    // though its fprop iterators only read A and B.  Preserve const throughout
    // the SFWan API and remove it only while binding those legacy references.
    InputRef activationRef(const_cast<ElementInput*>(activation),
        Layout::packed(cutlass::Tensor4DCoord(n, h, w, c)));
    InputRef weightRef(const_cast<ElementInput*>(weight),
        Layout::packed(cutlass::Tensor4DCoord(k, r, s, c)));
    OutputRef outputRef(
        output, Layout::packed(cutlass::Tensor4DCoord(n, p, q, k)));
    typename Conv::Arguments arguments{
        problem,
        activationRef,
        weightRef,
        outputRef,
        outputRef,
        {1, 0},
    };
    Conv operation;
    cutlass::Status status = operation.can_implement(arguments);
    if (status != cutlass::Status::kSuccess)
    {
        return status;
    }
    return operation(arguments, cutlassWorkspace, stream);
}

cutlass::Status dispatchConv(int32_t tileId, int8_t const* activation,
    int8_t const* weight, int32_t* output, void* cutlassWorkspace,
    int32_t n, int32_t h, int32_t w, int32_t c, int32_t k, int32_t r,
    int32_t s, int32_t p, int32_t q, int32_t padH, int32_t padW,
    int32_t strideH, int32_t strideW, int32_t dilationH,
    int32_t dilationW, cudaStream_t stream)
{
#ifdef SFWAN_NATIVE_INT8_V2
    tileId %= kBaseTileCount;
#endif
    switch (tileId)
    {
    case 0:
        return runConv<cutlass::gemm::GemmShape<64, 64, 64>,
            cutlass::gemm::GemmShape<32, 32, 64>, 2>(activation, weight,
            output, cutlassWorkspace, n, h, w, c, k, r, s, p, q, padH,
            padW, strideH, strideW, dilationH, dilationW, stream);
    case 1:
        return runConv<cutlass::gemm::GemmShape<128, 64, 64>,
            cutlass::gemm::GemmShape<64, 32, 64>, 2>(activation, weight,
            output, cutlassWorkspace, n, h, w, c, k, r, s, p, q, padH,
            padW, strideH, strideW, dilationH, dilationW, stream);
    case 2:
        return runConv<cutlass::gemm::GemmShape<64, 128, 64>,
            cutlass::gemm::GemmShape<32, 64, 64>, 2>(activation, weight,
            output, cutlassWorkspace, n, h, w, c, k, r, s, p, q, padH,
            padW, strideH, strideW, dilationH, dilationW, stream);
    case 3:
        return runConv<cutlass::gemm::GemmShape<64, 64, 64>,
            cutlass::gemm::GemmShape<32, 32, 64>, 3>(activation, weight,
            output, cutlassWorkspace, n, h, w, c, k, r, s, p, q, padH,
            padW, strideH, strideW, dilationH, dilationW, stream);
    case 4:
        return runConv<cutlass::gemm::GemmShape<128, 64, 64>,
            cutlass::gemm::GemmShape<64, 32, 64>, 3>(activation, weight,
            output, cutlassWorkspace, n, h, w, c, k, r, s, p, q, padH,
            padW, strideH, strideW, dilationH, dilationW, stream);
    case 5:
        return runConv<cutlass::gemm::GemmShape<64, 128, 64>,
            cutlass::gemm::GemmShape<32, 64, 64>, 3>(activation, weight,
            output, cutlassWorkspace, n, h, w, c, k, r, s, p, q, padH,
            padW, strideH, strideW, dilationH, dilationW, stream);
    default:
        return cutlass::Status::kErrorInvalidProblem;
    }
}

struct WorkspaceLayout
{
    size_t window1Offset{};
    size_t accumulator1Offset{};
    size_t window2Offset{};
    size_t accumulator2Offset{};
    size_t cutlassOffset{};
    size_t windowBytes{};
    size_t compactBytes{};
    size_t accumulator1Bytes{};
    size_t accumulator2Bytes{};
    size_t total{};
};

WorkspaceLayout workspaceLayout(SfWanNativeInt8BlockConfig const& config)
{
    WorkspaceLayout result{};
    int64_t const n = config.inputShape[0];
    int64_t const h1 = config.inputShape[3];
    int64_t const w1 = config.inputShape[4];
    int64_t const c1 = config.inputShape[1] * 3;
    int64_t const k1 = config.weight1Shape[0];
    int64_t const h2 = config.outputShape[3];
    int64_t const w2 = config.outputShape[4];
    int64_t const c2 = config.weight2Shape[3];
    int64_t const k2 = config.weight2Shape[0];
    size_t cursor = 0;
#ifdef SFWAN_NATIVE_INT8_V2
    int32_t const level = config.tile1 / kBaseTileCount + 1;
#else
    int32_t const level = 0;
#endif
    result.window1Offset = cursor;
    int64_t const d1 = config.inputShape[2];
    int64_t const d2 = config.outputShape[2];
    if (level < 2)
    {
        result.windowBytes
            = static_cast<size_t>(n * d1 * h1 * w1 * c1);
        cursor = alignUp(cursor + result.windowBytes);
    }
#ifdef SFWAN_NATIVE_INT8_V2
    else if (level == 2)
    {
        result.compactBytes
            = static_cast<size_t>(n * d1 * h1 * w1 * config.inputShape[1]);
        cursor = alignUp(cursor + result.compactBytes);
    }
#endif
    result.accumulator1Offset = cursor;
    if (level < 3)
    {
        result.accumulator1Bytes
            = static_cast<size_t>(n * d1 * h2 * w2 * k1)
            * sizeof(int32_t);
        cursor = alignUp(cursor + result.accumulator1Bytes);
    }
    result.window2Offset = cursor;
    if (level < 2)
    {
        size_t const bytes = static_cast<size_t>(n * d2 * h2 * w2 * c2);
        result.windowBytes += bytes;
        cursor = alignUp(cursor + bytes);
    }
#ifdef SFWAN_NATIVE_INT8_V2
    else if (level == 2)
    {
        size_t const bytes
            = static_cast<size_t>(n * d2 * h2 * w2 * k1);
        result.compactBytes += bytes;
        cursor = alignUp(cursor + bytes);
    }
#endif
    result.accumulator2Offset = cursor;
#ifndef SFWAN_NATIVE_INT8_V2
    result.accumulator2Bytes
        = static_cast<size_t>(n * d2 * h2 * w2 * k2)
        * sizeof(int32_t);
    cursor = alignUp(cursor + result.accumulator2Bytes);
#endif
    result.cutlassOffset = cursor;
#ifdef SFWAN_NATIVE_INT8_V2
    result.total = alignUp(cursor
        + (level < 3 ? kCutlassWorkspaceReserve
                     : sfwan_v2::kBlockScratchBytes));
#else
    result.total = alignUp(cursor + kCutlassWorkspaceReserve);
#endif
    return result;
}

bool validConfig(SfWanNativeInt8BlockConfig const& config)
{
    if (!zeroOrPositiveShape(config.inputShape, false)
        || !zeroOrPositiveShape(config.shortcutShape, false)
        || !zeroOrPositiveShape(config.outputShape, false)
        || !zeroOrPositiveShape(config.cache1OutputShape, false)
        || !zeroOrPositiveShape(config.cache2OutputShape, false)
        || config.inputShape[0] != 1 || config.outputShape[0] != 1
        || config.inputShape[2] != config.outputShape[2]
        || config.shortcutShape[0] != config.outputShape[0]
        || config.shortcutShape[1] != config.outputShape[1]
        || config.shortcutShape[2] != config.outputShape[2]
        || config.shortcutShape[3] != config.outputShape[3]
        || config.shortcutShape[4] != config.outputShape[4]
        || config.inputShape[1] % 32 != 0
        || config.outputShape[1] % 32 != 0 || config.tile1 < 0
        || config.tile1 >= kTileCount || config.tile2 < 0
        || config.tile2 >= kTileCount || config.inputScale <= 0.0F
        || config.conv1InputScale <= 0.0F
        || config.conv2InputScale <= 0.0F || config.outputScale <= 0.0F)
    {
        return false;
    }
    if (config.weight1Shape[1] <= 0 || config.weight1Shape[2] <= 0
        || config.weight1Shape[3] != config.inputShape[1] * 3
        || config.weight2Shape[3] != config.weight1Shape[0] * 3
        || config.weight2Shape[0] != config.outputShape[1])
    {
        return false;
    }
    return true;
}

struct EventSet
{
    std::array<cudaEvent_t, 7> events{};
    bool active{};

    explicit EventSet(bool enabled) : active(enabled)
    {
        if (!active)
        {
            return;
        }
        for (cudaEvent_t& event : events)
        {
            if (cudaEventCreateWithFlags(&event, cudaEventDefault) != cudaSuccess)
            {
                active = false;
                break;
            }
        }
    }

    ~EventSet()
    {
        for (cudaEvent_t event : events)
        {
            if (event != nullptr)
            {
                cudaEventDestroy(event);
            }
        }
    }
};

void commitProfile(EventSet& eventSet, int32_t profileId)
{
    if (!eventSet.active || profileId < 0 || profileId >= kMaxProfileIds)
    {
        return;
    }
    cudaEventSynchronize(eventSet.events.back());
    std::array<float, kProfileStageCount> values{};
    for (int32_t index = 0; index < 5; ++index)
    {
        cudaEventElapsedTime(
            &values[index], eventSet.events[index], eventSet.events[index + 1]);
    }
    values[5] = 0.0F;
    cudaEventElapsedTime(&values[6], eventSet.events.front(),
        eventSet.events.back());
    std::lock_guard<std::mutex> guard(gProfileMutex);
    ProfileRecord& record = gProfiles[profileId];
    for (int32_t index = 0; index < kProfileStageCount; ++index)
    {
        record.totals[index] += values[index];
    }
    ++record.calls;
}

} // namespace

extern "C" size_t sfwanNativeInt8WorkspaceSize(
    SfWanNativeInt8BlockConfig const* config)
{
    if (config == nullptr || !validConfig(*config))
    {
        return 0;
    }
    return workspaceLayout(*config).total;
}

#ifdef SFWAN_NATIVE_INT8_V2
extern "C" int32_t sfwanNativeInt8V2WorkspaceContract(
    SfWanNativeInt8BlockConfig const* config, uint64_t* values,
    int32_t valueCount)
{
    if (config == nullptr || values == nullptr || valueCount != 4
        || !validConfig(*config))
    {
        return -1;
    }
    WorkspaceLayout const layout = workspaceLayout(*config);
    values[0] = static_cast<uint64_t>(layout.accumulator1Bytes);
    values[1] = static_cast<uint64_t>(layout.accumulator2Bytes);
    values[2] = static_cast<uint64_t>(layout.windowBytes);
    values[3] = static_cast<uint64_t>(layout.total);
    return 0;
}

extern "C" int32_t sfwanNativeInt8V2PersistentTile(
    SfWanNativeInt8BlockConfig const* config, int32_t* tileH,
    int32_t* tileW, uint64_t* dynamicSharedBytes)
{
    if (config == nullptr || tileH == nullptr || tileW == nullptr
        || dynamicSharedBytes == nullptr || !validConfig(*config)
        || config->tile1 / kBaseTileCount + 1 != 3)
    {
        return -1;
    }
    size_t bytes{};
    int32_t status
        = sfwan_v2::selectPersistentTile(*config, *tileH, *tileW, bytes);
    *dynamicSharedBytes = static_cast<uint64_t>(bytes);
    return status;
}
#endif

extern "C" int32_t sfwanNativeInt8LaunchResidualBlock(
    SfWanNativeInt8BlockConfig const* config, void const* blockInput,
    void const* shortcut, void const* cache1, void const* cache2,
    void const* gamma1, void const* gamma2, void const* weight1,
    void const* weightScale1, void const* bias1, void const* weight2,
    void const* weightScale2, void const* bias2, void* blockOutput,
    void* cache1Output, void* cache2Output, void* workspace,
    cudaStream_t stream)
{
    if (config == nullptr || !validConfig(*config) || blockInput == nullptr
        || shortcut == nullptr || gamma1 == nullptr || gamma2 == nullptr
        || weight1 == nullptr || weightScale1 == nullptr || bias1 == nullptr
        || weight2 == nullptr || weightScale2 == nullptr || bias2 == nullptr
        || blockOutput == nullptr || cache1Output == nullptr
        || cache2Output == nullptr || workspace == nullptr
        || (config->hasCache1 && cache1 == nullptr)
        || (config->hasCache2 && cache2 == nullptr))
    {
        return -1;
    }
    WorkspaceLayout const layout = workspaceLayout(*config);
    auto* bytes = static_cast<uint8_t*>(workspace);
    auto* window1 = reinterpret_cast<int8_t*>(bytes + layout.window1Offset);
    auto* accumulator1
        = reinterpret_cast<int32_t*>(bytes + layout.accumulator1Offset);
    auto* window2 = reinterpret_cast<int8_t*>(bytes + layout.window2Offset);
    auto* accumulator2
        = reinterpret_cast<int32_t*>(bytes + layout.accumulator2Offset);
    void* cutlassWorkspace = bytes + layout.cutlassOffset;
    int32_t const n = config->inputShape[0];
    int32_t const c1 = config->inputShape[1];
    int32_t const d1 = config->inputShape[2];
    int32_t const h1 = config->inputShape[3];
    int32_t const w1 = config->inputShape[4];
    int32_t const k1 = config->weight1Shape[0];
    int32_t const d2 = config->outputShape[2];
    int32_t const h2 = config->outputShape[3];
    int32_t const w2 = config->outputShape[4];
    int32_t const k2 = config->weight2Shape[0];
    int32_t const cache1Depth
        = config->hasCache1 ? config->cache1InputShape[2] : 0;
    int32_t const cache2Depth
        = config->hasCache2 ? config->cache2InputShape[2] : 0;
    int64_t const sites1 = static_cast<int64_t>(n) * d1 * h1 * w1;
    int64_t const sites2 = static_cast<int64_t>(n) * d2 * h2 * w2;
    EventSet profile(gProfiling.load(std::memory_order_relaxed) != 0);
    if (profile.active)
    {
        cudaEventRecord(profile.events[0], stream);
    }

#ifdef SFWAN_NATIVE_INT8_V2
    int32_t const level = config->tile1 / kBaseTileCount + 1;
    if (config->tile2 / kBaseTileCount + 1 != level)
    {
        return -1;
    }
    auto* compact1 = reinterpret_cast<int8_t*>(bytes + layout.window1Offset);
    auto* compact2 = reinterpret_cast<int8_t*>(bytes + layout.window2Offset);
    if (level >= 2)
    {
        compact1 = reinterpret_cast<int8_t*>(bytes + layout.window1Offset);
        compact2 = reinterpret_cast<int8_t*>(bytes + layout.window2Offset);
    }
    if (level == 3)
    {
        int32_t persistentStatus = sfwan_v2::launchPersistentResidualBlock(
            *config, blockInput, shortcut,
            static_cast<int8_t const*>(cache1),
            static_cast<int8_t const*>(cache2),
            static_cast<half const*>(gamma1),
            static_cast<half const*>(gamma2),
            static_cast<int8_t const*>(weight1),
            static_cast<float const*>(weightScale1),
            static_cast<float const*>(bias1),
            static_cast<int8_t const*>(weight2),
            static_cast<float const*>(weightScale2),
            static_cast<float const*>(bias2), blockOutput,
            static_cast<int8_t*>(cache1Output),
            static_cast<int8_t*>(cache2Output), stream);
        if (profile.active)
        {
            for (int32_t index = 1; index < 7; ++index)
            {
                cudaEventRecord(profile.events[index], stream);
            }
            commitProfile(profile, config->profileId);
        }
        return persistentStatus;
    }
    if (level >= 2)
    {
        sfwan_v2::entryNormQuantCompactKernel<<<
            static_cast<int32_t>((sites1 + 3) / 4), 128, 0, stream>>>(
            blockInput, static_cast<int8_t const*>(cache1),
            static_cast<half const*>(gamma1), compact1,
            static_cast<int8_t*>(cache1Output), config->inputIsInt8 != 0,
            config->hasCache1 != 0, config->inputScale,
            config->conv1InputScale, n, c1, d1, h1, w1, cache1Depth,
            config->cache1OutputShape[2]);
        if (cudaPeekAtLastError() != cudaSuccess)
        {
            return -1;
        }
        if (profile.active) cudaEventRecord(profile.events[1], stream);
        if (sfwan_v2::launchDirectConvAccumulator(compact1,
                static_cast<int8_t const*>(cache1),
                config->hasCache1 != 0,
                static_cast<int8_t const*>(weight1), accumulator1, n, c1,
                d1, h1, w1, cache1Depth, k1, d2, h2, w2,
                config->weight1Shape[1], config->weight1Shape[2],
                config->conv1Params[0], config->conv1Params[1],
                config->conv1Params[2], config->conv1Params[3],
                config->conv1Params[4], config->conv1Params[5], stream)
            != 0)
        {
            return -1;
        }
        if (profile.active) cudaEventRecord(profile.events[2], stream);
        sfwan_v2::accumulatorNormQuantCompactKernel<<<
            static_cast<int32_t>((sites2 + 3) / 4), 128, 0, stream>>>(
            accumulator1, static_cast<float const*>(weightScale1),
            static_cast<float const*>(bias1),
            static_cast<int8_t const*>(cache2),
            static_cast<half const*>(gamma2), compact2,
            static_cast<int8_t*>(cache2Output), config->hasCache2 != 0,
            config->conv1InputScale, config->conv2InputScale, n, k1, d2,
            h2, w2, cache2Depth, config->cache2OutputShape[2]);
        if (cudaPeekAtLastError() != cudaSuccess)
        {
            return -1;
        }
        if (profile.active) cudaEventRecord(profile.events[3], stream);
        if (sfwan_v2::launchFusedResidualConv(true, compact2,
                static_cast<int8_t const*>(cache2),
                config->hasCache2 != 0,
                static_cast<int8_t const*>(weight2),
                static_cast<float const*>(weightScale2),
                static_cast<float const*>(bias2), shortcut, blockOutput,
                config->shortcutIsInt8 != 0, config->outputIsInt8 != 0,
                config->conv2InputScale, config->inputScale,
                config->outputScale, n, k1, d2, h2, w2, cache2Depth, k2,
                config->outputShape[2], config->outputShape[3],
                config->outputShape[4], config->weight2Shape[1],
                config->weight2Shape[2], config->conv2Params[0],
                config->conv2Params[1], config->conv2Params[2],
                config->conv2Params[3], config->conv2Params[4],
                config->conv2Params[5], stream)
            != 0)
        {
            return -1;
        }
        if (profile.active)
        {
            cudaEventRecord(profile.events[4], stream);
            cudaEventRecord(profile.events[5], stream);
            cudaEventRecord(profile.events[6], stream);
            commitProfile(profile, config->profileId);
        }
        return 0;
    }
#endif

    entryNormQuantWindowKernel<<<static_cast<int32_t>((sites1 + 3) / 4),
        128, 0, stream>>>(blockInput, static_cast<int8_t const*>(cache1),
        static_cast<half const*>(gamma1), window1,
        static_cast<int8_t*>(cache1Output), config->inputIsInt8 != 0,
        config->hasCache1 != 0, config->inputScale,
        config->conv1InputScale, n, c1, d1, h1, w1, cache1Depth,
        config->cache1OutputShape[2]);
    if (cudaPeekAtLastError() != cudaSuccess)
    {
        return -1;
    }
    if (profile.active)
    {
        cudaEventRecord(profile.events[1], stream);
    }
    cutlass::Status status = dispatchConv(config->tile1, window1,
        static_cast<int8_t const*>(weight1), accumulator1, cutlassWorkspace,
        n * d1, h1, w1, c1 * 3, k1, config->weight1Shape[1],
        config->weight1Shape[2], h2, w2, config->conv1Params[0],
        config->conv1Params[1], config->conv1Params[2],
        config->conv1Params[3], config->conv1Params[4],
        config->conv1Params[5], stream);
    if (status != cutlass::Status::kSuccess)
    {
        return -1;
    }
    if (profile.active)
    {
        cudaEventRecord(profile.events[2], stream);
    }
    accumulatorNormQuantWindowKernel<<<static_cast<int32_t>((sites2 + 3) / 4),
        128, 0, stream>>>(accumulator1,
        static_cast<float const*>(weightScale1),
        static_cast<float const*>(bias1), static_cast<int8_t const*>(cache2),
        static_cast<half const*>(gamma2), window2,
        static_cast<int8_t*>(cache2Output), config->hasCache2 != 0,
        config->conv1InputScale, config->conv2InputScale, n, k1, d2, h2,
        w2, cache2Depth, config->cache2OutputShape[2]);
    if (cudaPeekAtLastError() != cudaSuccess)
    {
        return -1;
    }
    if (profile.active)
    {
        cudaEventRecord(profile.events[3], stream);
    }
#ifdef SFWAN_NATIVE_INT8_V2
    int32_t fusedStatus = sfwan_v2::launchFusedResidualConv(false, window2,
        nullptr, false, static_cast<int8_t const*>(weight2),
        static_cast<float const*>(weightScale2),
        static_cast<float const*>(bias2), shortcut, blockOutput,
        config->shortcutIsInt8 != 0, config->outputIsInt8 != 0,
        config->conv2InputScale, config->inputScale, config->outputScale, n,
        k1, d2, h2, w2, 0, k2, config->outputShape[2],
        config->outputShape[3], config->outputShape[4],
        config->weight2Shape[1], config->weight2Shape[2],
        config->conv2Params[0], config->conv2Params[1],
        config->conv2Params[2], config->conv2Params[3],
        config->conv2Params[4], config->conv2Params[5], stream);
    if (fusedStatus != 0)
    {
        return -1;
    }
#else
    status = dispatchConv(config->tile2, window2,
        static_cast<int8_t const*>(weight2), accumulator2, cutlassWorkspace,
        n * d2, h2, w2, config->weight2Shape[3], k2,
        config->weight2Shape[1], config->weight2Shape[2],
        config->outputShape[3], config->outputShape[4],
        config->conv2Params[0], config->conv2Params[1],
        config->conv2Params[2], config->conv2Params[3],
        config->conv2Params[4], config->conv2Params[5], stream);
    if (status != cutlass::Status::kSuccess)
    {
        return -1;
    }
#endif
    if (profile.active)
    {
        cudaEventRecord(profile.events[4], stream);
    }
#ifndef SFWAN_NATIVE_INT8_V2
    int64_t const outputCount
        = static_cast<int64_t>(n) * k2 * config->outputShape[2]
        * config->outputShape[3] * config->outputShape[4];
    residualEpilogueKernel<<<launchCount(outputCount), 256, 0, stream>>>(
        accumulator2, static_cast<float const*>(weightScale2),
        static_cast<float const*>(bias2), shortcut, blockOutput,
        config->shortcutIsInt8 != 0, config->outputIsInt8 != 0,
        config->conv2InputScale, config->inputScale, config->outputScale, n,
        k2, config->outputShape[2], config->outputShape[3],
        config->outputShape[4]);
    if (cudaPeekAtLastError() != cudaSuccess)
    {
        return -1;
    }
#endif
    if (profile.active)
    {
        cudaEventRecord(profile.events[5], stream);
        cudaEventRecord(profile.events[6], stream);
        commitProfile(profile, config->profileId);
    }
    return 0;
}

extern "C" int32_t sfwanNativeInt8TileCount()
{
    return kTileCount;
}

extern "C" char const* sfwanNativeInt8CutlassCommit()
{
    return SFWAN_CUTLASS_COMMIT;
}

extern "C" int32_t sfwanNativeInt8TuneConv(int32_t const* inputShape,
    int32_t const* weightShape, int32_t const* outputShape,
    int32_t const* convParams, int32_t tileId, int32_t warmup,
    int32_t repeat, float* elapsedMs)
{
    if (inputShape == nullptr || weightShape == nullptr
        || outputShape == nullptr || convParams == nullptr
        || elapsedMs == nullptr || tileId < 0 || tileId >= kTileCount
        || warmup < 0 || repeat <= 0)
    {
        return -1;
    }
    cudaDeviceProp properties{};
    int32_t device{};
    if (cudaGetDevice(&device) != cudaSuccess
        || cudaGetDeviceProperties(&properties, device) != cudaSuccess
        || properties.major != 8 || properties.minor != 7)
    {
        return -1;
    }
#ifdef SFWAN_NATIVE_INT8_V2
    tileId %= kBaseTileCount;
#endif
    int32_t const n = inputShape[0] * inputShape[2];
    int32_t const h = inputShape[3];
    int32_t const w = inputShape[4];
    int32_t const c = weightShape[3];
    int32_t const k = weightShape[0];
    int32_t const p = outputShape[3];
    int32_t const q = outputShape[4];
    size_t const activationBytes = static_cast<size_t>(n) * h * w * c;
    size_t const weightBytes = static_cast<size_t>(k) * weightShape[1]
        * weightShape[2] * c;
    size_t const outputBytes
        = static_cast<size_t>(n) * p * q * k * sizeof(int32_t);
    int8_t* activation{};
    int8_t* weight{};
    int32_t* output{};
    void* workspace{};
    cudaStream_t stream{};
    cudaEvent_t start{}, end{};
    auto cleanup = [&] {
        if (start) cudaEventDestroy(start);
        if (end) cudaEventDestroy(end);
        if (stream) cudaStreamDestroy(stream);
        if (activation) cudaFree(activation);
        if (weight) cudaFree(weight);
        if (output) cudaFree(output);
        if (workspace) cudaFree(workspace);
    };
    if (cudaMalloc(reinterpret_cast<void**>(&activation), activationBytes)
            != cudaSuccess
        || cudaMalloc(reinterpret_cast<void**>(&weight), weightBytes)
            != cudaSuccess
        || cudaMalloc(reinterpret_cast<void**>(&output), outputBytes)
            != cudaSuccess
        || cudaMalloc(&workspace, kCutlassWorkspaceReserve) != cudaSuccess
        || cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess
        || cudaEventCreate(&start) != cudaSuccess
        || cudaEventCreate(&end) != cudaSuccess)
    {
        cleanup();
        return -1;
    }
    cudaMemsetAsync(activation, 1, activationBytes, stream);
    cudaMemsetAsync(weight, 1, weightBytes, stream);
    for (int32_t iteration = 0; iteration < warmup; ++iteration)
    {
        if (dispatchConv(tileId, activation, weight, output, workspace, n, h,
                w, c, k, weightShape[1], weightShape[2], p, q,
                convParams[0], convParams[1], convParams[2], convParams[3],
                convParams[4], convParams[5], stream)
            != cutlass::Status::kSuccess)
        {
            cleanup();
            return -1;
        }
    }
    cudaEventRecord(start, stream);
    for (int32_t iteration = 0; iteration < repeat; ++iteration)
    {
        if (dispatchConv(tileId, activation, weight, output, workspace, n, h,
                w, c, k, weightShape[1], weightShape[2], p, q,
                convParams[0], convParams[1], convParams[2], convParams[3],
                convParams[4], convParams[5], stream)
            != cutlass::Status::kSuccess)
        {
            cleanup();
            return -1;
        }
    }
    cudaEventRecord(end, stream);
    if (cudaEventSynchronize(end) != cudaSuccess)
    {
        cleanup();
        return -1;
    }
    float total{};
    cudaEventElapsedTime(&total, start, end);
    *elapsedMs = total / static_cast<float>(repeat);
    cleanup();
    return 0;
}

extern "C" int32_t sfwanNativeInt8SetProfiling(int32_t enabled)
{
    gProfiling.store(enabled != 0 ? 1 : 0, std::memory_order_relaxed);
    return 0;
}

extern "C" int32_t sfwanNativeInt8ResetProfiles()
{
    std::lock_guard<std::mutex> guard(gProfileMutex);
    for (ProfileRecord& record : gProfiles)
    {
        record = ProfileRecord{};
    }
    return 0;
}

extern "C" int32_t sfwanNativeInt8ReadProfile(int32_t profileId,
    float* stageTotalsMs, int32_t stageCount, int64_t* calls)
{
    if (profileId < 0 || profileId >= kMaxProfileIds
        || stageTotalsMs == nullptr || stageCount != kProfileStageCount
        || calls == nullptr)
    {
        return -1;
    }
    std::lock_guard<std::mutex> guard(gProfileMutex);
    ProfileRecord const& record = gProfiles[profileId];
    std::memcpy(stageTotalsMs, record.totals.data(),
        sizeof(float) * kProfileStageCount);
    *calls = record.calls;
    return 0;
}
