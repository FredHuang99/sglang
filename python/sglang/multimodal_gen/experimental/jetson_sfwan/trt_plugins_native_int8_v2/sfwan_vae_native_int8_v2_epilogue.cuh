// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_fp16.h>

#include <cutlass/array.h>
#include <cutlass/epilogue/threadblock/epilogue_base.h>
#include <cutlass/epilogue/threadblock/epilogue_with_broadcast.h>
#include <cutlass/matrix_shape.h>
#include <cutlass/numeric_conversion.h>

// P1 keeps the proven Native V1 CUTLASS implicit-GEMM mainloop.  This output
// operator is deliberately limited to the last step of Conv2: it consumes the
// accumulator fragment while it is still owned by the CUTLASS epilogue
// threads, applies the two per-channel constants and the shortcut, and writes
// the final SFWan layout.  There is no addressable accumulator2 tensor.
template <bool ShortcutIsInt8, bool OutputIsInt8>
class SfWanCutlassResidualOutputOp
{
public:
    using ElementOutput = int32_t;
    using ElementAccumulator = int32_t;
    using ElementCompute = float;
    using ElementVector = float;
    using ElementZ = int32_t;
    using ElementT = int32_t;
    using ElementTensor = int32_t;
    static int constexpr kElementsPerAccess = 4;
    static int constexpr kCount = kElementsPerAccess;
    static bool constexpr kStoreZ = false;
    static bool constexpr kStoreT = false;
    static bool constexpr kIsSingleSource = true;

    using FragmentAccumulator
        = cutlass::Array<ElementAccumulator, kElementsPerAccess>;
    using FragmentOutput = cutlass::Array<ElementOutput, kElementsPerAccess>;
    using FragmentSource = FragmentOutput;
    using FragmentCompute = cutlass::Array<ElementCompute, kElementsPerAccess>;
    using FragmentVector = cutlass::Array<ElementVector, kElementsPerAccess>;
    using FragmentZ = cutlass::Array<ElementZ, kElementsPerAccess>;
    using FragmentT = cutlass::Array<ElementT, kElementsPerAccess>;
    using FragmentTensor = cutlass::Array<ElementTensor, kElementsPerAccess>;
    using FragmentScale = cutlass::Array<ElementCompute, kElementsPerAccess>;

    struct Params
    {
        float activationScale{};
        float shortcutScale{};
        float outputScale{};
        float const* bias{};
        void const* shortcut{};
        void* output{};
        int64_t spatialSites{};
        int32_t channels{};
    };

private:
    Params params_;

public:
    CUTLASS_HOST_DEVICE
    explicit SfWanCutlassResidualOutputOp(Params const& params)
        : params_(params)
    {
    }

    CUTLASS_HOST_DEVICE
    bool is_source_needed() const
    {
        return false;
    }

    CUTLASS_HOST_DEVICE
    void set_k_partition(int, int)
    {
    }

    CUTLASS_DEVICE
    float const* bias_pointer() const
    {
        return params_.bias;
    }

    CUTLASS_DEVICE
    void apply(FragmentAccumulator const& accumulator,
        FragmentScale const& weightScale, FragmentScale const& bias,
        int64_t row, int32_t column, int64_t rowExtent,
        int32_t columnExtent) const
    {
        CUTLASS_PRAGMA_UNROLL
        for (int32_t index = 0; index < kElementsPerAccess; ++index)
        {
            int32_t const channel = column + index;
            if (row >= rowExtent || channel >= columnExtent)
            {
                continue;
            }
            float result = static_cast<float>(accumulator[index])
                    * params_.activationScale * weightScale[index]
                + bias[index];
            if constexpr (ShortcutIsInt8)
            {
                int64_t const offset
                    = (row * (params_.channels / 32) + channel / 32) * 32
                    + channel % 32;
                result += static_cast<float>(
                              static_cast<int8_t const*>(params_.shortcut)[offset])
                    * params_.shortcutScale;
            }
            else
            {
                int64_t const offset
                    = static_cast<int64_t>(channel) * params_.spatialSites + row;
                result += __half2float(
                    static_cast<half const*>(params_.shortcut)[offset]);
            }
            if constexpr (OutputIsInt8)
            {
                int64_t const offset
                    = (row * (params_.channels / 32) + channel / 32) * 32
                    + channel % 32;
                static_cast<int8_t*>(params_.output)[offset]
                    = quantizeSigned(result, params_.outputScale);
            }
            else
            {
                int64_t const offset
                    = static_cast<int64_t>(channel) * params_.spatialSites + row;
                static_cast<half*>(params_.output)[offset]
                    = __float2half_rn(result);
            }
        }
    }
};

// This is the no-source specialization of CUTLASS EpilogueWithBroadcast with
// one surgical difference: instead of materialising Z/T, it hands each aligned
// accumulator access and its per-channel weight scale to the SFWan output op.
// Keeping the stock EpilogueBase preserves the TensorOp fragment permutation,
// partitions-K reduction and the six tuned Native V1 mainloop configurations.
template <typename Shape_, typename WarpMmaOperator_, int PartitionsK,
    typename OutputTileIterator_, typename TensorTileIterator_,
    typename ElementVector_, typename AccumulatorFragmentIterator_,
    typename WarpTileIterator_, typename SharedLoadIterator_,
    typename OutputOp_, typename Padding_, int FragmentsPerPartition = 1,
    int IterationsUnroll = (!cutlass::epilogue::threadblock::
            IsEpilogueFunctorHeavy<OutputOp_>::value)>
class SfWanCutlassResidualEpilogue
    : public cutlass::epilogue::threadblock::EpilogueBase<Shape_,
          typename WarpMmaOperator_::Shape, PartitionsK,
          AccumulatorFragmentIterator_, WarpTileIterator_, Padding_,
          FragmentsPerPartition>
{
public:
    using Base = cutlass::epilogue::threadblock::EpilogueBase<Shape_,
        typename WarpMmaOperator_::Shape, PartitionsK,
        AccumulatorFragmentIterator_, WarpTileIterator_, Padding_,
        FragmentsPerPartition>;
    static bool constexpr kIsSingleSource = true;
    using Shape = Shape_;
    using WarpMmaOperator = WarpMmaOperator_;
    static int constexpr kPartitionsK = PartitionsK;
    using OutputTileIterator = OutputTileIterator_;
    using TensorTileIterator = TensorTileIterator_;
    using ElementVector = ElementVector_;
    using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
    using WarpTileIterator = WarpTileIterator_;
    using SharedLoadIterator = SharedLoadIterator_;
    using OutputOp = OutputOp_;
    using Padding = Padding_;
    using AccumulatorTile = typename Base::AccumulatorTile;
    using ElementAccumulator = typename WarpTileIterator::Element;
    using ElementCompute = typename OutputOp::ElementCompute;
    using ElementTensor = typename TensorTileIterator::Element;
    using ThreadMap = typename OutputTileIterator::ThreadMap;
    using BroadcastFragment = cutlass::Array<ElementCompute,
        ThreadMap::Iterations::kColumn * ThreadMap::kElementsPerAccess>;
    using AccumulatorAccess = cutlass::Array<ElementAccumulator,
        OutputTileIterator::kElementsPerAccess>;
    using ScaleAccess = cutlass::Array<ElementCompute,
        OutputTileIterator::kElementsPerAccess>;
    using BaseSharedStorage = typename Base::SharedStorage;
    using MatrixCoord = cutlass::MatrixCoord;

    static_assert(kPartitionsK == 1,
        "SFWan P1 kernels use the Native V1 single-K-partition tiles");
    static_assert(Base::kFragmentsPerIteration == 1,
        "SFWan P1 epilogue consumes one accumulator fragment per iteration");

    struct SharedStorage
    {
        union
        {
            BaseSharedStorage base;
        };

        CUTLASS_HOST_DEVICE
        SharedStorage() {}
    };

private:
    SharedLoadIterator sharedLoadIterator_;
    int32_t threadIndex_;

    CUTLASS_DEVICE
    void loadBroadcast(BroadcastFragment& fragment,
        ElementVector const* pointer, MatrixCoord const& problemSize,
        MatrixCoord const& threadblockOffset)
    {
        fragment.clear();
        if (pointer == nullptr)
        {
            return;
        }
        int32_t const initialColumn
            = ThreadMap::initial_offset(threadIndex_).column();
        int32_t column = threadblockOffset.column() + initialColumn;
        pointer += initialColumn;
        using Load = cutlass::AlignedArray<ElementVector,
            ThreadMap::kElementsPerAccess>;
        using Compute = cutlass::Array<ElementCompute,
            ThreadMap::kElementsPerAccess>;
        cutlass::NumericArrayConverter<ElementCompute, ElementVector,
            ThreadMap::kElementsPerAccess>
            converter;
        Compute* destination = reinterpret_cast<Compute*>(&fragment);
        CUTLASS_PRAGMA_UNROLL
        for (int32_t iteration = 0;
             iteration < ThreadMap::Iterations::kColumn; ++iteration)
        {
            Load loaded;
            loaded.clear();
            if (column < problemSize.column())
            {
                loaded = *reinterpret_cast<Load const*>(pointer);
            }
            destination[iteration] = converter(loaded);
            column += ThreadMap::Delta::kColumn;
            pointer += ThreadMap::Delta::kColumn;
        }
    }

    CUTLASS_DEVICE
    void apply(OutputOp const& outputOp,
        typename SharedLoadIterator::Fragment const& accumulator,
        BroadcastFragment const& scales, BroadcastFragment const& biases,
        MatrixCoord const& problemSize, MatrixCoord const& threadblockOffset,
        int32_t fragmentIteration)
    {
        AccumulatorAccess const* accumulatorAccess
            = reinterpret_cast<AccumulatorAccess const*>(&accumulator);
        ScaleAccess const* scaleAccess
            = reinterpret_cast<ScaleAccess const*>(&scales);
        ScaleAccess const* biasAccess
            = reinterpret_cast<ScaleAccess const*>(&biases);
        auto const initial = ThreadMap::initial_offset(threadIndex_);
        int32_t constexpr accessCount
            = OutputTileIterator::Fragment::kElements
            / OutputTileIterator::kElementsPerAccess;
        CUTLASS_PRAGMA_UNROLL
        for (int32_t access = 0; access < accessCount; ++access)
        {
            int32_t const columnIteration
                = access % ThreadMap::Iterations::kColumn;
            int32_t fragmentRow
                = access / ThreadMap::Iterations::kColumn;
            int32_t const rowIteration
                = fragmentRow % ThreadMap::Iterations::kRow;
            fragmentRow /= ThreadMap::Iterations::kRow;
            int32_t const groupIteration
                = fragmentRow % ThreadMap::Iterations::kGroup;
            int32_t const clusterIteration
                = fragmentRow / ThreadMap::Iterations::kGroup;
            int64_t const row = static_cast<int64_t>(
                                    threadblockOffset.row() + initial.row())
                + static_cast<int64_t>(fragmentIteration)
                    * ThreadMap::Shape::kRow
                + rowIteration * ThreadMap::Delta::kRow
                + groupIteration * ThreadMap::Delta::kGroup
                + clusterIteration * ThreadMap::Delta::kCluster;
            int32_t const column = threadblockOffset.column()
                + initial.column()
                + columnIteration * ThreadMap::Delta::kColumn;
            outputOp.apply(accumulatorAccess[access],
                scaleAccess[columnIteration], biasAccess[columnIteration],
                row, column,
                problemSize.row(), problemSize.column());
        }
    }

public:
    CUTLASS_DEVICE
    SfWanCutlassResidualEpilogue(SharedStorage& storage,
        int32_t threadIndex, int32_t warpIndex, int32_t laneIndex)
        : Base(storage.base, threadIndex, warpIndex, laneIndex)
        , sharedLoadIterator_(storage.base.reference(), threadIndex)
        , threadIndex_(threadIndex)
    {
    }

    CUTLASS_DEVICE
    void operator()(OutputOp const& outputOp,
        ElementVector const* broadcastPointer,
        OutputTileIterator, AccumulatorTile const& accumulators,
        OutputTileIterator, TensorTileIterator,
        MatrixCoord const& problemSize = MatrixCoord(Shape::kM, Shape::kN),
        MatrixCoord const& threadblockOffset = MatrixCoord())
    {
        BroadcastFragment scales;
        BroadcastFragment biases;
        loadBroadcast(scales, broadcastPointer, problemSize,
            threadblockOffset);
        loadBroadcast(biases, outputOp.bias_pointer(), problemSize,
            threadblockOffset);
        AccumulatorFragmentIterator accumulatorIterator(accumulators);
        CUTLASS_PRAGMA_UNROLL
        for (int32_t iteration = 0;
             iteration < OutputTileIterator::kIterations;
             ++iteration)
        {
            __syncthreads();
            typename AccumulatorFragmentIterator::Fragment fragment;
            accumulatorIterator.load(fragment);
            ++accumulatorIterator;
            this->warp_tile_iterator_.store(fragment);
            __syncthreads();
            typename SharedLoadIterator::Fragment aligned;
            sharedLoadIterator_.load(aligned);
            apply(outputOp, aligned, scales, biases, problemSize,
                threadblockOffset, iteration);
        }
    }
};

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
