// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

// Keep the serialized TensorRT field layout identical to Native INT8 V1.
// V2 encodes the optimization level in tile1/tile2 (six base tile IDs per
// level), so old plans and the V1 creator are never reinterpreted.
struct SfWanNativeInt8V2BlockConfig
{
    int32_t inputShape[5];
    int32_t shortcutShape[5];
    int32_t outputShape[5];
    int32_t cache1InputShape[5];
    int32_t cache2InputShape[5];
    int32_t cache1OutputShape[5];
    int32_t cache2OutputShape[5];
    int32_t weight1Shape[4];
    int32_t weight2Shape[4];
    int32_t conv1Params[6];
    int32_t conv2Params[6];
    int32_t inputIsInt8;
    int32_t shortcutIsInt8;
    int32_t outputIsInt8;
    int32_t hasCache1;
    int32_t hasCache2;
    int32_t tile1;
    int32_t tile2;
    int32_t profileId;
    float inputScale;
    float conv1InputScale;
    float conv2InputScale;
    float outputScale;
};

extern "C" size_t sfwanNativeInt8V2WorkspaceSize(
    SfWanNativeInt8V2BlockConfig const* config);

extern "C" int32_t sfwanNativeInt8V2LaunchResidualBlock(
    SfWanNativeInt8V2BlockConfig const* config, void const* blockInput,
    void const* shortcut, void const* cache1, void const* cache2,
    void const* gamma1, void const* gamma2, void const* weight1,
    void const* weightScale1, void const* bias1, void const* weight2,
    void const* weightScale2, void const* bias2, void* blockOutput,
    void* cache1Output, void* cache2Output, void* workspace,
    cudaStream_t stream);

extern "C" int32_t sfwanNativeInt8V2TileCount();
extern "C" char const* sfwanNativeInt8V2CutlassCommit();
extern "C" int32_t sfwanNativeInt8V2TuneConv(int32_t const* inputShape,
    int32_t const* weightShape, int32_t const* outputShape,
    int32_t const* convParams, int32_t tileId, int32_t warmup,
    int32_t repeat, float* elapsedMs);

extern "C" int32_t sfwanNativeInt8V2SetProfiling(int32_t enabled);
extern "C" int32_t sfwanNativeInt8V2ResetProfiles();
extern "C" int32_t sfwanNativeInt8V2ReadProfile(int32_t profileId,
    float* stageTotalsMs, int32_t stageCount, int64_t* calls);

// Structural audit entry point.  Values are bytes for the concrete config.
// Order: accumulator1, accumulator2, temporal windows, total workspace.
extern "C" int32_t sfwanNativeInt8V2WorkspaceContract(
    SfWanNativeInt8V2BlockConfig const* config, uint64_t* values,
    int32_t valueCount);

// P1 algorithm contract. Values are, in order: tensor-core INT8,
// accumulator2 global store, separate residual kernel, direct-conv kernel
// used by P1, and the unchanged serialized-config ABI revision.
extern "C" char const* sfwanNativeInt8V2P1Algorithm();
extern "C" int32_t sfwanNativeInt8V2P1KernelContract(
    uint64_t* values, int32_t valueCount);

extern "C" int32_t sfwanNativeInt8V2PersistentTile(
    SfWanNativeInt8V2BlockConfig const* config, int32_t* tileH,
    int32_t* tileW, uint64_t* dynamicSharedBytes);
