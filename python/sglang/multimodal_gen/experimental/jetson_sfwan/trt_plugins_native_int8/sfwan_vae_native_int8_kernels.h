// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

struct SfWanNativeInt8BlockConfig
{
    int32_t inputShape[5];
    int32_t shortcutShape[5];
    int32_t outputShape[5];
    int32_t cache1InputShape[5];
    int32_t cache2InputShape[5];
    int32_t cache1OutputShape[5];
    int32_t cache2OutputShape[5];
    int32_t weight1Shape[4]; // K,R,S,T*C after temporal folding.
    int32_t weight2Shape[4];
    int32_t conv1Params[6]; // pad_h,pad_w,stride_h,stride_w,dilation_h,dilation_w.
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

extern "C" size_t sfwanNativeInt8WorkspaceSize(
    SfWanNativeInt8BlockConfig const* config);

extern "C" int32_t sfwanNativeInt8LaunchResidualBlock(
    SfWanNativeInt8BlockConfig const* config, void const* blockInput,
    void const* shortcut, void const* cache1, void const* cache2,
    void const* gamma1, void const* gamma2, void const* weight1,
    void const* weightScale1, void const* bias1, void const* weight2,
    void const* weightScale2, void const* bias2, void* blockOutput,
    void* cache1Output, void* cache2Output, void* workspace,
    cudaStream_t stream);

extern "C" int32_t sfwanNativeInt8TileCount();
extern "C" char const* sfwanNativeInt8CutlassCommit();

extern "C" int32_t sfwanNativeInt8TuneConv(int32_t const* inputShape,
    int32_t const* weightShape, int32_t const* outputShape,
    int32_t const* convParams, int32_t tileId, int32_t warmup,
    int32_t repeat, float* elapsedMs);

extern "C" int32_t sfwanNativeInt8SetProfiling(int32_t enabled);
extern "C" int32_t sfwanNativeInt8ResetProfiles();
extern "C" int32_t sfwanNativeInt8ReadProfile(int32_t profileId,
    float* stageTotalsMs, int32_t stageCount, int64_t* calls);
