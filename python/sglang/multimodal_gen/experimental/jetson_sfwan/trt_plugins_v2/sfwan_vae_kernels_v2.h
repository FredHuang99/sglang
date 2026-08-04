#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

extern "C" int32_t sfwanV2LaunchBoundary(int32_t mode, void const* input0,
    void const* input1, void const* input2, void const* input3, void* output0,
    void* output1, void* output2, float consumeScale, float produceScale,
    int32_t const* currentShape, int32_t const* cacheShape,
    int32_t const* packedShape, int32_t const* pads, cudaStream_t stream);

extern "C" int32_t sfwanFusionV2MigrateCache(void const* sourceHalf,
    void* destinationInt8, float scale, int32_t const* shape,
    cudaStream_t stream);
