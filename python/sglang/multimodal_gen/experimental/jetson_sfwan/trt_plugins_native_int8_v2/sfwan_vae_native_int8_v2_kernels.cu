// SPDX-License-Identifier: Apache-2.0

// Compile the proven V1 producer/CUTLASS plumbing under isolated V2 symbols.
// SFWAN_NATIVE_INT8_V2 activates the fused-epilogue/direct-iterator/persistent
// dispatch branches without changing the V1 translation unit.
#define SFWAN_NATIVE_INT8_V2 1
#define SfWanNativeInt8BlockConfig SfWanNativeInt8V2BlockConfig
#define sfwanNativeInt8WorkspaceSize sfwanNativeInt8V2WorkspaceSize
#define sfwanNativeInt8LaunchResidualBlock \
    sfwanNativeInt8V2LaunchResidualBlock
#define sfwanNativeInt8TileCount sfwanNativeInt8V2TileCount
#define sfwanNativeInt8CutlassCommit sfwanNativeInt8V2CutlassCommit
#define sfwanNativeInt8TuneConv sfwanNativeInt8V2TuneConv
#define sfwanNativeInt8SetProfiling sfwanNativeInt8V2SetProfiling
#define sfwanNativeInt8ResetProfiles sfwanNativeInt8V2ResetProfiles
#define sfwanNativeInt8ReadProfile sfwanNativeInt8V2ReadProfile

#include "../trt_plugins_native_int8/sfwan_vae_native_int8_kernels.cu"
