// SPDX-License-Identifier: Apache-2.0

// Reuse only the stable TensorRT IPluginV3 adapter.  Creator identity and all
// CUDA entry points are rebound to Native INT8 V2, so loading this DSO cannot
// register or mutate the V1 creator.
#define SFWAN_NATIVE_INT8_KERNEL_HEADER \
    "../trt_plugins_native_int8_v2/sfwan_vae_native_int8_v2_kernels.h"
#define SFWAN_NATIVE_INT8_CONFIG_TYPE SfWanNativeInt8V2BlockConfig
#define SFWAN_NATIVE_INT8_WORKSPACE_SIZE sfwanNativeInt8V2WorkspaceSize
#define SFWAN_NATIVE_INT8_LAUNCH sfwanNativeInt8V2LaunchResidualBlock
#define SFWAN_NATIVE_INT8_TILE_COUNT sfwanNativeInt8V2TileCount
#define SFWAN_NATIVE_INT8_PLUGIN_NAME "SfWanNativeInt8V2ResidualBlockPlugin"
#define SFWAN_NATIVE_INT8_PLUGIN_INIT initSfWanVaeNativeInt8V2Plugins

#include "../trt_plugins_native_int8/sfwan_vae_native_int8_plugin.cpp"
