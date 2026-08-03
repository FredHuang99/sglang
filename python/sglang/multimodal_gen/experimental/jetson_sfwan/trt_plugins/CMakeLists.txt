cmake_minimum_required(VERSION 3.24)

project(sfwan_vae_trt_fusion LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(SFWAN_CUDA_ARCHITECTURES "87" CACHE STRING "CUDA architectures")
if(NOT SFWAN_CUDA_ARCHITECTURES STREQUAL "87")
  message(FATAL_ERROR "fusion_v1 is intentionally restricted to Jetson Orin SM87")
endif()

find_package(CUDAToolkit 12.9 REQUIRED)

find_path(
  TENSORRT_INCLUDE_DIR
  NvInfer.h
  HINTS
    /usr/include/aarch64-linux-gnu
    /usr/include
    /usr/local/include
)
find_library(
  TENSORRT_NVINFER_LIBRARY
  nvinfer
  HINTS
    /usr/lib/aarch64-linux-gnu
    /usr/lib
    /usr/local/lib
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER_LIBRARY)
  message(FATAL_ERROR "TensorRT headers or libnvinfer were not found")
endif()

add_library(
  sfwan_vae_trt_fusion
  SHARED
  sfwan_vae_plugin.cpp
  sfwan_vae_kernels.cu
)

set_target_properties(
  sfwan_vae_trt_fusion
  PROPERTIES
    OUTPUT_NAME sfwan_vae_trt_fusion
    CUDA_ARCHITECTURES "${SFWAN_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION OFF
)

target_include_directories(
  sfwan_vae_trt_fusion
  PRIVATE
    "${TENSORRT_INCLUDE_DIR}"
)

target_compile_options(
  sfwan_vae_trt_fusion
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra;-Werror=return-type>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;--threads=2;-lineinfo>
)

target_link_libraries(
  sfwan_vae_trt_fusion
  PRIVATE
    "${TENSORRT_NVINFER_LIBRARY}"
    CUDA::cudart
)

install(TARGETS sfwan_vae_trt_fusion LIBRARY DESTINATION .)
