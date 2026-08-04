cmake_minimum_required(VERSION 3.24)

project(sfwan_vae_native_int8_v1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(SFWAN_CUDA_ARCHITECTURES "87" CACHE STRING "CUDA architectures")
if(NOT SFWAN_CUDA_ARCHITECTURES STREQUAL "87")
  message(FATAL_ERROR "native_int8_v1 is intentionally restricted to Jetson Orin SM87")
endif()

set(SFWAN_CUTLASS_DIR "" CACHE PATH "Path to the pinned NVIDIA CUTLASS checkout")
set(
  SFWAN_CUTLASS_COMMIT
  "57e3cfb47a2d9e0d46eb6335c3dc411498efa198"
  CACHE STRING
  "Pinned CUTLASS commit for the native weight/iterator ABI"
)
if(NOT SFWAN_CUTLASS_DIR)
  message(FATAL_ERROR "SFWAN_CUTLASS_DIR must point to a local CUTLASS checkout")
endif()
if(NOT EXISTS "${SFWAN_CUTLASS_DIR}/include/cutlass/cutlass.h")
  message(FATAL_ERROR "SFWAN_CUTLASS_DIR does not contain include/cutlass/cutlass.h")
endif()
execute_process(
  COMMAND git -C "${SFWAN_CUTLASS_DIR}" rev-parse HEAD
  RESULT_VARIABLE SFWAN_CUTLASS_GIT_STATUS
  OUTPUT_VARIABLE SFWAN_CUTLASS_HEAD
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT SFWAN_CUTLASS_GIT_STATUS EQUAL 0)
  message(FATAL_ERROR "SFWAN_CUTLASS_DIR must retain Git metadata")
endif()
if(NOT SFWAN_CUTLASS_HEAD STREQUAL SFWAN_CUTLASS_COMMIT)
  message(FATAL_ERROR
    "native_int8_v1 requires CUTLASS ${SFWAN_CUTLASS_COMMIT}; got ${SFWAN_CUTLASS_HEAD}"
  )
endif()

find_package(CUDAToolkit 12.9 REQUIRED)
find_path(
  TENSORRT_INCLUDE_DIR
  NvInfer.h
  HINTS /usr/include/aarch64-linux-gnu /usr/include /usr/local/include
)
find_library(
  TENSORRT_NVINFER_LIBRARY
  nvinfer
  HINTS /usr/lib/aarch64-linux-gnu /usr/lib /usr/local/lib
)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER_LIBRARY)
  message(FATAL_ERROR "TensorRT headers or libnvinfer were not found")
endif()

add_library(
  sfwan_vae_native_int8_v1
  SHARED
  sfwan_vae_native_int8_plugin.cpp
  sfwan_vae_native_int8_kernels.cu
)
set_target_properties(
  sfwan_vae_native_int8_v1
  PROPERTIES
    OUTPUT_NAME sfwan_vae_native_int8_v1
    CUDA_ARCHITECTURES "${SFWAN_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION OFF
)
target_include_directories(
  sfwan_vae_native_int8_v1
  PRIVATE
    "${TENSORRT_INCLUDE_DIR}"
    "${SFWAN_CUTLASS_DIR}/include"
    "${SFWAN_CUTLASS_DIR}/tools/util/include"
)
target_compile_definitions(
  sfwan_vae_native_int8_v1
  PRIVATE
    CUTLASS_ENABLE_TENSOR_CORE_MMA=1
    CUTLASS_ARCH_MMA_SM80_ENABLED=1
    SFWAN_CUTLASS_COMMIT="${SFWAN_CUTLASS_COMMIT}"
)
target_compile_options(
  sfwan_vae_native_int8_v1
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra;-Werror=return-type>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;--threads=2;-lineinfo>
)
target_link_libraries(
  sfwan_vae_native_int8_v1
  PRIVATE "${TENSORRT_NVINFER_LIBRARY}" CUDA::cudart
)

install(TARGETS sfwan_vae_native_int8_v1 LIBRARY DESTINATION .)
