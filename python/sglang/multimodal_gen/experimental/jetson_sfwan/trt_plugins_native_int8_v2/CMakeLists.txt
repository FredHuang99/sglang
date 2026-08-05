cmake_minimum_required(VERSION 3.24)

project(sfwan_vae_native_int8_v2 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "87")
  message(FATAL_ERROR "native_int8_v2 is intentionally restricted to Jetson Orin SM87")
endif()

set(SFWAN_CUTLASS_DIR "" CACHE PATH "Pinned CUTLASS source checkout")
set(SFWAN_CUTLASS_COMMIT
    "57e3cfb47a2d9e0d46eb6335c3dc411498efa198"
    CACHE STRING "Pinned CUTLASS commit")
if(NOT EXISTS "${SFWAN_CUTLASS_DIR}/include/cutlass/cutlass.h")
  message(FATAL_ERROR "SFWAN_CUTLASS_DIR must contain include/cutlass/cutlass.h")
endif()
execute_process(
  COMMAND git -C "${SFWAN_CUTLASS_DIR}" rev-parse HEAD
  OUTPUT_VARIABLE SFWAN_CUTLASS_HEAD
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE SFWAN_CUTLASS_GIT_RESULT)
if(NOT SFWAN_CUTLASS_GIT_RESULT EQUAL 0)
  message(FATAL_ERROR "SFWAN_CUTLASS_DIR must be a pinned git checkout")
endif()
if(NOT SFWAN_CUTLASS_HEAD STREQUAL SFWAN_CUTLASS_COMMIT)
  message(FATAL_ERROR
    "native_int8_v2 requires CUTLASS ${SFWAN_CUTLASS_COMMIT}; got ${SFWAN_CUTLASS_HEAD}")
endif()

find_package(CUDAToolkit REQUIRED)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS /usr/include/aarch64-linux-gnu /usr/include)
find_library(TENSORRT_LIBRARY nvinfer
  HINTS /usr/lib/aarch64-linux-gnu /usr/lib)
find_library(TENSORRT_PLUGIN_LIBRARY nvinfer_plugin
  HINTS /usr/lib/aarch64-linux-gnu /usr/lib)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY OR NOT TENSORRT_PLUGIN_LIBRARY)
  message(FATAL_ERROR "TensorRT headers/libraries were not found")
endif()

add_library(sfwan_vae_native_int8_v2 SHARED
  sfwan_vae_native_int8_v2_plugin.cpp
  sfwan_vae_native_int8_v2_kernels.cu)
target_include_directories(sfwan_vae_native_int8_v2 PRIVATE
  "${TENSORRT_INCLUDE_DIR}"
  "${SFWAN_CUTLASS_DIR}/include")
target_compile_definitions(sfwan_vae_native_int8_v2 PRIVATE
  SFWAN_CUTLASS_COMMIT="${SFWAN_CUTLASS_COMMIT}")
target_compile_options(sfwan_vae_native_int8_v2 PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra;-Werror=return-type>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;--expt-relaxed-constexpr;--expt-extended-lambda;--threads=2>)
target_link_libraries(sfwan_vae_native_int8_v2 PRIVATE
  CUDA::cudart ${TENSORRT_LIBRARY} ${TENSORRT_PLUGIN_LIBRARY})
set_target_properties(sfwan_vae_native_int8_v2 PROPERTIES
  OUTPUT_NAME sfwan_vae_native_int8_v2
  CUDA_SEPARABLE_COMPILATION OFF)

install(TARGETS sfwan_vae_native_int8_v2 LIBRARY DESTINATION .)
