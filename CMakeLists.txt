cmake_minimum_required(VERSION 3.10)
project(GPU_Benchmark)

cmake_policy(SET CMP0146 NEW)
find_package(CUDAToolkit REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")

include_directories(${CUDA_INCLUDE_DIRS})

set(SOURCE_FILES benchmark.cu)
enable_language(CUDA)
add_executable(benchmark ${SOURCE_FILES})
target_compile_features(benchmark PUBLIC cuda_std_11)
target_link_libraries(benchmark ${CUDAToolkit_LIBRARIES})