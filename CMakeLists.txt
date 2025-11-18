cmake_minimum_required(VERSION 3.20)

project(cs180_cuda LANGUAGES C CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

# Debug vs Release flags
if(DEFINED ENABLE_CUDA_DEBUG AND ENABLE_CUDA_DEBUG)
    set(CUDA_EXTRA_FLAGS "-G")  # enables cuda-gdb
else()
    set(CUDA_EXTRA_FLAGS "-lineinfo") # line info for profiling/debug
endif()

find_package(CUDAToolkit REQUIRED)
find_package(CCCL REQUIRED)

add_subdirectory(src/Common)
add_subdirectory(src/Fractal)
add_subdirectory(src/Sample)
add_subdirectory(src/Password)
