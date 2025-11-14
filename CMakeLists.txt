cmake_minimum_required(VERSION 3.20)

project(cs180_cuda LANGUAGES C CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

# Debug vs Release flags
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enables cuda-gdb
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # line info for profiling/debug
endif()

# Source Files
file(GLOB_RECURSE SRC CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/*.[ch]pp"
    "${CMAKE_SOURCE_DIR}/src/*.[ch]"
    "${CMAKE_SOURCE_DIR}/src/*.cu"
)

find_package(CUDAToolkit REQUIRED)
find_package(CCCL REQUIRED)

add_executable(cs180_cuda ${SRC})
target_link_libraries(cs180_cuda PRIVATE CCCL::CCCL CUDA::cudart CUDA::cudart_static)

target_compile_options(cs180_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/std:c++20>)

# Set C++20 for host and CUDA code
set_target_properties(cs180_cuda PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
    CUDA_SEPARABLE_COMPILATION ON
)
