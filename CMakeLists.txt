cmake_minimum_required(VERSION 3.20)

project(cs180_cuda LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90 100 110 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

# Source file
file(GLOB_RECURSE SRC CONFIGURE_DEPENDS
  "./src/*.cpp"
  "./src/*.hpp"
  "./src/*.c"
  "./src/*.h"
  "./src/*.cu"
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

add_executable(cs180_cuda ${SRC})

target_compile_options(cs180_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(cs180_cuda PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(cs180_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(cs180_cuda PRIVATE CUDA::cudart)
