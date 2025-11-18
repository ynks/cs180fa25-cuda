cmake_minimum_required(VERSION 3.18)
project(Common LANGUAGES CXX CUDA)

# Source files
file(GLOB_RECURSE COMMON_SOURCES CONFIGURE_DEPENDS
    "*.cpp" "*.c" "*.hpp" "*.h" "*.cu"
)

# Create library
add_library(Common ${COMMON_SOURCES})

# Link CUDA runtime
target_link_libraries(Common PUBLIC CUDA::cudart CCCL::CCCL )

# Include directory
target_include_directories(Common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# --- Standard settings applied to consumers ---
# This will propagate C++20 and CUDA 20 to anything linking Common
target_compile_features(Common PUBLIC cxx_std_20)
set_target_properties(Common PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
    CUDA_SEPARABLE_COMPILATION ON
)

# --- Apply CUDA debug/profiling flags ---
# Use interface compile options so consumers inherit them
if(NOT DEFINED CUDA_EXTRA_FLAGS)
    set(CUDA_EXTRA_FLAGS "-lineinfo")
endif()

target_compile_options(Common INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_EXTRA_FLAGS}>)

# Extended lambda for Linux CUDA
if(UNIX)
    target_compile_options(Common INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
endif()
