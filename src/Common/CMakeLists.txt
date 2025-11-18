cmake_minimum_required(VERSION 3.18)
project(Common LANGUAGES CXX CUDA)

file(GLOB_RECURSE COMMON_SOURCES CONFIGURE_DEPENDS
    "*.cpp" "*.c" "*.hpp" "*.h" "*.cu"
)

add_library(Common ${COMMON_SOURCES})

target_link_libraries(Common PUBLIC CUDA::cudart CCCL::CCCL )

target_include_directories(Common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_features(Common PUBLIC cxx_std_20)
set_target_properties(Common PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
    CUDA_SEPARABLE_COMPILATION ON
)

if(NOT DEFINED CUDA_EXTRA_FLAGS)
    set(CUDA_EXTRA_FLAGS "-lineinfo")
endif()

target_compile_options(Common INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_EXTRA_FLAGS}>)

if(UNIX)
    target_compile_options(Common INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
endif()
