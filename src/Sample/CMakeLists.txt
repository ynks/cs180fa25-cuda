cmake_minimum_required(VERSION 3.18)
project(Sample LANGUAGES CXX CUDA)

file(GLOB_RECURSE FRACTAL_SOURCES CONFIGURE_DEPENDS
    "*.cpp" "*.c" "*.hpp" "*.h" "*.cu"
)

add_executable(Sample ${FRACTAL_SOURCES})

target_link_libraries(Sample PRIVATE Common)
