cmake_minimum_required(VERSION 3.18)
project(Fractal LANGUAGES CXX CUDA)

file(GLOB_RECURSE FRACTAL_SOURCES CONFIGURE_DEPENDS
    "*.cpp" "*.c" "*.hpp" "*.h" "*.cu"
)

add_executable(Fractal ${FRACTAL_SOURCES})

target_link_libraries(Fractal PRIVATE Common)
