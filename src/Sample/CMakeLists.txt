cmake_minimum_required(VERSION 3.18)
project(Sample LANGUAGES CXX CUDA)

# Source files
file(GLOB_RECURSE FRACTAL_SOURCES CONFIGURE_DEPENDS
    "*.cpp" "*.c" "*.hpp" "*.h" "*.cu"
)

# Create executable
add_executable(Sample ${FRACTAL_SOURCES})

# Just link Common and other libraries
target_link_libraries(Sample PRIVATE Common)
