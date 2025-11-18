/**
 * @file file.hpp
 * @author Dante Harper
 * @date 13/11/25
 *
 * @brief [TODO: Brief description of the file's purpose]
 */

#pragma once

// #define STB_IMAGE_WRITE_IMPLEMENTATION
// #include "std_image_write.h"

#include <cstddef>
void write_pgm(const char* filename, size_t width, size_t height, unsigned char* data);
