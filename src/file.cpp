#include "file.hpp"
#include <fstream>
#include <iostream>

void write_pgm(const char *filename, size_t width, size_t height,
               unsigned char *data) {
  std::ofstream out(filename, std::ios::binary);
  if (!out) {
    std::cerr << "Failed to open file for writing\n";
    return;
  }

  // Write the PGM header
  out << "P5\n" << width << " " << height << "\n255\n";

  // Write pixel data
  out.write(reinterpret_cast<char *>(data), width * height);
  out.close();
}
