#include "file.hpp"
#include <cstdio>
#include <cuda_runtime.h>

__global__ void FractalKernel(unsigned char *d_img, int width, int height, int offset) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int id = idx + offset;
  if (id >= width * height) {
    return;
  }

  int x = id % width;
  int y = id / width;

  float real = -2.0f + x * (3.0f) / width;
  float imag = -1.5f + y * (3.0f) / height;
  int i = 1;
  float zReal = 0.0f;
  float zImag = 0.0f;
  float realSqr = 0.0f;
  float imagSqr = 0.0f;

  while ((realSqr + imagSqr) < 4.0f && i < 255) {
    zImag = 2.0f * zReal * zImag + imag;
    zReal = realSqr - imagSqr + real;
    i++;
    realSqr = zReal * zReal;
    imagSqr = zImag * zImag;
  }

  d_img[idx] = (unsigned char)(((float)i / 255) * 255.0f);
}

void FractalGenerate() {
  size_t width = 1024ull * 32;
  size_t height = 1024ull * 32;
  size_t size = width * height;
  unsigned char *h_img = new unsigned char[size];
  if (!h_img) {
    printf("Host allocation failed for %zu bytes\n", size);
    return;
  }

  size_t chunk_size = 1024 * 1024;
  unsigned int threads = 1024;
  unsigned int blocks = (int)((chunk_size + threads - 1) / threads);
  unsigned char *d_img;
  cudaMalloc(&d_img, chunk_size);

  printf("Width: %zu Height: %zu  Size: %zu\n", width, height, size);
  printf("Segments: %zu Blocks: %d  Threads: %d\n", width * height / chunk_size, blocks, threads);
  printf("Chuck Size: %zu\n", chunk_size);
  fflush(stdout);

  for (size_t offset = 0; offset < size; offset += chunk_size) {
    FractalKernel<<<blocks, threads>>>(d_img, width, height, offset);
    cudaDeviceSynchronize();

    cudaMemcpy(h_img + offset, d_img, chunk_size, cudaMemcpyDeviceToHost);
  }

  printf("Job Finnished Writing to file");
  fflush(stdout);

  write_pgm("cool.pgm", width, height, h_img);

  cudaFree(d_img);
  delete[] h_img;
}
