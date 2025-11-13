#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "std_image_write.h"
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

  while ((realSqr + imagSqr) < 4.0f && i < 100000) {
    zImag = 2.0f * zReal * zImag + imag;
    zReal = realSqr - imagSqr + real;
    i++;
    realSqr = zReal * zReal;
    imagSqr = zImag * zImag;
  }

  d_img[id + 0] = i;
}

void FractalGenerate() {
  size_t width = 32768;
  size_t height = 32768;
  size_t chunk_size = 1024 * 1024;
  size_t size = width * height;

  unsigned char *d_img;
  cudaError_t err = cudaMalloc(&d_img, size);
  if (err != cudaSuccess) {
    printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));
    return;
  }

  unsigned int threads = 1024;
  unsigned int blocks = (int)((chunk_size + threads - 1) / threads);

  printf("Segments: %zu Blocks: %d  Threads: %d\n", width * height / chunk_size, blocks, threads);
  fflush(stdout);
  for (size_t offset = 0; offset < width * height; offset += chunk_size) {
    FractalKernel<<<blocks, threads>>>(d_img, width, height, offset);
    cudaDeviceSynchronize();
  }

  unsigned char *h_img = new unsigned char[size];
  if (!h_img) {
    printf("Host allocation failed!\n");
    return;
  }
  cudaMemcpy(h_img, d_img, size, cudaMemcpyDeviceToHost);

  printf("Job Finnished Writing to file");
  fflush(stdout);

  stbi_write_png("cool.png", width, height, 1, h_img, width);

  cudaFree(d_img);
  delete[] h_img;
}
