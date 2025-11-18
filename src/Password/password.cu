#include "password.h"
#include <cstdio>
#include <curand.h>
#include <curand_kernel.h>

__global__ void PasswordKernel() {
  unsigned long long idx = threadIdx.x + (unsigned long long)blockIdx.x * blockDim.x + (unsigned long long)blockIdx.y * gridDim.x * blockDim.x + (unsigned long long)blockIdx.z * gridDim.x * gridDim.y * blockDim.x;
  if (idx % (1000000ull) == 0) {
    printf("Hello from thread %llu!\n", idx);
  }
}

void StartKernel() {

  // Max threads per block
  dim3 threadsPerBlock(1024, 1, 1);

  // Max grid size in x, y, z
  dim3 numBlocks(2147483647, 65535, 65535);

  unsigned long long totalThreads = threadsPerBlock.x;
  totalThreads *= numBlocks.x;
  totalThreads *= numBlocks.y;
  totalThreads *= numBlocks.z;

  printf("Theoretical max threads: %llu\n", totalThreads);
  // eighteen quintillion 18,158,511,498,668,801,024

  printf("Launching Kernel...\n");
  fflush(stdout);
  PasswordKernel<<<numBlocks, threadsPerBlock>>>();
  cudaDeviceSynchronize();
}
