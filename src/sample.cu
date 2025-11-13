#include <cstdio>
#include <cuda_runtime.h>

__global__ void test() {
  // do nothing
}

int main() {
  // Launch kernel
  test<<<1, 1>>>();

  // Always check for errors and wait for the kernel to finish
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    return 1;
  }

  printf("Kernel launched successfully!");
  return 0;
}
