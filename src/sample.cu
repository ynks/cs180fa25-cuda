#include <cstdio>
#include <cuda_runtime.h>

#include "device_info.h"

__global__ void VecAddKernel(int *A, int *B, int *C, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}

void print(int *arr, int N) {
  for (int i = 0; i < N; ++i) {
    printf("%d, ", *arr);
    arr++;
  }
  printf("\n");
}

// this is a sample program that adds 2 really big vectors together into the C
// vector
int VecAdd() {
  int N = 48;
  size_t size = N * sizeof(int);

  // Allocates Memory on Host
  int *h_A = (int *)malloc(size);
  int *h_B = (int *)malloc(size);
  int *h_C = (int *)malloc(size);

  // Initialize Data
  for (int i = 0; i < N; i++) {
    h_A[i] = i * 1.0f;
    h_B[i] = i * 2.0f;
  }

  // Allocate GLOBAL Memory on Device
  int *d_A;
  int *d_B;
  int *d_C;
  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);

  // Copy Memory From Host To Device
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  // Launch kernel
  VecAddKernel<<<1024 * 1024 * 2, 1024>>>(d_A, d_B, d_C, N);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    return 1;
  }

  // Copy Memory from Device to Host
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  // Print Stuff
  print(h_A, N);
  print(h_B, N);
  print(h_C, N);

  free(h_A);
  free(h_B);
  free(h_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  printf("Kernel launched successfully!");
  return 0;
}
