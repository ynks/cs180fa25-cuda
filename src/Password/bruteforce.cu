#include "bruteforce.h"
#include <cstdio>
#include <ctime>
#include <curand.h>
#include <curand_kernel.h>

#define N 36ull

namespace brute {

__constant__ char password[] = "99999999";

__constant__ char characters[] = {
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
    'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
};

__constant__ unsigned long long powers_of_N[] = { //
    1ull,
    N,
    N *N,
    N * N * N,
    N * N * N *N,
    N * N * N * N * N,
    N * N * N * N * N *N,
    N * N * N * N * N * N * N,
    N * N * N * N * N * N * N *N};

__device__ int mem_compare(char *a, char *b, int n) {
  for (int i = 0; i < n; ++i) {
    if (a[i] != b[i]) {
      return 0;
    }
  }
  return 1;
}

__device__ __forceinline__ void ToString(unsigned long long in, char *out) {
  out[0] = characters[(in / powers_of_N[0]) % N];
  out[1] = characters[(in / powers_of_N[1]) % N];
  out[2] = characters[(in / powers_of_N[2]) % N];
  out[3] = characters[(in / powers_of_N[3]) % N];
  out[4] = characters[(in / powers_of_N[4]) % N];
  out[5] = characters[(in / powers_of_N[5]) % N];
  out[6] = characters[(in / powers_of_N[6]) % N];
  out[7] = characters[(in / powers_of_N[7]) % N];
  out[8] = '\0';
}

__device__ int stop_flag = 0;

__device__ inline bool should_stop() { return atomicAdd(&stop_flag, 0) != 0; }

__global__ void Kernel(unsigned long long total) {
  unsigned long long tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned long long stride = gridDim.x * blockDim.x;

  curandState state;
  curand_init(tid, 0, 0, &state);

  for (unsigned long long i = tid; i < total; i += stride) {
    if (should_stop()) {
      return;
    }
    char string[9];
    ToString(i, string);

    if (i % 100000000000 == 0) {
      printf("Index %llu -> %.8s\n", i, string);
    }

    if (mem_compare(string, password, 8)) {
      printf("Found Password Index %llu -> %.8s\n", i, string);
      atomicExch(&stop_flag, 1);
    }
  }
}

__host__ void StartKernel() {

  dim3 threadsPerBlock(1024);

  dim3 numBlocks(36 * 16 * 16 * 16);

  unsigned long long totalThreads = threadsPerBlock.x;
  totalThreads *= numBlocks.x;
  totalThreads *= numBlocks.y;
  totalThreads *= numBlocks.z;

  printf("Total Threads: %llu\n", totalThreads);
  printf("Launching Kernel (Possibilities %llu)...\n", N * N * N * N * N * N * N * N);
  fflush(stdout);

  clock_t start = clock();
  Kernel<<<numBlocks, threadsPerBlock>>>(N * N * N * N * N * N * N * N);
  cudaDeviceSynchronize();

  clock_t end = clock();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }
  double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
  printf("Kernel time elapsed: %.6f seconds\n", elapsed);
  printf("Done.\n");
}
} // namespace brute
