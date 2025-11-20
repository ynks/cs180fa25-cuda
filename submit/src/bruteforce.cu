#include "bruteforce.h"
#include <cstdio>
#include <ctime>
#include <cstring>
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>

#define N 36ull

namespace brute {

__device__ char d_password[100];

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
__device__ unsigned long long winner_thread_id = 0;

__device__ inline bool should_stop() { return atomicAdd(&stop_flag, 0) != 0; }

__global__ void Kernel(unsigned long long total, int passwordLength) {
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

    if (mem_compare(string, d_password, passwordLength)) {
      if (atomicCAS(&stop_flag, 0, 1) == 0) {
        winner_thread_id = tid;
      }
      return;
    }
  }
}

__host__ Result StartKernel(const char* target, int maxLength) {
  Result result = {false, "", 0, 0, 0, 0, 0, 0.0, 0};
  
  if (maxLength > 8 || maxLength < 1) {
    maxLength = 8;
  }
  
  int targetLen = strlen(target);
  if (targetLen > maxLength) {
    return result;
  }
  
  char hostPassword[100] = {0};
  strncpy(hostPassword, target, 99);
  cudaMemcpyToSymbol(d_password, hostPassword, 100);
  
  int h_stop_flag = 0;
  cudaMemcpyToSymbol(stop_flag, &h_stop_flag, sizeof(int));
  
  auto startTime = std::chrono::high_resolution_clock::now();

  dim3 threadsPerBlock(512);
  dim3 numBlocks(480);
  
  result.blocksX = numBlocks.x;
  result.blocksY = numBlocks.y;
  result.blocksZ = numBlocks.z;
  result.threadsPerBlock = threadsPerBlock.x;

  unsigned long long totalPossibilities = 1;
  for (int i = 0; i < targetLen; i++) {
    totalPossibilities *= N;
  }
  
  result.totalAttempts = totalPossibilities;

  Kernel<<<numBlocks, threadsPerBlock>>>(totalPossibilities, targetLen);
  cudaDeviceSynchronize();

  int found_flag;
  cudaMemcpyFromSymbol(&found_flag, stop_flag, sizeof(int));
  
  if (found_flag) {
    result.found = true;
    strncpy(result.password, target, 99);
    result.password[99] = '\0';
    cudaMemcpyFromSymbol(&result.winnerThreadId, winner_thread_id, sizeof(unsigned long long));
  }
  
  auto endTime = std::chrono::high_resolution_clock::now();
  result.elapsedTime = std::chrono::duration<double>(endTime - startTime).count();
  
  return result;
}
} // namespace brute
