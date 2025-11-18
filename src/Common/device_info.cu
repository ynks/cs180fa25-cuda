#include "device_info.h"

#include <cstdio>
#include <cuda_runtime.h>

void PrintDeviceInfo() {
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);

  if (deviceCount == 0) {
    printf("No CUDA devices found.\n");
  }

  for (int dev = 0; dev < deviceCount; ++dev) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    // clang-format off
    printf("=== Device %d: %s ===\n", dev, prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Number of multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Warp size: %d\n", prop.warpSize);

    printf("\n");
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max threads dimension (block): (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max grid size: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("\n");
    printf("Concurrent kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");
    printf("Async engine count: %d\n", prop.asyncEngineCount);
    printf("\n");
    printf("Registers per block: %d\n", prop.regsPerBlock);
    printf("Total global memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
    printf("Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Total constant memory: %zu KB\n", prop.totalConstMem / 1024);
    printf("L2 cache size: %d KB\n", prop.l2CacheSize / 1024);
    printf("Memory bus width: %d bits\n", prop.memoryBusWidth);
    printf("\n");
    printf("Can map host memory: %s\n", prop.canMapHostMemory ? "Yes" : "No");
    printf("Unified addressing (UVA): %s\n", prop.unifiedAddressing ? "Yes" : "No");
    printf("Managed memory supported: %s\n", prop.managedMemory ? "Yes" : "No");
    printf("ECC enabled: %s\n", prop.ECCEnabled ? "Yes" : "No");
    printf("\n");
    // clang-format on
  }
}
