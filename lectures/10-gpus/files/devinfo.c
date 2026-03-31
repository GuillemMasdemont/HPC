// informacije o napravi
// prevajanje:
//      module load CUDA
//      nvcc -o devinfo devinfo.c
// izvajanje:
//      srun --reservation=fri --partition=gpu --gpus=1 ./devinfo

#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "helper_cuda.h"

int main(int argc, char **argv) {
    
    // Get number of GPUs
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        printf("cudaGetDeviceCount error %d\n-> %s\n", error, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Get propreties of each device
    for (int dev = 0; dev < deviceCount; dev++) {

        struct cudaDeviceProp prop;
        int valueCache, valueBlocks;

        cudaGetDeviceProperties(&prop, dev);
        cudaDeviceGetAttribute (&valueCache, cudaDevAttrL2CacheSize, dev);
        cudaDeviceGetAttribute (&valueBlocks, cudaDevAttrMaxBlocksPerMultiprocessor, dev);  // works in newer CUDA versions
        
        printf("\n\n======= Device %d: \"%s\" =======\n\n", dev, prop.name);
        printf("  CUDA Architecture:                                      %s, %d.%d\n", _ConvertSMVer2ArchName(prop.major, prop.minor), prop.major, prop.minor);
        printf("\n");
        printf("  GPU clock rate (MHz):                                   %d\n", prop.clockRate/1000);
        printf("  Memory clock rate (MHz):                                %d\n", prop.memoryClockRate/1000);
        printf("  Memory bus width (bits):                                %d\n", prop.memoryBusWidth);
        printf("  Peak memory bandwidth (GB/s):                           %.0f\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        printf("\n");
        printf("  Number of compute units:                                %d\n", prop.multiProcessorCount);
        printf("  Number of processing elements per compute unit:         %d\n", _ConvertSMVer2Cores(prop.major, prop.minor));
        printf("  Total number of processing elemets:                     %d\n", _ConvertSMVer2Cores(prop.major, prop.minor) * prop.multiProcessorCount);
        printf("\n");
        printf("  Total amount of global memory (GB):                     %.0f\n", prop.totalGlobalMem / 1073741824.0f);
        printf("  Size of L2 cache (MB):                                  %.0f\n", valueCache/1048576.0f);
        printf("  Total amount of local memory per compute unit (kB):     %d\n", prop.sharedMemPerMultiprocessor/1024);
        printf("  Total amount of local memory per thread block (kB):     %zu\n", prop.sharedMemPerBlock/1024);
        printf("  Maximum number of registers per compute unit:           %d\n", prop.regsPerMultiprocessor);
        printf("  Maximum number of registers available per thread block: %d\n", prop.regsPerBlock);
        printf("\n");
        printf("  Maximum number of threads per compute unit:             %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per thread block:             %d\n", prop.maxThreadsPerBlock);
        printf("  Maximum number of blocks per compute unit:              %d\n", valueBlocks);  
        printf("  Thread warp size:                                       %d\n", prop.warpSize);
        printf("\n");
        printf("  Maximum size of a thread block (x,y,z):                 (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Maximum size of a thread grid (x,y,z):                  (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
}
