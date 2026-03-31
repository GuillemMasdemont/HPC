// compute y = a*x+y on vectors
//      support for multiple blocks, check size to stay in the range of the allocated memory 
//      works with limited number of blocks
//  nvcc -o saxpy4 saxpy4.cu
//  srun --reservation=fri --partition=gpu --gpus=1 ./saxpy4


#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "helper_cuda.h"


#define VECTOR_SIZE 2048
#define BLOCK_SIZE 256


__global__ void saxpy(float a, float *x, float *y, int size) {    
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    while (tid < size) {
        y[tid] = a * x[tid] + y[tid];
        tid += gridDim.x * blockDim.x;
    }
}


int main(void) {
    int i;

    // Allocate space for vectors X and Y
    float *x, *y;
    checkCudaErrors(cudaMallocManaged((void **)&x, VECTOR_SIZE * sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **)&y, VECTOR_SIZE * sizeof(float)));

    // init vectors X and Y @ host
    float a = 0.5;
    for(i = 0; i < VECTOR_SIZE; i++) {
        x[i] = i;
        y[i] = i;
    }

    // Compute on device
    dim3 blockSize(BLOCK_SIZE);
    //dim3 gridSize((VECTOR_SIZE - 1)/blockSize.x + 1);
    dim3 gridSize(1);
    saxpy<<<gridSize, blockSize>>>(a, x, y, VECTOR_SIZE);
    checkCudaErrors(cudaGetLastError());

    // display result
    for(i = 0; i < VECTOR_SIZE; i++)
        printf("%f\n", y[i]);

    // free space: device
    checkCudaErrors(cudaFree(x));
    checkCudaErrors(cudaFree(y));

    return 0;
}
