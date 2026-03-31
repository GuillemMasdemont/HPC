// compute y = a*x+y on vectors
//      support for multiple blocks
//  nvcc -o saxpy1 saxpy1.cu
//  srun --reservation=fri --partition=gpu --gpus=1 ./saxpy1


#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "helper_cuda.h"


#define VECTOR_SIZE 2048
#define BLOCK_SIZE 256

__global__ void saxpy(float a, float *x, float *y) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    y[tid] = a * x[tid] + y[tid];
}


int main(void) {
    int i;

    // Allocate space for vectors X and Y @ host
    float a = 0.5;
    float *h_x = (float*)malloc(sizeof(float)*VECTOR_SIZE);
    float *h_y = (float*)malloc(sizeof(float)*VECTOR_SIZE);
    for(i = 0; i < VECTOR_SIZE; i++) {
        h_x[i] = i;
        h_y[i] = i;
    }

    // Allocate space for vectors X and Y @ device
    float *d_x, *d_y;
    checkCudaErrors(cudaMalloc((void **)&d_x, VECTOR_SIZE * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_y, VECTOR_SIZE * sizeof(float)));

    // Transfer data: device <-- host
    checkCudaErrors(cudaMemcpy(d_x, h_x, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_y, h_y, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Compute on device
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((VECTOR_SIZE - 1)/blockSize.x + 1);
    saxpy<<<gridSize, blockSize>>>(a, d_x, d_y);
    checkCudaErrors(cudaGetLastError());

    // Transfer data: device --> host
    checkCudaErrors(cudaMemcpy(h_y, d_y, VECTOR_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // free space: device
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));

    // display result
    for(i = 0; i < VECTOR_SIZE; i++)
        printf("%f\n", h_y[i]);

    // free space: host
    free(h_x);
    free(h_y);

    return 0;
}
