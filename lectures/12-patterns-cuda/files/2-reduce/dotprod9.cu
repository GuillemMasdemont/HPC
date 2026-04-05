// nvcc -o dotprod9 dotprod9.cu
// srun --reservation=fri --partition=gpu --gpus=1 ./dotprod9 16777216 256
// improvement: atomic add, dynamically allocated shared memory

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda.h"
#include "helper_cuda.h"

__global__ void dotprod(float *a, float *b, float *p, int n) {

    extern __shared__ float part[];

    part[threadIdx.x] = 0.0;

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    while (tid < n) {
        part[threadIdx.x] += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    __syncthreads();

	int floorPow2 = blockDim.x;
	while (floorPow2 & (floorPow2-1))
		floorPow2 &= floorPow2-1;

	if (blockDim.x != floorPow2) {
		if (threadIdx.x >= floorPow2)
			part[threadIdx.x - floorPow2] += part[threadIdx.x];
        __syncthreads();
	}

    int idxStep;
	for(idxStep = floorPow2 >> 1; idxStep > 32 ; idxStep >>= 1 ) {
		if (threadIdx.x < idxStep)
			part[threadIdx.x] += part[threadIdx.x+idxStep];
        __syncthreads();
	}
	for( ; idxStep > 0 ; idxStep >>= 1 ) {
		if (threadIdx.x < idxStep)
			part[threadIdx.x] += part[threadIdx.x+idxStep];
        __syncwarp();
	}

    if (threadIdx.x == 0)
        atomicAdd(p, part[0]);

}

int main(int argc, char *argv[]) {
    float *h_a, *h_b, *h_p;
    float *d_a, *d_b, *d_p;

    // arguments
	int size = atoi(argv[1]);
    int threadsperblock = atoi(argv[2]);

    // GPU thread organization
    dim3 gridsize((size-1)/threadsperblock + 1);
    dim3 blocksize(threadsperblock);

	// memory allocation @ host
	h_a = (float *)malloc(size * sizeof(float));
	h_b = (float *)malloc(size * sizeof(float));
	h_p = (float *)malloc(1 * sizeof(float));

	// vectors initialization
    srand(time(NULL));
	for (int i = 0; i < size; i++) {
		h_a[i] = (double)rand()/RAND_MAX;
		h_b[i] = (double)rand()/RAND_MAX;;
	}
    *h_p = 0.0;

    // memory allocation @ device
    checkCudaErrors(cudaMalloc((void **)&d_a, size * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_b, size * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_p, 1 * sizeof(float)));

    // data transfer to device
    checkCudaErrors(cudaMemcpy(d_a, h_a, size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, h_b, size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_p, h_p, 1 * sizeof(float), cudaMemcpyHostToDevice));

	// computing
    float elapsedTime;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));

    dotprod<<<gridsize,blocksize,threadsperblock*sizeof(float)>>>(d_a, d_b, d_p, size);
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));

    // data transfer from device
    checkCudaErrors(cudaMemcpy(h_p, d_p, 1 * sizeof(float), cudaMemcpyDeviceToHost));

    // memory release @ device
    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_p));

    // dot product @ device        
    float dotProdGPU = *h_p;

    // dot product @ host
    float dotProdCPU = 0.0;
    for(int i =0; i< size; i++)
        dotProdCPU += h_a[i] * h_b[i];

    // memory release @ host
	free(h_a);
	free(h_b);
	free(h_p);

	// results
    printf("dotprod(CPU): %f\ndotprod(GPU): %f\nTime(ms): %f\n", dotProdCPU, dotProdGPU, elapsedTime);

	return 0;
}
