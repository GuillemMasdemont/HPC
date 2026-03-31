// nvcc -o dotprodA dotprodA.cu
// srun --reservation=fri --partition=gpu --gpus=1 ./dotprodA 16777216 256
// improvement: atomic add, managed memory

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda.h"
#include "helper_cuda.h"

__managed__ float sum;

__global__ void dotprod(float *a, float *b, int n)
{
    extern __shared__ float part[];

    part[threadIdx.x] = 0.0;

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    while (tid < n)
    {
        part[threadIdx.x] += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    __syncthreads();

	int floorPow2 = blockDim.x;
	while (floorPow2 & (floorPow2-1))
		floorPow2 &= floorPow2-1;

	if (blockDim.x != floorPow2)
	{
		if (threadIdx.x >= floorPow2)
			part[threadIdx.x - floorPow2] += part[threadIdx.x];
        __syncthreads();
	}

    int idxStep;
	for(idxStep = floorPow2 >> 1; idxStep > 32 ; idxStep >>= 1 )
	{
		if (threadIdx.x < idxStep)
			part[threadIdx.x] += part[threadIdx.x+idxStep];
        __syncthreads();
	}
	for( ; idxStep > 0 ; idxStep >>= 1 )
	{
		if (threadIdx.x < idxStep)
			part[threadIdx.x] += part[threadIdx.x+idxStep];
	}

    if (threadIdx.x == 0)
        atomicAdd(&sum, part[0]);

}

int main(int argc, char *argv[])
{
    float *a, *b;

    // arguments
	int size = atoi(argv[1]);
    int threadsperblock = atoi(argv[2]);

    // GPU thread organization
    dim3 gridsize((size-1)/threadsperblock + 1);
    dim3 blocksize(threadsperblock);

    // memory allocation @ device
    checkCudaErrors(cudaMallocManaged((void **)&a, size * sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **)&b, size * sizeof(float)));

	// vectors initialization
    srand(time(NULL));
	for (int i = 0; i < size; i++)
	{
		a[i] = (double)rand()/RAND_MAX;
		b[i] = (double)rand()/RAND_MAX;;
	}
    sum = 0.0;

	// computing
    float elapsedTime;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));

    dotprod<<<gridsize,blocksize,threadsperblock*sizeof(float)>>>(a, b, size);
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));

    // dot product @ device        
    float dotProdGPU = sum;

    // dot product @ host
    float dotProdCPU = 0.0;
    for(int i =0; i< size; i++)
        dotProdCPU += a[i] * b[i];

    // memory release
    checkCudaErrors(cudaFree(a));
    checkCudaErrors(cudaFree(b));

	// results
    printf("dotprod(CPU): %f\ndotprod(GPU): %f\nTime(ms): %f\n", dotProdCPU, dotProdGPU, elapsedTime);

	return 0;
}
