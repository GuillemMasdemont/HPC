// nvcc -o dotprod3 dotprod3.cu
// srun --reservation=fri --partition=gpu --gpus=1 ./dotprod3 16777216 256
// improvement: reduction (increasing stride)

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda.h"
#include "helper_cuda.h"

#define THREADS_PER_BLOCK_MAX 1024

__global__ void dotprod(float *a, float *b, float *p, int n)
{
    __shared__ float part[THREADS_PER_BLOCK_MAX];

    part[threadIdx.x] = 0.0;

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    while (tid < n)
    {
        part[threadIdx.x] += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    __syncthreads();

    int idxStep;
	for(idxStep = 1; idxStep < blockDim.x ; idxStep *= 2 )
	{
		if (threadIdx.x % (idxStep*2) == 0)
			part[threadIdx.x] += part[threadIdx.x+idxStep];
		__syncthreads();
	}

    if (threadIdx.x == 0)
        p[blockIdx.x] = part[0];

}

int main(int argc, char *argv[])
{
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
	h_p = (float *)malloc(gridsize.x * sizeof(float));

	// vectors initialization
    srand(time(NULL));
	for (int i = 0; i < size; i++)
	{
		h_a[i] = (double)rand()/RAND_MAX;
		h_b[i] = (double)rand()/RAND_MAX;;
	}

    // memory allocation @ device
    checkCudaErrors(cudaMalloc((void **)&d_a, size * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_b, size * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_p, gridsize.x * sizeof(float)));

    // data transfer to device
    checkCudaErrors(cudaMemcpy(d_a, h_a, size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, h_b, size * sizeof(float), cudaMemcpyHostToDevice));

	// computing
    float elapsedTime;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));

    dotprod<<<gridsize,blocksize>>>(d_a, d_b, d_p, size);
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));

    // data transfer from device
    checkCudaErrors(cudaMemcpy(h_p, d_p, gridsize.x * sizeof(float), cudaMemcpyDeviceToHost));

    // memory release @ device
    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_p));

    // dot product @ device        
    float dotProdGPU = 0.0;
    for(int i =0; i< gridsize.x; i++)
        dotProdGPU += h_p[i];

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
