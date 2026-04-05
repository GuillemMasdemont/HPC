// nvcc -o dotprod0 dotprod0.cu
// srun --reservation=fri --partition=gpu --gpus=1 ./dotprod0 16777216 256

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda.h"
#include "helper_cuda.h"


__global__ void dotprod(float *a, float *b, float *c, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    while (tid < n) {
        c[tid] = a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

int main(int argc, char *argv[]) {

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    
    // arguments
	int size = atoi(argv[1]);
    int threadsperblock = atoi(argv[2]);

    // GPU thread organization
    dim3 gridsize((size-1)/threadsperblock + 1);
    dim3 blocksize(threadsperblock);

	// memory allocation @ host
	h_a = (float *)malloc(size * sizeof(float));
	h_b = (float *)malloc(size * sizeof(float));
	h_c = (float *)malloc(size * sizeof(float));

	// vectors initialization
    srand(time(NULL));
	for (int i = 0; i < size; i++) {
		h_a[i] = (double)rand()/RAND_MAX;
		h_b[i] = (double)rand()/RAND_MAX;;
	}

    // memory allocation @ device
    checkCudaErrors(cudaMalloc((void **)&d_a, size * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_b, size * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_c, size * sizeof(float)));

    // data transfer to device
    checkCudaErrors(cudaMemcpy(d_a, h_a, size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, h_b, size * sizeof(float), cudaMemcpyHostToDevice));

	// computing
    dotprod<<<gridsize,blocksize>>>(d_a, d_b, d_c, size);
    checkCudaErrors(cudaGetLastError());

    // data transfer from device
    checkCudaErrors(cudaMemcpy(h_c, d_c, size * sizeof(float), cudaMemcpyDeviceToHost));

    // memory release @ device
    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_c));

    // dot product @ device        
    float dotProdGPU = 0.0;
    for(int i =0; i< size; i++)
        dotProdGPU += h_c[i];

    // dot product @ host
    float dotProdCPU = 0.0;
    for(int i =0; i< size; i++)
        dotProdCPU += h_a[i] * h_b[i];

    // memmory release @ host
	free(h_a);
	free(h_b);
	free(h_c);

	// Results
    printf("dotprod(CPU): %f\ndotprod(GPU): %f\n", dotProdCPU, dotProdGPU);

	return 0;
}
