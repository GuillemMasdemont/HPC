// nvcc -o scan1 scan1.cu
// srun --reservation=fri --partition=gpu --gpus=1 ./scan1

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda.h"
#include "helper_cuda.h"

#define SIZE				(1024)
#define THREADS_PER_BLOCK	(256)

// kernels

__global__ void scan(float *in, float *out, float *blockSum, int size) {		
    
	__shared__ float tile[2*THREADS_PER_BLOCK];
	
	float *tileIn = &tile[0];
	float *tileOut = &tile[THREADS_PER_BLOCK];
	float *tmp;

	int lid = threadIdx.x;
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	
	// Read to local memory
	if (gid < size)
		tileIn[lid] = in[gid];
	else
		tileIn[lid] = 0.0f;
	tileOut[lid] = 0.0f;

	__syncthreads();

	for (int step = 1; step < blockDim.x; step <<= 1) {
		tileOut[lid] = tileIn[lid];
		if(lid >= step)
			tileOut[lid] += tileIn[lid - step];

		__syncthreads();

		tmp = tileIn;				
		tileIn = tileOut;
		tileOut = tmp;
	}

	if (gid < size)
		out[gid] = tileIn[lid];

	if (lid == 0)
		blockSum[blockIdx.x] = tileIn[blockDim.x - 1];
}														

__global__ void add(float *out, float *blockSum, int size) {		
	
	__shared__ float tile[THREADS_PER_BLOCK];

	int lid = threadIdx.x;
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
 
	// calculate tile[0] = blockSum[0] + ... + blockSum[blockIdx.x - 1]
	// prepare data
	tile[lid] = 0.0f;
	int idx = lid;
	while (idx < blockIdx.x) {
		tile[lid] += blockSum[idx];
		idx += blockDim.x;
	}

	__syncthreads();

	// reduction
	int floorPow2 = blockDim.x;
	while (floorPow2 & (floorPow2-1))
		floorPow2 &= floorPow2-1;
    if (blockDim.x != floorPow2) {
		if (lid >= floorPow2)
            tile[lid - floorPow2] += tile[lid];
		__syncthreads();
    }
	for(int i = (floorPow2 >> 1); i > 0; i >>= 1) {
		if(lid < i) 
			tile[lid] += tile[lid + i];
		__syncthreads();
	}	

	// addition
	if (gid < size)
		out[gid] += tile[0];		
}														


int main(int argc, char *argv[]) {
    
    float *h_in, *h_out;
    float *d_in, *d_out, *d_blockSum;

	int vectorSize = SIZE;

	// Allocate memory
	h_in = (float*)malloc(vectorSize*sizeof(float));
    h_out = (float*)malloc(vectorSize*sizeof(float));

    // Initialize vectors
	srand((int)time(NULL));
	for(int i = 0; i < vectorSize; i++) {
        h_in[i] = rand()/(float)RAND_MAX;
        h_out[i] = rand()/(float)RAND_MAX;
    }
 
	// Thread organization
    dim3 blockSize(THREADS_PER_BLOCK);
	dim3 gridSize((vectorSize-1)/blockSize.x+1);		

    // allocate memory @ device
    checkCudaErrors(cudaMalloc((void **)&d_in, vectorSize * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_out, vectorSize * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_blockSum, gridSize.x * sizeof(float)));

    // data transfer to device
    checkCudaErrors(cudaMemcpy(d_in, h_in, vectorSize * sizeof(float), cudaMemcpyHostToDevice));

    // computation
    scan<<<gridSize, blockSize>>>(d_in, d_out, d_blockSum, vectorSize);
	checkCudaErrors(cudaGetLastError());
    add<<<gridSize, blockSize>>>(d_out, d_blockSum, vectorSize);
	checkCudaErrors(cudaGetLastError());

    // data transfer from device
    checkCudaErrors(cudaMemcpy(h_out, d_out, vectorSize * sizeof(float), cudaMemcpyDeviceToHost));

    // memory release @ device
    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_out));
    checkCudaErrors(cudaFree(d_blockSum));

    // results
    float sum = 0.0;
    for (int i = 0; i < vectorSize; i++) {
        sum += h_in[i];
        printf("%d: %f =? %f\n", i, sum, h_out[i]);
    }

    // memory release @ host
    free(h_in);
    free(h_out);

    return 0;
}
