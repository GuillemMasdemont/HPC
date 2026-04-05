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

__global__ void scan(float *in, float *out, float *blocksum, int size)
{		
    __shared__ float loc[2*THREADS_PER_BLOCK];

	int lid = threadIdx.x;
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int din = 0;			// displacement of local input array in loc
	int dout = blockDim.x;	// displacement of local output array in loc

	// Read to local memory
	if (gid < size)
		loc[din + lid] = in[gid];
	else
		loc[din + lid] = 0.0f;
	loc[dout + lid] = 0.0f;

	__syncthreads();

	for (int offset = 1; offset < size; offset <<= 1)
	{
		loc[dout + lid] = loc[din + lid];
		if(lid >= offset)
			loc[dout + lid] += loc[din + lid - offset];

		__syncthreads();

		din = blockDim.x - din;		// swap in <-> out
		dout = blockDim.x - dout;
	}

	din = blockDim.x - din;			// swap in <-> out
	dout = blockDim.x - dout;

	if (gid < size)
		out[gid] = loc[dout + lid];

	if (lid == 0)
		blocksum[blockIdx.x] = loc[dout + blockDim.x - 1];
}														

__global__ void add(float *out, float *blocksum, int size)
{		
	__shared__ float loc[THREADS_PER_BLOCK];

	int lid = threadIdx.x;
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
 
	// calculate loc[0] = blocksum[0] + ... + blocksum[blockIdx.x - 1]
	// prepare data
	loc[lid] = 0.0f;
	int idx = lid;
	while (idx < blockIdx.x)
	{
		loc[lid] += blocksum[idx];
		idx += blockDim.x;
	}

	__syncthreads();

	// reduction
	int floorPow2 = exp2((float)(int)log2((float)blockDim.x));
    if (blockDim.x != floorPow2)										
	{
		if (lid >= floorPow2)
            loc[lid - floorPow2] += loc[lid];
		__syncthreads();
    }
	for(int i = (floorPow2 >> 1); i>0; i >>= 1) 
	{
		if(lid < i) 
			loc[lid] += loc[lid + i];
		__syncthreads();
	}	

	// addition
	if (gid < size)
		out[gid] += loc[0];		
}														


int main(int argc, char *argv[]) 
{
    float *h_in, *h_out;
    float *d_in, *d_out, *d_blocksum;

	int vectorSize = SIZE;

	// Allocate memory
	h_in = (float*)malloc(vectorSize*sizeof(float));
    h_out = (float*)malloc(vectorSize*sizeof(float));

    // Initialize vectors
	srand((int)time(NULL));
	for(int i = 0; i < vectorSize; i++) 
	{
        h_in[i] = rand()/(float)RAND_MAX;
        h_out[i] = rand()/(float)RAND_MAX;
    }
 
	// Thread organization
    dim3 blocksize(THREADS_PER_BLOCK);
	dim3 gridsize((vectorSize-1)/blocksize.x+1);		

    // allocate memory @ device
    checkCudaErrors(cudaMalloc((void **)&d_in, vectorSize * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_out, vectorSize * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_blocksum, gridsize.x * sizeof(float)));

    // data transfer to device
    checkCudaErrors(cudaMemcpy(d_in, h_in, vectorSize * sizeof(float), cudaMemcpyHostToDevice));

    // computation
    scan<<<gridsize, blocksize>>>(d_in, d_out, d_blocksum, vectorSize);
	checkCudaErrors(cudaGetLastError());
    add<<<gridsize, blocksize>>>(d_out, d_blocksum, vectorSize);
	checkCudaErrors(cudaGetLastError());

    // data transfer from device
    checkCudaErrors(cudaMemcpy(h_out, d_out, vectorSize * sizeof(float), cudaMemcpyDeviceToHost));

    // memory release @ device
    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_out));
    checkCudaErrors(cudaFree(d_blocksum));

    // results
    float sum = 0.0;
    for (int i = 0; i < vectorSize; i++)
    {
        sum += h_in[i];
        printf("%d: %f =? %f\n", i, sum, h_out[i]);
    }

    // memory release @ host
    free(h_in);
    free(h_out);

    return 0;
}
