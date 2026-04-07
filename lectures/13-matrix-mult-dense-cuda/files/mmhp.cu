// nvcc -arch=sm_70 -o mmh mmhp.cu
// srun --reservation=fri --partition=gpu --gpus=1 ./mmhp 2048
// block multiplication algorithm -- warp assignment matches row-major matrix format
// half precision 

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "cuda.h"
#include "helper_cuda.h"
#include "cuda_fp16.h"


#define BLOCK_SIZE	16


// gpu kernel
__global__ void matrixMultiply(half *A, half *B, half *C, int wA, int hA, int wB, int hB) {

    // shared memory allocation
    __shared__ half tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ half tileB[BLOCK_SIZE][BLOCK_SIZE];

    // block index
    int bi = blockIdx.y;
    int bj = blockIdx.x;
 
    // thread index
    int ti = threadIdx.y;
    int tj = threadIdx.x;
 
    // first element in a row of matrix A
    int aBegin = wA * (BLOCK_SIZE * bi);
    // first element in a row + 1
    int aEnd   = aBegin + wA;
	// step size (block) 
    int aStep  = BLOCK_SIZE;
    // first element in a block of matrix B
    int bBegin = BLOCK_SIZE * bj;
    // step to the first element in the next block of B
    int bStep  = BLOCK_SIZE * wB;
	// first element in a block of matrix C
    int cBegin = bStep * bi + aStep * bj;
  
    // initialize sum
	half sum = 0.0;

    // go over all blocks of A and B
    for (int a = aBegin, b = bBegin; a < aEnd; a += aStep, b += bStep) {
		
        // transfer data to the shared memory
        tileA[ti][tj] = A[a + wA * ti + tj];
        tileB[ti][tj] = B[b + wB * ti + tj];
 
		__syncthreads();
        
		// multiply blocks in the shared memory
        for (int k = 0; k < BLOCK_SIZE; k++)
            sum += tileA[ti][k] * tileB[k][tj];
		
		__syncthreads();
    }

	// write the result to the global memory
    C[cBegin + wB * ti + tj] = sum;
}


// cpu main routine
int main(int argc, char *argv[]) {
    
	int size = atoi(argv[1]);
	
	int hA = size;
	int wA = size;
	int hB = wA;
	int wB = size;

	// memory allocation
	half *h_A = (half *)malloc(hA*wA*sizeof(half));
    half *h_B = (half *)malloc(hB*wB*sizeof(half));
    half *h_C = (half *)malloc(hA*wB*sizeof(half));

    // initialization of A and B
	srand((int)time(NULL));
	for(int i=0; i<hA; i++) 
		for(int j=0; j<wA; j++)
			h_A[i*wA+j] = __float2half(rand()/(float)RAND_MAX);
	for(int i=0; i<hB; i++) 
		for(int j=0; j<wB; j++)
			h_B[i*wB+j] = __float2half(rand()/(float)RAND_MAX);

    // allocate memory @ device and transfer data from host
	half *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc((void **)&d_A, hA*wA * sizeof(half)));
    checkCudaErrors(cudaMalloc((void **)&d_B, hB*wB * sizeof(half)));
    checkCudaErrors(cudaMalloc((void **)&d_C, hA*wB * sizeof(half)));

    // data transfer to device
    checkCudaErrors(cudaMemcpy(d_A, h_A, hA*wA*sizeof(half), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, hB*wB*sizeof(half), cudaMemcpyHostToDevice));

    // computation
	float d_dt_kernel;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

	dim3 gridSize((wB-1)/BLOCK_SIZE+1, (hA-1)/BLOCK_SIZE+1);
	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    checkCudaErrors(cudaEventRecord(start));
    matrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, hA, wA, hB, wB);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));

    checkCudaErrors(cudaEventElapsedTime(&d_dt_kernel, start, stop));
    d_dt_kernel /= 1000;

    // data transfer from device
    checkCudaErrors(cudaMemcpy(h_C, d_C, hA*wB*sizeof(half), cudaMemcpyDeviceToHost));

	// release memory @ device
	checkCudaErrors(cudaFree(d_A));
	checkCudaErrors(cudaFree(d_B));
	checkCudaErrors(cudaFree(d_C));

	printf("device kernel: %lfs\n", d_dt_kernel);
 
    if (argc > 2) {
        for (int i = 0; i < hA; i++) {
            for(int j = 0; j < wB; j++) {
                float cpu = 0.0;
                for (int k = 0; k < wA; k++)
                    cpu += __half2float(h_A[i * wA + k]) * __half2float(h_B[k * wB + j]);
                printf("(%d, %d):%f - %f\n", i, j, cpu, __half2float(h_C[i * wB + j]));
            }
        }
    }

    // release memory @ host
	free(h_A);
	free(h_B);
	free(h_C);

    return 0;
}
