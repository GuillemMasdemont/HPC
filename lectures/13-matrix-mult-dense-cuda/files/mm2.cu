// nvcc -Xcompiler -fopenmp -o mm2 mm2.cu
// srun --reservation=fri --partition=gpu --gpus=1 ./mm2 2048
// block multiplication algorithm -- warp assignment DOES NOT match row-major matrix format

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "omp.h"
#include "cuda.h"
#include "helper_cuda.h"


#define BLOCK_SIZE	16


// gpu kernel
__global__ void matrixMultiply(float *A, float *B, float *C, int wA, int hA, int wB, int hB) {

    // shared memory allocation
    __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];

    // block index
    int bj = blockIdx.y;
    int bi = blockIdx.x;
 
    // thread index
    int tj = threadIdx.y;
    int ti = threadIdx.x;
 
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
	float sum = 0.0f;

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
	float *h_A = (float *)malloc(hA*wA*sizeof(float));
    float *h_B = (float *)malloc(hB*wB*sizeof(float));
    float *h_C_cpu = (float *)malloc(hA*wB*sizeof(float));
    float *h_C_gpu = (float *)malloc(hA*wB*sizeof(float));

    // initialization of A and B
	srand((int)time(NULL));
	for(int i=0; i<hA; i++) 
		for(int j=0; j<wA; j++)
			h_A[i*wA+j] = rand()/(float)RAND_MAX;
	for(int i=0; i<hB; i++) 
		for(int j=0; j<wB; j++)
			h_B[i*wB+j] = rand()/(float)RAND_MAX;
	for(int i=0; i<hA; i++) 
		for(int j=0; j<wB; j++)
			h_C_cpu[i*wB+j] = 0.0;

	double d_dt = omp_get_wtime();

    // allocate memory @ device and transfer data from host
	float *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc((void **)&d_A, hA*wA * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_B, hB*wB * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_C, hA*wB * sizeof(float)));

    // data transfer to device
    checkCudaErrors(cudaMemcpy(d_A, h_A, hA*wA*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, hB*wB*sizeof(float), cudaMemcpyHostToDevice));

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
    checkCudaErrors(cudaMemcpy(h_C_gpu, d_C, hA*wB*sizeof(float), cudaMemcpyDeviceToHost));

	// release memory @ device
	checkCudaErrors(cudaFree(d_A));
	checkCudaErrors(cudaFree(d_B));
	checkCudaErrors(cudaFree(d_C));

	d_dt = omp_get_wtime() - d_dt;

    // results host
	double h_dt = omp_get_wtime();
    if (argc > 2)
        for(int i=0; i<hA; i++)
            for(int j=0; j<wB; j++)
                for(int k=0; k<wA; k++)
                    h_C_cpu[i*wB+j] += h_A[i*wA+k] * h_B[k*wB+j];
	h_dt = omp_get_wtime() - h_dt;

	printf("host: %lfs, device: %lfs (%lfs), speedup: %lf\n", h_dt, d_dt, d_dt_kernel, h_dt/d_dt);

	// check for correctness
	if(argc > 2)
		for(int i=0; i<hA; i++)
			for(int j=0; j<wB; j++)
				printf("C[%d,%d] = %f : %f\n", i, j, h_C_cpu[i*wB+j], h_C_gpu[i*wB+j]);
 
    // release memory @ host
	free(h_A);
	free(h_B);
	free(h_C_cpu);
	free(h_C_gpu);

    return 0;
}
