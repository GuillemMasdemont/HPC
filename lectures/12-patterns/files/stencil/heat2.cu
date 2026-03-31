// Poisson equation solver using finite difference method on CPU
//      d2u/dx2 + d2u/dy2 = 0
//      u(x,y) = 100 on three borders, 0 inside the domain at the beginning
// compile and run:
//      nvcc -o heat2 heat2.cu
//      srun --reservation=fri --partition=gpu --gpus=1 ./heat2 8192

#include <stdio.h>
#include "cuda.h"
#include "helper_cuda.h"

#define MAXITERS	1000
#define BLOCK_SIZE	32

__global__ void heatStep(float* surfaceOut, float* surfaceIn, int width, int height) {

    __shared__ float tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx <= 0 || gx >= width - 1 || gy <= 0 || gy >= height - 1) 
        return;

    int idx = gy * width + gx;

    int lx = threadIdx.x;
    int ly = threadIdx.y;
    int tx = lx + 1;
    int ty = ly + 1;

    // Load center
    tile[ty][tx] = surfaceIn[gy * width + gx];

    // Halo loading
    if (lx == 0)         
        tile[ty][0] = surfaceIn[gy * width + (gx - 1)];
    if (lx == BLOCK_SIZE-1)   
        tile[ty][BLOCK_SIZE+1] = surfaceIn[gy * width + (gx + 1)];
    if (ly == 0)
        tile[0][tx] = surfaceIn[(gy - 1) * width + gx];
    if (ly == BLOCK_SIZE-1)
       tile[BLOCK_SIZE+1][tx] = surfaceIn[(gy + 1) * width + gx];

    __syncthreads();

    surfaceOut[idx] = 0.25 * (
        tile[ty][tx - 1] + tile[ty][tx + 1] +
        tile[ty - 1][tx] + tile[ty + 1][tx]
    );
}
 
int main(int argc, char *argv[]) {
	
	int N = atoi(argv[1]);

	// init surface
	float* h_surface = (float*)malloc(N * N * sizeof(float));
	for(int i = 0; i < N * N; i++)
		h_surface[i] = 0.0;
	for(int i = 0; i < N; i++) {
		h_surface[i * N] = 100.0;
		h_surface[i * N + N - 1] = 100.0;
		h_surface[i] = 100.0;
	}

	// timing 
    cudaEvent_t start, stop, startKernel, stopKernel;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventCreate(&startKernel));
    checkCudaErrors(cudaEventCreate(&stopKernel));

    checkCudaErrors(cudaEventRecord(start));

	float *d_surface, *d_surfaceNew, *d_temp;
    checkCudaErrors(cudaMalloc((void **)&d_surface, N * N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_surfaceNew, N * N * sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_surface, h_surface, N * N * sizeof(float), cudaMemcpyHostToDevice));

	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((N-1)/BLOCK_SIZE+1, (N-1)/BLOCK_SIZE+1);

	cudaEventRecord(startKernel);
	for (int i = 0; i < MAXITERS; i++) {
		heatStep<<<grid, block>>>(d_surfaceNew, d_surface, N, N);
        checkCudaErrors(cudaGetLastError());
        // Swap pointers
        d_temp = d_surface;
        d_surface = d_surfaceNew;
        d_surfaceNew = d_temp;
	}
	checkCudaErrors(cudaEventRecord(stopKernel));
    checkCudaErrors(cudaEventSynchronize(stopKernel));
	
	checkCudaErrors(cudaMemcpy(h_surface, d_surfaceNew, N * N * sizeof(float), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_surface));
	checkCudaErrors(cudaFree(d_surfaceNew));

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));

   	free(h_surface);

    float time, timeKernel;
    checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
    checkCudaErrors(cudaEventElapsedTime(&timeKernel, startKernel, stopKernel));
	printf("Time: device = %f s\n", time/1000.0);
    printf("Time: kernel = %f s\n\n", timeKernel/1000.0);

	return 0;
}
