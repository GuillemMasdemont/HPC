// Poisson equation solver using finite difference method on CPU
//      T(x,y) = 100 on three borders, 0 inside the domain at the beginning
//      d2T/dx2 + d2T/dy2 = 0
// compile and run:
//      nvcc -o heat3 heat3.cu
//      srun --reservation=fri --partition=gpu --gpus=1 ./heat3 512

#include <stdio.h>
#include "cuda.h"
#include "helper_cuda.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define MAXITERS	1000000
#define BLOCK_SIZE	16

__global__ void heatStep(float* surfaceOut, float* surfaceIn, int width, int height) {

    extern __shared__ float tile[];

    int gx = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int gy = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int idx = gy * width + gx;

    int interior = gx > 0 && gx < width - 1 && gy > 0 && gy < height - 1;

    int tx = threadIdx.x + 1;   // tile indices 0 .. BLOCK_SIZE + 1
    int ty = threadIdx.y + 1;

    int tileWidth = BLOCK_SIZE + 2;

    // halo loading 
    if (interior) { 
        // central area loading 
        tile[ty*tileWidth + tx] = surfaceIn[idx];
        // halo loading
        if (tx == 1)
            tile[ty*tileWidth + tx-1] = surfaceIn[idx-1];
        if (tx == BLOCK_SIZE)
            tile[ty*tileWidth + tx+1] = surfaceIn[idx+1];
        if (ty == 1)
            tile[(ty-1)*tileWidth + tx] = surfaceIn[idx-width];
        if (ty == BLOCK_SIZE)
            tile[(ty+1)*tileWidth + tx] = surfaceIn[idx+width];
    }

    __syncthreads();

    // only interior cells are updated
    if(interior) {
        surfaceOut[idx] = 0.25 * (
                tile[ty*tileWidth + tx-1] + tile[ty*tileWidth + tx+1] +
                tile[(ty-1)*tileWidth + tx] + tile[(ty+1)*tileWidth + tx]
        );
    }
}
 
int main(int argc, char *argv[]) {
	
	int N = atoi(argv[1]);
    if (N % BLOCK_SIZE != 0) {
        printf("Surface size must be a multiple of %d\n", BLOCK_SIZE);
        return -1;
    }

	// init surface with added halo
	float* h_surface = (float*)malloc((N+2) * (N+2) * sizeof(float));
	for(int i = 0; i < (N+2) * (N+2); i++)
		h_surface[i] = 0.0;
	for(int i = 1; i < N+1; i++) {
		h_surface[i * (N+2)] = 100.0;
		h_surface[i * (N+2) + (N+1)] = 100.0;
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
    checkCudaErrors(cudaMalloc((void **)&d_surface, (N+2) * (N+2) * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_surfaceNew, (N+2) * (N+2) * sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_surface, h_surface, (N+2) * (N+2) * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_surfaceNew, h_surface, (N+2) * (N+2) * sizeof(float), cudaMemcpyHostToDevice));

	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((N-1)/BLOCK_SIZE+1, (N-1)/BLOCK_SIZE+1);
    int localMemSize = (BLOCK_SIZE+2)*(BLOCK_SIZE+2)*sizeof(float);

	checkCudaErrors(cudaEventRecord(startKernel));
	for (int i = 0; i < MAXITERS; i++) {
		heatStep<<<grid, block, localMemSize>>>(d_surfaceNew, d_surface, N+2, N+2);
        checkCudaErrors(cudaGetLastError());
        // Swap pointers
        d_temp = d_surface;
        d_surface = d_surfaceNew;
        d_surfaceNew = d_temp;
	}
	checkCudaErrors(cudaEventRecord(stopKernel));
    checkCudaErrors(cudaEventSynchronize(stopKernel));
	
	checkCudaErrors(cudaMemcpy(h_surface, d_surface, (N+2) * (N+2) * sizeof(float), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_surface));
	checkCudaErrors(cudaFree(d_surfaceNew));

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));

    unsigned char* img = (unsigned char*)malloc((N+2) * (N+2) * sizeof(unsigned char));
    for(int i = 0; i < (N+2) * (N+2); i++)
        img[i] = 255 - (unsigned char)(h_surface[i] * 255.0 / 100.0);
   	stbi_write_png("heat.png", N+2, N+2, 1, img, N+2);
    free(img);

   	free(h_surface);

    float time, timeKernel;
    checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
    checkCudaErrors(cudaEventElapsedTime(&timeKernel, startKernel, stopKernel));
	printf("Time: device = %f s\n", time/1000.0);
    printf("Time: kernel = %f s\n\n", timeKernel/1000.0);

	return 0;
}
