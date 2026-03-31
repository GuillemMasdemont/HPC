// Poisson equation solver using finite difference method on CPU
//      d2u/dx2 + d2u/dy2 = 0
//      u(x,y) = 100 on three borders, 0 inside the domain at the beginning
// compile and run:
//      nvcc -o heat0 heat0.cu
//      srun --reservation=fri --partition=gpu --gpus=1 ./heat0 8192

#include <stdio.h>
#include "cuda.h"
#include "helper_cuda.h"

#define MAXITERS	10
#define BLOCK_SIZE	32

void heatStep(float* surfaceOut, float* surfaceIn, int width, int height) {

    for(int y = 1; y < height-1; y++) {
        for(int x = 1; x < width-1; x++) {
            surfaceOut[y * width + x] = 0.25 * (
                surfaceIn[y * width + (x - 1)] + surfaceIn[y * width + (x + 1)] +
                surfaceIn[(y - 1) * width + x] + surfaceIn[(y + 1) * width + x]
                );
        }
    }
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
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaEventRecord(start));

    float* h_surfaceNew = (float*)malloc(N * N * sizeof(float));

    for(int i = 0; i < MAXITERS; i++) {
        heatStep(h_surfaceNew, h_surface, N, N);
        float *temp = h_surface;
        h_surface = h_surfaceNew;
        h_surfaceNew = temp;
    }

    free(h_surfaceNew);

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));

   	free(h_surface);

    float time;
    checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
	printf("Time: host = %f s\n", time/1000.0);

	return 0;
}
