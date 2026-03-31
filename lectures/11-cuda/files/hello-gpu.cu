// GPU hello world
// compilation: 
//		module load CUDA
//		nvcc -o hello-gpu hello-gpu.cu
// execution: 
//		srun --partition=gpu --gpus=1 ./hello-gpu 2 4

#include <stdio.h>
#include "cuda.h"
#include "helper_cuda.h"

__global__ void greetings(void) {
	printf("Hello from thread %d.%d!\n", blockIdx.x, threadIdx.x);
}

int main(int argc, char **argv) {

	// command line argument parsing
	int numBlocks = 0;
	int numThreads = 0;
	if (argc == 3) {
		numBlocks = atoi(argv[1]);
		numThreads = atoi(argv[2]);
	}
	if (numBlocks == 0 || numThreads == 0) {
		printf("usage:\n\t%s <number of blocks> <number of threads>\n\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	// trigger execution of the kernel on the device
	dim3 gridSize(numBlocks, 1, 1);
	dim3 blockSize(numThreads, 1, 1);
	greetings<<<gridSize, blockSize>>>();
	checkCudaErrors(cudaGetLastError());
	
	// wait all threads to finish before exiting
	checkCudaErrors(cudaDeviceSynchronize());
 
	return 0;

}
