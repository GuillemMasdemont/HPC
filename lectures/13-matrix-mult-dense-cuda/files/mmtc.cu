// nvcc -arch=sm_70 -o mmtc mmtc.cu
//      sm_70 is the minimum architecture that supports WMMA (Tensor Cores)  
// srun --reservation=fri --partition=gpu --gpus=1 ./mmtc 2048
// block multiplication algorithm -- warp assignment matches row-major matrix format 
//      solution with tensor cores (WMMA API)
//      whole warp is needed to perform one WMMA operation
//      1D organization of threads in a block, each warp computes one 16×16 tile of C

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "cuda.h"
#include "helper_cuda.h"
#include "cuda_fp16.h"
#include <mma.h>


using namespace nvcuda;

                                            
#define WMMA_Y      16                      // WMMA tile height, fixed by HW
#define WMMA_X      16                      // WMMA tile width, fixed by HW      
#define WMMA_K      16                      // WMMA inner dimension (dot product), fixed by HW
#define WARPS_Y     2                       // number of warps per block in Y dimension
#define WARPS_X     2                       // number of warps per block in X dimension
#define TILE_Y      (WMMA_Y*WARPS_Y)        // 32
#define TILE_X      (WMMA_X*WARPS_X)        // 32
#define BLOCK_SIZE  (WARPS_Y*WARPS_X*32)    // 128


__global__ void matrixMultiply(half *A, half *B, half *C, int wA, int hA, int wB, int hB) {

    // shared memory — one K-slice of the block tile
    __shared__ half tileA[TILE_Y][WMMA_K];  // 32×16
    __shared__ half tileB[WMMA_K][TILE_X];  // 16×32

    // top-left corner of this block's tile in C
    const int aBegin = blockIdx.y * TILE_Y;
    const int bBegin = blockIdx.x * TILE_X;

    const int warpId  = threadIdx.x / 32;
    const int aTileBegin = (warpId/WARPS_X) * WMMA_Y;
    const int bTileBegin = (warpId%WARPS_X) * WMMA_X;

    // prepare per-warp WMMA fragments
    // wmma::matrix_a, wmma::matrix_b, wmma::accumulator determine operands in C = A x B + C
    wmma::fragment<wmma::matrix_a,    WMMA_Y, WMMA_X, WMMA_K, half, wmma::row_major> fragA;
    wmma::fragment<wmma::matrix_b,    WMMA_Y, WMMA_X, WMMA_K, half, wmma::row_major> fragB;
    wmma::fragment<wmma::accumulator, WMMA_Y, WMMA_X, WMMA_K, half> fragC;
    wmma::fill_fragment(fragC, 0.0f);

    // walk WMMA_K-wide sub-tiles across k (dimension x in A, dimension y in B)
    for (int k = 0; k < wA; k += WMMA_K) {

        // tileA loading: 32x16=512 elements, each thread loads 4 times
        for (int idx = threadIdx.x; idx < TILE_Y*WMMA_K; idx += BLOCK_SIZE) {
            int iTile = idx / WMMA_K;
            int jTile = idx % WMMA_K;
            tileA[iTile][jTile] = A[wA*(aBegin+iTile) + (k+jTile)];
        }
        // tile B loading: 16x32=512 elements, each thread loads 4 times
        for (int idx = threadIdx.x; idx < WMMA_K*TILE_X; idx += BLOCK_SIZE) {
            int iTile = idx / TILE_X;
            int jTile = idx % TILE_X;
            tileB[iTile][jTile] = B[wB*(k+iTile) + (bBegin+jTile)];
        }
        __syncthreads();

        // each warp loads its 16×16 fragment and multiplies using tensor cores
        wmma::load_matrix_sync(fragA, (half *)&tileA[aTileBegin][0], WMMA_K);
        wmma::load_matrix_sync(fragB, (half *)&tileB[0][bTileBegin], TILE_X);
        wmma::mma_sync(fragC, fragA, fragB, fragC);

        __syncthreads();
    }

    // store warp result to global C
    int j = aBegin + aTileBegin;
    int i = bBegin + bTileBegin;
    wmma::store_matrix_sync(&C[wB*i + j], fragC, wB, wmma::mem_row_major);
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

    dim3 gridSize((wB+TILE_X-1)/TILE_X, (hA+TILE_Y-1)/TILE_Y);
    dim3 blockSize(BLOCK_SIZE);
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
