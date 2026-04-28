// nvcc -Xcompiler -fopenmp -o spmvCSRsp spmvCSRsp.cu mtxsparse.c
// srun --partition=gpu --reservation=fri --gpus=1 spmvCSRsp data/scircuit.mtx 
// srun --partition=gpu --reservation=fri --gpus=1 spmvCSRsp data/pdb1HYS.mtx
// Parallel CSR SpMV, whole warp is operating in one row, after summation follows reduction within a warp

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "omp.h"
#include <cuda.h>
#include "helper_cuda.h"
#include "mtxsparse.h"


#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define REPEAT 1000


__global__ void mCSRxVecSer(int *rowPtr, int *col, float *data, float *vIn, float *vOut, int rows) {		
    
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
	
    if(gid < rows) {
		float sum = 0.0f;
        for (int j = rowPtr[gid]; j < rowPtr[gid + 1]; j++)
            sum += data[j] * vIn[col[j]];
		vOut[gid] = sum;
	}
}														

__global__ void mCSRxVecPar(int *rowPtr, int *col, float *data, float *vIn, float *vOut, int rows) {		
	
    __shared__ float buffer[THREADS_PER_BLOCK];

    int gid = blockDim.x * blockIdx.x + threadIdx.x;   
	int wid = gid / WARP_SIZE; 		                    // warp id
	int wlid = gid % WARP_SIZE; 	                    // local id within a warp
    int offset = (threadIdx.x / WARP_SIZE) * WARP_SIZE; // offset in buffer for current warp

    if (wid < rows) {
        buffer[offset+wlid] = 0.0f;
		for (int j = rowPtr[wid] + wlid; j < rowPtr[wid + 1]; j += WARP_SIZE)
			buffer[offset+wlid] += data[j] * vIn[col[j]];
        __syncwarp();
		if (wlid < 16) buffer[offset+wlid] += buffer[offset+wlid+16];
        __syncwarp();
		if (wlid <  8) buffer[offset+wlid] += buffer[offset+wlid+8];
        __syncwarp();
		if (wlid <  4) buffer[offset+wlid] += buffer[offset+wlid+4];
        __syncwarp();
		if (wlid <  2) buffer[offset+wlid] += buffer[offset+wlid+2];
        __syncwarp();
		if (wlid <  1) vOut[wid] = buffer[offset+wlid] + buffer[offset+wlid+1];
	}
}														


int main(int argc, char *argv[]) {

    FILE *f;
    struct mtxCOO h_mCOO;
    struct mtxCSR h_mCSR;
    int repeat;

    if (argc < 2) {
		fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
		exit(1);
	}
    else { 
        if ((f = fopen(argv[1], "r")) == NULL) 
            exit(1);
    }

    // create sparse matrices
    if (mtx_COO_create_from_file(&h_mCOO, f) != 0)
        exit(1);
    mtx_CSR_create_from_mtx_COO(&h_mCSR, &h_mCOO);

    // allocate vectors
    float *h_vecIn = (float *)malloc(h_mCOO.numCols * sizeof(float));
    for (int i = 0; i < h_mCOO.numCols; i++)
        h_vecIn[i] = 1.0;
    float *h_vecOutCSRser = (float *)calloc(h_mCSR.numRows, sizeof(float));
    float *h_vecOutCSRpar = (float *)calloc(h_mCSR.numRows, sizeof(float));
    float *h_vecOutCOO_cpu = (float *)calloc(h_mCSR.numRows, sizeof(float));

    // compute with COO
    double dtimeCOO_cpu = omp_get_wtime();
    for (repeat = 0; repeat < REPEAT; repeat++) {
        for (int i = 0; i < h_mCOO.numRows; i++)
            h_vecOutCOO_cpu[i] = 0.0;
        for (int i = 0; i < h_mCOO.numNonzero; i++)
            h_vecOutCOO_cpu[h_mCOO.row[i]] += h_mCOO.data[i] * h_vecIn[h_mCOO.col[i]];
    }
    dtimeCOO_cpu = omp_get_wtime() - dtimeCOO_cpu;


    // allocate memory on device and transfer data from host 
    // CSR
    int *d_mCSRrowPtr, *d_mCSRcol;
    float *d_mCSRdata;
    checkCudaErrors(cudaMalloc((void **)&d_mCSRrowPtr, (h_mCSR.numRows + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_mCSRcol, (h_mCSR.numNonzero + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_mCSRdata, h_mCSR.numNonzero * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_mCSRrowPtr, h_mCSR.rowPtr, (h_mCSR.numRows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_mCSRcol, h_mCSR.col, h_mCSR.numNonzero * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_mCSRdata, h_mCSR.data, h_mCSR.numNonzero * sizeof(float), cudaMemcpyHostToDevice));

    // vectors
    float *d_vecIn, *d_vecOut;
    checkCudaErrors(cudaMalloc((void **)&d_vecIn, h_mCOO.numCols * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_vecOut, h_mCOO.numRows * sizeof(float)));

	// Divide work 
    dim3 blockSize(THREADS_PER_BLOCK);
    // CSRser
    dim3 gridSizeCSRser((h_mCSR.numRows - 1) / blockSize.x + 1);
	// CSRpar
    dim3 gridSizeCSRpar((WARP_SIZE * h_mCSR.numRows - 1) / blockSize.x + 1);

	// CSRser: write, execute, read
    double dtimeCSRser = omp_get_wtime();
    for (repeat = 0; repeat < REPEAT; repeat++) {
        checkCudaErrors(cudaMemcpy(d_vecIn, h_vecIn, h_mCSR.numCols*sizeof(float), cudaMemcpyHostToDevice));
        mCSRxVecSer<<<gridSizeCSRser, blockSize>>>(d_mCSRrowPtr, d_mCSRcol, d_mCSRdata, d_vecIn, d_vecOut, h_mCSR.numRows);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaMemcpy(h_vecOutCSRser, d_vecOut, h_mCSR.numRows*sizeof(float), cudaMemcpyDeviceToHost));
    }
    dtimeCSRser = omp_get_wtime()-dtimeCSRser;

	// CSRpar: write, execute, read
    double dtimeCSRpar = omp_get_wtime();
    for (repeat = 0; repeat < REPEAT; repeat++) {
        checkCudaErrors(cudaMemcpy(d_vecIn, h_vecIn, h_mCSR.numCols*sizeof(float), cudaMemcpyHostToDevice));
        mCSRxVecPar<<<gridSizeCSRpar, blockSize>>>(d_mCSRrowPtr, d_mCSRcol, d_mCSRdata, d_vecIn, d_vecOut, h_mCSR.numRows);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaMemcpy(h_vecOutCSRpar, d_vecOut, h_mCSR.numRows*sizeof(float), cudaMemcpyDeviceToHost));
    }
    dtimeCSRpar = omp_get_wtime()-dtimeCSRpar;

    checkCudaErrors(cudaFree(d_mCSRrowPtr));
    checkCudaErrors(cudaFree(d_mCSRcol));
    checkCudaErrors(cudaFree(d_mCSRdata));
    checkCudaErrors(cudaFree(d_vecIn));
    checkCudaErrors(cudaFree(d_vecOut));


    // output
    printf("size: %ld x %ld, nonzero: %ld\n", h_mCOO.numRows, h_mCOO.numCols, h_mCOO.numNonzero);
    int errorsCSRpar = 0;
    for(int i = 0; i < h_mCOO.numRows; i++) {
        if (fabs(h_vecOutCSRser[i]-h_vecOutCSRpar[i]) > 1e-3 ) {
            printf("Err(CSRpar): %d %f %f %f\n", i, h_vecOutCOO_cpu[i], h_vecOutCSRser[i], h_vecOutCSRpar[i]);
            errorsCSRpar++;
        }
    }
    printf("Errors: %d(CSRpar)\n", errorsCSRpar);
    printf("Times: %lf(CSRser), %lf(CSRpar)\n", dtimeCSRser, dtimeCSRpar);

    // deallocate
    free(h_vecIn);
    free(h_vecOutCSRser);
    free(h_vecOutCSRpar);
    free(h_vecOutCOO_cpu);

    mtx_COO_free(&h_mCOO);
    mtx_CSR_free(&h_mCSR);

	return 0;
}
