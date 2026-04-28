// nvcc -Xcompiler -fopenmp -o spmv spmv.cu mtxsparse.c
// srun --partition=gpu --reservation=fri --gpus=1 spmv data/pdb1HYS.mtx
// srun --partition=gpu --reservation=fri --gpus=1 spmv data/scircuit.mtx
// pdb1HYS:   36k x  36k, 2200k nonzero, 184 max row els.
// scircuit: 171k x 171k,   59k nonzero, 353 max ro els.


#include <stdio.h>
#include <stdlib.h>
#include "omp.h"
#include <cuda.h>
#include "helper_cuda.h"
#include "mtxsparse.h"


#define THREADS_PER_BLOCK 256
#define REPEAT 10


__global__ void mCSRxVec(int *rowPtr, int *col, float *data, float *vIn, float *vOut, int rows) {		
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    
	if(gid < rows) {
		float sum = 0.0f;
        for (int j = rowPtr[gid]; j < rowPtr[gid + 1]; j++)
            sum += data[j] * vIn[col[j]];
		vOut[gid] = sum;
	}
}														

__global__ void mELLxVec(int *col, float *data, float *vIn, float *vOut, int rows, int numElementsInRow) {		
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

	if(gid < rows) {
		float sum = 0.0f;
		int idx;
		for (int j = 0; j < numElementsInRow; j++) {
			idx = j * rows + gid;
            sum += data[idx] * vIn[col[idx]];
		}
		vOut[gid] = sum;
	}
}


int main(int argc, char *argv[]) {
    FILE *f;
    struct mtxCOO h_mCOO;
    struct mtxCSR h_mCSR;
    struct mtxELL h_mELL;
    int repeat;

    if (argc < 2) {
		fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
		exit(1);
	}
    else{ 
        if ((f = fopen(argv[1], "r")) == NULL) 
            exit(1);
    }

    // create sparse matrices
    if (mtx_COO_create_from_file(&h_mCOO, f) != 0)
        exit(1);
    mtx_CSR_create_from_mtx_COO(&h_mCSR, &h_mCOO);
    mtx_ELL_create_from_mtx_CSR(&h_mELL, &h_mCSR);

    // allocate vectors
    float *h_vecIn = (float *)malloc(h_mCOO.numCols * sizeof(float));
    for (int i = 0; i < h_mCOO.numCols; i++)
        h_vecIn[i] = 1.0;
    float *h_vecOutCOO_cpu = (float *)calloc(h_mCOO.numRows, sizeof(float));
    float *h_vecOutCSR_cpu = (float *)calloc(h_mCSR.numRows, sizeof(float));
    float *h_vecOutELL_cpu = (float *)calloc(h_mELL.numRows, sizeof(float));
    float *h_vecOutCSR_gpu = (float *)calloc(h_mCSR.numRows, sizeof(float));
    float *h_vecOutELL_gpu = (float *)calloc(h_mELL.numRows, sizeof(float));

    // compute with COO
    double dtimeCOO_cpu = omp_get_wtime();
    for (repeat = 0; repeat < REPEAT; repeat++) {
        for (int i = 0; i < h_mCOO.numRows; i++)
            h_vecOutCOO_cpu[i] = 0.0;
        for (int i = 0; i < h_mCOO.numNonzero; i++)
            h_vecOutCOO_cpu[h_mCOO.row[i]] += h_mCOO.data[i] * h_vecIn[h_mCOO.col[i]];
    }
    dtimeCOO_cpu = omp_get_wtime() - dtimeCOO_cpu;

    // compute with CSR
    double dtimeCSR_cpu = omp_get_wtime();
    for (repeat = 0; repeat < REPEAT; repeat++) {
        for (int i = 0; i < h_mCSR.numRows; i++)
            h_vecOutCSR_cpu[i] = 0.0;
        for (int i = 0; i < h_mCSR.numRows; i++)
            for (int j = h_mCSR.rowPtr[i]; j < h_mCSR.rowPtr[i + 1]; j++)
                h_vecOutCSR_cpu[i] += h_mCSR.data[j] * h_vecIn[h_mCSR.col[j]];
    }
    dtimeCSR_cpu = omp_get_wtime() - dtimeCSR_cpu;

    // compute with ELL
    double dtimeELL_cpu = omp_get_wtime();
    for (repeat = 0; repeat < REPEAT; repeat++) {
        for (int i = 0; i < h_mELL.numRows; i++)
            h_vecOutELL_cpu[i] = 0.0;
        for (int i = 0; i < h_mELL.numRows; i++)
            for (int j = 0; j < h_mELL.numElementsInRow; j++)
                h_vecOutELL_cpu[i] += h_mELL.data[j * h_mELL.numRows + i] * h_vecIn[h_mELL.col[j * h_mELL.numRows + i]];
    }
    dtimeELL_cpu = omp_get_wtime() - dtimeELL_cpu;

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
    // ELL
    int *d_mELLcol;
    float *d_mELLdata;
    checkCudaErrors(cudaMalloc((void **)&d_mELLcol, h_mELL.numElements * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_mELLdata, h_mELL.numElements * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_mELLcol, h_mELL.col, h_mELL.numElements * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_mELLdata, h_mELL.data, h_mELL.numElements * sizeof(float), cudaMemcpyHostToDevice));

    // vectors
    float *d_vecIn, *d_vecOut;
    checkCudaErrors(cudaMalloc((void **)&d_vecIn, h_mCOO.numCols * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_vecOut, h_mCOO.numRows * sizeof(float)));

	// Divide work 
    dim3 blockSize(THREADS_PER_BLOCK);
    // CSR
    dim3 gridSizeCSR((h_mCSR.numRows - 1) / blockSize.x + 1);
	// ELL
    dim3 gridSizeELL((h_mELL.numRows - 1) / blockSize.x + 1);
    
	// CSR write, execute, read
    double dtimeCSR_gpu = omp_get_wtime();
    for (repeat = 0; repeat < REPEAT; repeat++) {
        checkCudaErrors(cudaMemcpy(d_vecIn, h_vecIn, h_mCSR.numCols*sizeof(float), cudaMemcpyHostToDevice));
        mCSRxVec<<<gridSizeCSR, blockSize>>>(d_mCSRrowPtr, d_mCSRcol, d_mCSRdata, d_vecIn, d_vecOut, h_mCSR.numRows);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaMemcpy(h_vecOutCSR_gpu, d_vecOut, h_mCSR.numRows*sizeof(float), cudaMemcpyDeviceToHost));
    }
    dtimeCSR_gpu = omp_get_wtime()-dtimeCSR_gpu;
																						
	// ELL write, execute, read
    double dtimeELL_gpu = omp_get_wtime();
    for (repeat = 0; repeat < REPEAT; repeat++) {
        checkCudaErrors(cudaMemcpy(d_vecIn, h_vecIn, h_mCSR.numCols*sizeof(float), cudaMemcpyHostToDevice));
        mELLxVec<<<gridSizeELL, blockSize>>>(d_mELLcol, d_mELLdata, d_vecIn, d_vecOut, h_mELL.numRows, h_mELL.numElementsInRow);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaMemcpy(h_vecOutELL_gpu, d_vecOut, h_mELL.numRows*sizeof(float), cudaMemcpyDeviceToHost));
    }
    dtimeELL_gpu = omp_get_wtime()-dtimeELL_gpu;

    // release device memory
    checkCudaErrors(cudaFree(d_mCSRrowPtr));
    checkCudaErrors(cudaFree(d_mCSRcol));
    checkCudaErrors(cudaFree(d_mCSRdata));
    checkCudaErrors(cudaFree(d_mELLcol));
    checkCudaErrors(cudaFree(d_mELLdata));
    checkCudaErrors(cudaFree(d_vecIn));
    checkCudaErrors(cudaFree(d_vecOut));

    // output
    printf("size: %ld x %ld, nonzero: %ld, max elems in row: %d\n", h_mCOO.numRows, h_mCOO.numCols, h_mCOO.numNonzero, h_mELL.numElementsInRow);
    int errorsCSR_cpu = 0;
    int errorsELL_cpu = 0;
    int errorsCSR_gpu = 0;
    int errorsELL_gpu = 0;
    for(int i = 0; i < h_mCOO.numRows; i++) {
        if (fabs(h_vecOutCOO_cpu[i]-h_vecOutCSR_cpu[i]) > 1e-4 )
            errorsCSR_cpu++;
        if (fabs(h_vecOutCOO_cpu[i]-h_vecOutELL_cpu[i]) > 1e-4 )
            errorsELL_cpu++;
        if (fabs(h_vecOutCOO_cpu[i]-h_vecOutCSR_gpu[i]) > 1e-4 )
            errorsCSR_gpu++;
        if (fabs(h_vecOutCOO_cpu[i]-h_vecOutELL_gpu[i]) > 1e-4 )
            errorsELL_gpu++;
    }
    printf("Errors: %d(CSR_cpu), %d(ELL_cpu), %d(CSR_gpu), %d(ELL_gpu)\n", 
           errorsCSR_cpu, errorsELL_cpu, errorsCSR_gpu, errorsELL_gpu);
    printf("Times: %lf(COO_cpu), %lf(CSR_cpu), %lf(ELL_cpu)\n", dtimeCOO_cpu, dtimeCSR_cpu, dtimeELL_cpu);
    printf("Times: %lf(CSR_gpu), %lf(ELL_gpu)\n\n", dtimeCSR_gpu, dtimeELL_gpu);

    // release host memory
    mtx_COO_free(&h_mCOO);
    mtx_CSR_free(&h_mCSR);
    mtx_ELL_free(&h_mELL);

	return 0;
}
