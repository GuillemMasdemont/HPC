// compile as
// module load CUDA
// nvcc -Xcompiler -fopenmp -o dmv mtxsparse.c dmv.cu 
// srun --partition=gpu --reservation=fri --gpus=1 --mem=5G ./DenseMV ./data/dw8192.mtx r
// srun --partition=gpu --reservation=fri --gpus=1 --mem=5G ./DenseMV ./data/dw8192.mtx c

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "omp.h"
#include <cuda.h>
#include "helper_cuda.h"
#include "mtxsparse.h"

#define THREADS_PER_BLOCK 256

#define REPEAT 1

__global__ void matxvec(float *mData, float *vIn, float *vOut, int rows, int cols, int rowMajor) {
    
    if (rowMajor) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if(i < rows) {
            float sum = 0.0;
            for (int j = 0; j < cols; j++)
                sum += mData[i * cols + j] * vIn[j];
            vOut[i] = sum;
        }
    }
    else {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if(i < rows) {
            float sum = 0.0;
            for (int j = 0; j < cols; j++)
                sum += mData[i + j * rows] * vIn[j];
            vOut[i] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    
    FILE *f;
    struct mtxCOO mCOO;

    if (argc != 3) {
		fprintf(stderr, "Usage: %s [martix-market-filename] [row/column major]\n", argv[0]);
		exit(1);
	}
    else { 
        if ((f = fopen(argv[1], "r")) == NULL) 
            exit(1);
    }
    int rowMajor = argv[2][0] != 'c';

    // create sparse matrix
    if (mtx_COO_create_from_file(&mCOO, f) != 0)
        exit(1);

    // create full matrix from COO
    printf("%d %d %d\n", mCOO.numCols, mCOO.numRows, mCOO.numNonzero);

	float* h_mData = (float *)calloc(mCOO.numRows * mCOO.numCols, sizeof(float));
    float** h_m;

    if (rowMajor) {
        // row-major
        h_m = (float **)malloc(sizeof(float *) * mCOO.numRows);

        for (int i = 0; i < mCOO.numRows; i++)
            h_m[i] = &h_mData[i * mCOO.numCols];

        for (int k = 0; k < mCOO.numNonzero; k++)
            h_m[mCOO.row[k]][mCOO.col[k]] = mCOO.data[k];
    }
    else {
        // col-major
        h_m = (float **)malloc(sizeof(float *) * mCOO.numCols);
        for (int i = 0; i < mCOO.numCols; i++)
            h_m[i] = &h_mData[i * mCOO.numRows];
        for (int k = 0; k < mCOO.numNonzero; k++)
            h_m[mCOO.col[k]][mCOO.row[k]] = mCOO.data[k];
    }

    // allocate vectors
    float *h_vecIn = (float *)malloc(mCOO.numCols * sizeof(float));
    for (int i = 0; i < mCOO.numCols; i++)
        h_vecIn[i] = 1.0;   
    float *h_vecOut_cpu = (float *)calloc(mCOO.numRows, sizeof(float));
    float *h_vecOut_gpu = (float *)calloc(mCOO.numRows, sizeof(float));

    // compute
    double h_dt = omp_get_wtime();
    if (rowMajor)
        for(int repeat = 0; repeat < REPEAT; repeat++) {
            for (int i = 0; i < mCOO.numRows; i++) {
                h_vecOut_cpu[i] = 0;
                for (int j = 0; j < mCOO.numCols; j++)
                    h_vecOut_cpu[i] += h_m[i][j] * h_vecIn[j];
            }
        }
    else
        for(int repeat = 0; repeat < REPEAT; repeat++) {
            for (int i = 0; i < mCOO.numRows; i++) {
                h_vecOut_cpu[i] = 0;
                for (int j = 0; j < mCOO.numCols; j++)
                    h_vecOut_cpu[i] += h_m[j][i] * h_vecIn[j];
            }
        }
    h_dt = omp_get_wtime() - h_dt;

    float *d_mData, *d_vecIn, *d_vecOut;
    checkCudaErrors(cudaMalloc((void **)&d_mData, mCOO.numRows * mCOO.numCols * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_vecIn, mCOO.numCols * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_vecOut, mCOO.numRows * sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_mData, h_mData, mCOO.numRows * mCOO.numCols * sizeof(float), cudaMemcpyHostToDevice));

	// Divide work (square matrix)
    dim3 gridSize((mCOO.numRows - 1) / THREADS_PER_BLOCK + 1);
	dim3 blockSize(THREADS_PER_BLOCK);
    // COO write, execute, read
    double d_dt = omp_get_wtime();
    for(int repeat = 0; repeat < REPEAT; repeat++) {
        checkCudaErrors(cudaMemcpy(d_vecIn, h_vecIn, mCOO.numCols * sizeof(float), cudaMemcpyHostToDevice));
        matxvec<<<gridSize, blockSize>>>(d_mData, d_vecIn, d_vecOut, mCOO.numRows, mCOO.numCols, rowMajor);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaMemcpy(h_vecOut_gpu, d_vecOut, mCOO.numRows * sizeof(float), cudaMemcpyDeviceToHost));
    }
    d_dt = omp_get_wtime()-d_dt;

    // output
    printf("size: %ld x %ld, nonzero: %ld\n", mCOO.numRows, mCOO.numCols, mCOO.numNonzero);
    int d_errors = 0;
    for(int i = 0; i < mCOO.numRows; i++) {
        if (fabs(h_vecOut_cpu[i]-h_vecOut_gpu[i]) > 1e-3 ) {
            printf("Err (gpu): %d %f %f\n", i, h_vecOut_cpu[i], h_vecOut_gpu[i]);
            d_errors++;
        }
    }
    printf("Errors: %d(gpu)\n", d_errors);
    printf("Times: %lf(cpu), %lf(gpu)\n", h_dt, d_dt);

    // release device memory
    checkCudaErrors(cudaFree(d_mData));
    checkCudaErrors(cudaFree(d_vecIn));
    checkCudaErrors(cudaFree(d_vecOut));

    // release host memory
    free(h_vecIn);
    free(h_vecOut_cpu);
    free(h_vecOut_gpu);
    free(h_m);
    free(h_mData);
    mtx_COO_free(&mCOO);

	return 0;
}
