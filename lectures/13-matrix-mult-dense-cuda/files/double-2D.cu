#include <stdio.h>

#define N 1024
#define BLOCK_SIZE 32

__global__ void process2DArray1(float *array) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        int index = row * N + col;
        array[index] *= array[index];
    }
}

__global__ void process2DArray2(float *array) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        int index = row * N + col;
        array[index] = array[index] * 2.0f;  // simple operation
    }
}

int main() {
    float h_array[N * N];
    float *d_array;

    // Initialize host array
    for (int i = 0; i < N * N; i++)
        h_array[i] = (float)i;

    cudaMalloc((void**)&d_array, N * N * sizeof(float));
    cudaMemcpy(d_array, h_array, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

	float d_dt_kernel;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    process2DArray2<<<blocksPerGrid, threadsPerBlock>>>(d_array);

    cudaEventRecord(start);
    process2DArray1<<<blocksPerGrid, threadsPerBlock>>>(d_array);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d_dt_kernel, start, stop);
    d_dt_kernel /= 1000;
    printf("\ntime: %f\n", d_dt_kernel);
   
    cudaEventRecord(start);
    process2DArray2<<<blocksPerGrid, threadsPerBlock>>>(d_array);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d_dt_kernel, start, stop);
    d_dt_kernel /= 1000;
    printf("\ntime: %f\n", d_dt_kernel);

    cudaMemcpy(h_array, d_array, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print part of the result
    for (int i = 0; i < 5; i++)
        printf("h_array[%d] = %f\n", i, h_array[i]);

    cudaFree(d_array);
    return 0;
}