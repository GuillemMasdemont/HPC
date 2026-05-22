// nccl_demo.cu  –  Single-node, 2-GPU NCCL vs naive AllReduce (plain C style)
//
// Compile:
//   nvcc -o nccl_demo nccl_demo.cu -lnccl -O2

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <nccl.h>

#define NUM_GPUS   2
#define N          (64 * 1024 * 1024)   /* 64M floats = 256 MB */

#define CUDA_CHECK(cmd) \
    do { \
        cudaError_t e = (cmd); \
        if (e != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d  %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define NCCL_CHECK(cmd) \
    do { \
        ncclResult_t r = (cmd); \
        if (r != ncclSuccess) { \
            fprintf(stderr, "NCCL error %s:%d  %s\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

// fill arrays with data
static void fill(float *d[NUM_GPUS], float *h[NUM_GPUS]) {
    int i, j;
    for (i = 0; i < NUM_GPUS; i++) {
        for (j = 0; j < N; j++)
            h[i][j] = (float)(i + 1);   /* GPU0 = 1.0, GPU1 = 2.0 */
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMemcpy(d[i], h[i], N * sizeof(float), cudaMemcpyHostToDevice));
    }
}

// verify results
static void verify(float *d, int gpu, float expected, const char *label) {
    float val;
    CUDA_CHECK(cudaSetDevice(gpu));
    CUDA_CHECK(cudaMemcpy(&val, d, sizeof(float), cudaMemcpyDeviceToHost));
    printf("  result[0] on GPU%d: %s\n", gpu, (val == expected) ? "OK" : "FAIL");
}

// summing on host
static void naive_allreduce(float *d[NUM_GPUS], float *h[NUM_GPUS]) {
    int i;

    // device -> host
    for (i = 0; i < NUM_GPUS; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMemcpy(h[i], d[i], N * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // sum on CPU into h[0]
    for (i = 0; i < N; i++)
        h[0][i] += h[1][i];

    // host -> device
    for (i = 0; i < NUM_GPUS; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMemcpy(d[i], h[0], N * sizeof(float), cudaMemcpyHostToDevice));
    }
}

// NCCL: direct GPU-GPU AllReduce
static void nccl_allreduce(float *send[NUM_GPUS], float *recv[NUM_GPUS],
                           ncclComm_t comms[NUM_GPUS], cudaStream_t streams[NUM_GPUS]) {
    int i;
    NCCL_CHECK(ncclGroupStart());
    for (i = 0; i < NUM_GPUS; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        NCCL_CHECK(ncclAllReduce(send[i], recv[i], N, ncclFloat, ncclSum, comms[i], streams[i]));
    }
    NCCL_CHECK(ncclGroupEnd());

    for (i = 0; i < NUM_GPUS; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }
}

// main
int main(void) {
    int i;
    float *d_send[NUM_GPUS], *d_recv[NUM_GPUS];
    float *h[NUM_GPUS];
    ncclComm_t    comms[NUM_GPUS];
    cudaStream_t  streams[NUM_GPUS];
    int dev_ids[NUM_GPUS] = {0, 1};
    double tNaive, tNCCL;

    // allocate host buffers and device buffers, create streams
    for (i = 0; i < NUM_GPUS; i++) {
        h[i] = (float *)malloc(N * sizeof(float));
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMalloc(&d_send[i], N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_recv[i], N * sizeof(float)));
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // init NCCL
    NCCL_CHECK(ncclCommInitAll(comms, NUM_GPUS, dev_ids));

    // allreduce via host
    fill(d_send, h);
    tNaive = now_ms();
    naive_allreduce(d_send, h);
    tNaive = now_ms() - tNaive;
    printf("AllReduce via host:\n");
    printf("  Time: %.2f ms\n", tNaive);
    verify(d_send[0], 0, 3.0f, "via host");
    verify(d_send[0], 1, 3.0f, "via host");
    printf("\n");

    // allreduce via NCCL
    fill(d_send, h);
    tNCCL = now_ms();
    nccl_allreduce(d_send, d_recv, comms, streams);
    tNCCL = now_ms() - tNCCL;
    printf("AllReduce via NCCL:\n");
    printf("  Time: %.2f ms\n", tNCCL);
    verify(d_recv[0], 0, 3.0f, "via NCCL");
    verify(d_recv[0], 1, 3.0f, "via NCCL");
    printf("\n");

    // speedup
    printf("NCCL/host: %.2fx\n", tNaive / tNCCL);
    
    // cleanup
    for (i = 0; i < NUM_GPUS; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaFree(d_send[i]));
        CUDA_CHECK(cudaFree(d_recv[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        NCCL_CHECK(ncclCommDestroy(comms[i]));
        free(h[i]);
    }

    return EXIT_SUCCESS;
}
