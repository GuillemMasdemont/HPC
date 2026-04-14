#ifndef LENIA_H
#define LENIA_H

#ifdef __cplusplus
extern "C" {
#endif

struct orbium_coo { 
    int row;
    int col;
    int angle;
};

// 1. CPU Sequential
double *evolve_lenia_seq(const unsigned int rows, const unsigned int cols, const unsigned int steps, const double dt, const unsigned int kernel_size, const struct orbium_coo *orbiums, const unsigned int num_orbiums);

// 1b. CPU Sequential V2
double *evolve_lenia_seq_v2(const unsigned int rows, const unsigned int cols, const unsigned int steps, const double dt, const unsigned int kernel_size, const struct orbium_coo *orbiums, const unsigned int num_orbiums);

// 2. GPU Naive CUDA
double *evolve_lenia_cuda(const unsigned int rows, const unsigned int cols, const unsigned int steps, const double dt, const unsigned int kernel_size, const struct orbium_coo *orbiums, const unsigned int num_orbiums);

// 3. GPU V1: Shared Memory 
double *evolve_lenia_v1(const unsigned int rows, const unsigned int cols, const unsigned int steps, const double dt, const unsigned int kernel_size, const struct orbium_coo *orbiums, const unsigned int num_orbiums);

// 4. GPU V2: Optimized Double Precision (Constant Mem + If/Else)
double *evolve_lenia_v2(const unsigned int rows, const unsigned int cols, const unsigned int steps, const double dt, const unsigned int kernel_size, const struct orbium_coo *orbiums, const unsigned int num_orbiums);

// 5. GPU V3: Optimized Float Precision (FP32)
// NOTE: Returns float* and takes float dt!
float *evolve_lenia_v3(const unsigned int rows, const unsigned int cols, const unsigned int steps, const float dt, const unsigned int kernel_size, const struct orbium_coo *orbiums, const unsigned int num_orbiums);

// V5: Block-shape experiment (FP32, constant memory).
// Benchmarks four thread-block geometries at startup and runs the full
// simulation with whichever configuration is fastest on the current GPU.
// Returns a malloc'd float* of size rows*cols; caller must free().
float *evolve_lenia_v5(const unsigned int rows,
                       const unsigned int cols,
                       const unsigned int steps,
                       const float        dt,
                       const unsigned int kernel_size,
                       const struct orbium_coo *orbiums,
                       const unsigned int num_orbiums);

// Returns autotune time (seconds) spent during the most recent evolve_lenia_v5 call.
double lenia_v5_get_last_autotune_seconds(void);

// V6: Dual-GPU simulation.
// Splits the world horizontally: GPU 0 takes the top half, GPU 1 the bottom.
// Halo rows are exchanged after each step via cudaMemcpyPeer (P2P) or via
// a host-staged bounce buffer if P2P is unavailable.
// Returns NULL if fewer than 2 CUDA GPUs are present (caller should fall back).
// On success returns a malloc'd double* of size rows*cols; caller must free().
double *evolve_lenia_v6(const unsigned int rows,
                        const unsigned int cols,
                        const unsigned int steps,
                        const double       dt,
                        const unsigned int kernel_size,
                        const struct orbium_coo *orbiums,
                        const unsigned int num_orbiums);

#ifdef __cplusplus
}
#endif

#endif
