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

// 2. GPU Naive CUDA
double *evolve_lenia_cuda(const unsigned int rows, const unsigned int cols, const unsigned int steps, const double dt, const unsigned int kernel_size, const struct orbium_coo *orbiums, const unsigned int num_orbiums);

// 3. GPU V1: Shared Memory 
double *evolve_lenia_v1(const unsigned int rows, const unsigned int cols, const unsigned int steps, const double dt, const unsigned int kernel_size, const struct orbium_coo *orbiums, const unsigned int num_orbiums);

// 4. GPU V2: Optimized Double Precision (Constant Mem + If/Else)
double *evolve_lenia_v2(const unsigned int rows, const unsigned int cols, const unsigned int steps, const double dt, const unsigned int kernel_size, const struct orbium_coo *orbiums, const unsigned int num_orbiums);

// 5. GPU V3: Optimized Float Precision (FP32)
// NOTE: Returns float* and takes float dt!
float *evolve_lenia_v3(const unsigned int rows, const unsigned int cols, const unsigned int steps, const float dt, const unsigned int kernel_size, const struct orbium_coo *orbiums, const unsigned int num_orbiums);

#ifdef __cplusplus
}
#endif

#endif