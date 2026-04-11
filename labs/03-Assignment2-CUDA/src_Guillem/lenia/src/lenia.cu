#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "lenia.h"
#include "orbium.h"
#include "gifenc.h" // Uncommented for GIF generation

// UNCOMMENT THE LINE BELOW TO GENERATE GIFS (WILL SLOW DOWN GPU BENCHMARK)
// #define GENERATE_GIF

#define TILE 32
#define MAX_KERNEL_SIZE 31 // Increased to accommodate the 26x26 kernel from main.c

// Error-checking macro for the optimized versions
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// -------------------------------------------------------------------------
// 1. CPU SEQUENTIAL CASE
// -------------------------------------------------------------------------

#define w_mac(r, c) (w[(r) * kernel_size + (c)])
#define input_mac(r, c) (input[((r) % rows) * cols + ((c) % cols)])

inline double gauss(double x, double mu, double sigma) {
    return exp(-0.5 * pow((x - mu) / sigma, 2));
}

double growth_lenia(double u) {
    double mu = 0.15;
    double sigma = 0.015;
    return -1.0 + 2.0 * gauss(u, mu, sigma);
}

// USED BOTH IN CPU AND GPU CASES!
double *generate_kernel(double *K, const unsigned int size) {
    double mu = 0.5;
    double sigma = 0.15;
    int r = size / 2;
    double sum = 0;
    
    if (K != NULL) {
        for (int y = -r; y < r; y++) {
            for (int x = -r; x < r; x++) {
                double distance = sqrt((1.0 + x) * (1.0 + x) + (1.0 + y) * (1.0 + y)) / r;
                K[(y + r) * size + x + r] = gauss(distance, mu, sigma);
                if (distance > 1.0) {
                    K[(y + r) * size + x + r] = 0; 
                }
                sum += K[(y + r) * size + x + r];
            }
        }
        for (unsigned int y = 0; y < size; y++) {
            for (unsigned int x = 0; x < size; x++) {
                K[y * size + x] /= sum;
            }
        }
    }
    return K;
}

inline double *convolve2d_seq(double *result, const double *input, const double *w, 
                              const unsigned int rows, const unsigned int cols, 
                              const unsigned int kernel_size) {
    for (unsigned int i = 0; i < rows; i++) {
        for (unsigned int j = 0; j < cols; j++) {
            double sum = 0;
            for (int ki = kernel_size - 1, kri = 0; ki >= 0; ki--, kri++) {
                for (int kj = kernel_size - 1, kcj = 0; kj >= 0; kj--, kcj++) {
                    sum += w_mac(ki, kj) * input_mac((i - kernel_size / 2 + rows + kri), 
                                                     (j - kernel_size / 2 + cols + kcj));
                }
            }
            result[i * cols + j] = sum;
        }
    }
    return result;
}

double *evolve_lenia_seq(const unsigned int rows, const unsigned int cols, const unsigned int steps, 
                         const double dt, const unsigned int kernel_size, 
                         const struct orbium_coo *orbiums, const unsigned int num_orbiums) {
                         
#ifdef GENERATE_GIF
    ge_GIF *gif = ge_new_gif("lenia_seq.gif", cols, rows, inferno_pallete, 8, -1, 0);
#endif

    double *w = (double *)calloc(kernel_size * kernel_size, sizeof(double));
    double *world = (double *)calloc(rows * cols, sizeof(double));
    double *tmp = (double *)calloc(rows * cols, sizeof(double));

    w = generate_kernel(w, kernel_size);

    for (unsigned int o = 0; o < num_orbiums; o++) {
        world = place_orbium(world, rows, cols, orbiums[o].row, orbiums[o].col, orbiums[o].angle);
    }

    for (unsigned int step = 0; step < steps; step++) {
        tmp = convolve2d_seq(tmp, world, w, rows, cols, kernel_size);
        
        for (unsigned int i = 0; i < rows; i++) {
            for (unsigned int j = 0; j < cols; j++) {
                world[i * cols + j] += dt * growth_lenia(tmp[i * cols + j]);
                world[i * cols + j] = fmin(1.0, fmax(0.0, world[i * cols + j]));
                
#ifdef GENERATE_GIF
                gif->frame[i * cols + j] = world[i * cols + j] * 255;
#endif
            }
        }
#ifdef GENERATE_GIF
        ge_add_frame(gif, 5);
#endif
    }
    
#ifdef GENERATE_GIF
    ge_close_gif(gif);
#endif

    free(w); free(tmp);
    return world;
}

// -------------------------------------------------------------------------
// 2. GPU NAIVE CUDA
// -------------------------------------------------------------------------

__device__ inline double gauss_dev(double x, double mu, double sigma) {
    return exp(-0.5 * pow((x - mu) / sigma, 2.0));
}

__device__ inline double growth_lenia_dev(double u) {
    double mu = 0.15;
    double sigma = 0.015;
    return -1.0 + 2.0 * gauss_dev(u, mu, sigma);
}

__global__ void convolve2d_kernel(double *result, const double *input, const double *w, int rows, int cols, int kernel_size) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    int j = blockIdx.x * blockDim.x + threadIdx.x; 

    if (i >= rows || j >= cols) return;

    double sum = 0.0;
    for (int ki = kernel_size - 1, kri = 0; ki >= 0; ki--, kri++) {
        for (int kj = kernel_size - 1, kcj = 0; kj >= 0; kj--, kcj++) {
            int ri = ((i - kernel_size / 2 + rows + kri) % rows);
            int ci = ((j - kernel_size / 2 + cols + kcj) % cols);
            sum += w[ki * kernel_size + kj] * input[ri * cols + ci];
        }
    }
    result[i * cols + j] = sum;
}

__global__ void update_world_kernel(double *world, const double *tmp, int rows, int cols, double dt) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    int j = blockIdx.x * blockDim.x + threadIdx.x; 

    if (i >= rows || j >= cols) return;

    int idx = i * cols + j;
    world[idx] += dt * growth_lenia_dev(tmp[idx]);
    world[idx] = fmin(1.0, fmax(0.0, world[idx]));
}

double *evolve_lenia_cuda(const unsigned int rows, const unsigned int cols, const unsigned int steps, const double dt, const unsigned int kernel_size, const struct orbium_coo *orbiums, const unsigned int num_orbiums) {

#ifdef GENERATE_GIF
    ge_GIF *gif = ge_new_gif("lenia_cuda.gif", cols, rows, inferno_pallete, 8, -1, 0);
#endif

    size_t grid_size = rows * cols * sizeof(double);
    size_t kernel_bytes = kernel_size * kernel_size * sizeof(double);

    double *h_w = (double *)calloc(kernel_size * kernel_size, sizeof(double));
    double *h_world = (double *)calloc(rows * cols, sizeof(double));
    
    h_w = generate_kernel(h_w, kernel_size);
    for (unsigned int o = 0; o < num_orbiums; o++) {
        h_world = place_orbium(h_world, rows, cols, orbiums[o].row, orbiums[o].col, orbiums[o].angle);
    }

    double *d_world, *d_tmp, *d_w;
    cudaMalloc(&d_world, grid_size);
    cudaMalloc(&d_tmp, grid_size);
    cudaMalloc(&d_w, kernel_bytes);

    cudaMemcpy(d_world, h_world, grid_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, kernel_bytes, cudaMemcpyHostToDevice);

    dim3 threads_per_block(16, 16);
    dim3 num_blocks((cols + 15) / 16, (rows + 15) / 16);

    for (unsigned int step = 0; step < steps; step++) {
        convolve2d_kernel<<<num_blocks, threads_per_block>>>(d_tmp, d_world, d_w, rows, cols, kernel_size);
        cudaDeviceSynchronize();  
        
        update_world_kernel<<<num_blocks, threads_per_block>>>(d_world, d_tmp, rows, cols, dt);
        cudaDeviceSynchronize();  

#ifdef GENERATE_GIF
        cudaMemcpy(h_world, d_world, grid_size, cudaMemcpyDeviceToHost);
        for (unsigned int i = 0; i < rows; i++) {
            for (unsigned int j = 0; j < cols; j++) {
                gif->frame[i * cols + j] = h_world[i * cols + j] * 255;
            }
        }
        ge_add_frame(gif, 5);
#endif
    }

#ifndef GENERATE_GIF
    cudaMemcpy(h_world, d_world, grid_size, cudaMemcpyDeviceToHost);
#endif

#ifdef GENERATE_GIF
    ge_close_gif(gif);
#endif

    cudaFree(d_world); cudaFree(d_tmp); cudaFree(d_w); free(h_w);
    return h_world;
}

// -------------------------------------------------------------------------
// 3. GPU V1: SHARED MEMORY 
// -------------------------------------------------------------------------

__global__ void convolve2d_kernel_v1(const double * __restrict__ world,
                                  double       * __restrict__ tmp,
                                  const double * __restrict__ K,
                                  int rows, int cols, int ksize) 
{
    const int halo    = ksize / 2;
    const int shared_w = TILE + 2 * halo;
    const int shared_h = TILE + 2 * halo;

    extern __shared__ double smem[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int out_col0 = blockIdx.x * TILE;
    const int out_row0 = blockIdx.y * TILE;

    for (int dy = ty; dy < shared_h; dy += TILE) {
        for (int dx = tx; dx < shared_w; dx += TILE) {
            int gr = (out_row0 - halo + dy + rows) % rows;
            int gc = (out_col0 - halo + dx + cols) % cols;
            smem[dy * shared_w + dx] = world[gr * cols + gc];
        }
    }
    __syncthreads();

    const int out_col = out_col0 + tx;
    const int out_row = out_row0 + ty;

    if (out_row < rows && out_col < cols) {
        double sum = 0.0;
        for (int ki = ksize - 1, si = 0; ki >= 0; ki--, si++) {
            for (int kj = ksize - 1, sj = 0; kj >= 0; kj--, sj++) {
                sum += K[ki * ksize + kj] * smem[(ty + si) * shared_w + (tx + sj)];
            }
        }
        tmp[out_row * cols + out_col] = sum;
    }
}

__global__ void evolve_kernel_v1(double * __restrict__ world, const double * __restrict__ tmp, int rows, int cols, double dt) {
    const int c = blockIdx.x * TILE + threadIdx.x; //TILE improves slightly the results over blockDim.y
    const int r = blockIdx.y * TILE + threadIdx.y;

    if (r < rows && c < cols) {
        int idx  = r * cols + c;
        double v = world[idx] + dt * growth_lenia_dev(tmp[idx]);
        world[idx] = fmin(1.0, fmax(0.0, v));
    }
}

double *evolve_lenia_v1(const unsigned int rows, const unsigned int cols,
                     const unsigned int steps, const double dt,
                     const unsigned int kernel_size,
                     const struct orbium_coo *orbiums,
                     const unsigned int num_orbiums)
{
    size_t world_bytes  = rows * cols * sizeof(double);
    size_t kernel_bytes = kernel_size * kernel_size * sizeof(double);

    double *h_world  = (double *)calloc(rows * cols, sizeof(double));
    double *h_kernel = (double *)calloc(kernel_size * kernel_size, sizeof(double));

    generate_kernel(h_kernel, kernel_size);

    for (unsigned int o = 0; o < num_orbiums; o++)
        h_world = place_orbium(h_world, rows, cols, orbiums[o].row, orbiums[o].col, orbiums[o].angle);

    double *d_world, *d_tmp, *d_kernel;
    CUDA_CHECK(cudaMalloc(&d_world,  world_bytes));
    CUDA_CHECK(cudaMalloc(&d_tmp,    world_bytes));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_bytes));

    CUDA_CHECK(cudaMemcpy(d_world,  h_world,  world_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, kernel_bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
    int halo = kernel_size / 2;
    size_t smem_bytes = (TILE + 2*halo) * (TILE + 2*halo) * sizeof(double);

    for (unsigned int step = 0; step < steps; step++) {
        convolve2d_kernel_v1<<<grid, block, smem_bytes>>>(d_world, d_tmp, d_kernel, (int)rows, (int)cols, (int)kernel_size);
        evolve_kernel_v1<<<grid, block>>>(d_world, d_tmp, (int)rows, (int)cols, dt);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_world, d_world, world_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_world); cudaFree(d_tmp); cudaFree(d_kernel); free(h_kernel);
    return h_world;
}

// -------------------------------------------------------------------------
// 4. GPU V2: OPTIMIZED DOUBLE (Constant Mem + If/Else)
// -------------------------------------------------------------------------

__constant__ double d_kernel_const_v2[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

__global__ void convolve2d_kernel_v2(const double * __restrict__ world,
                                  double       * __restrict__ tmp,
                                  int rows, int cols, int ksize) 
{
    const int halo    = ksize / 2;
    const int shared_w = TILE + 2 * halo;
    const int shared_h = TILE + 2 * halo;

    extern __shared__ double smem[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int out_col0 = blockIdx.x * TILE;
    const int out_row0 = blockIdx.y * TILE;

    for (int dy = ty; dy < shared_h; dy += TILE) {
        for (int dx = tx; dx < shared_w; dx += TILE) {
            
            int gr = out_row0 - halo + dy;
            if (gr < 0) gr += rows;
            else if (gr >= rows) gr -= rows;

            int gc = out_col0 - halo + dx;
            if (gc < 0) gc += cols;
            else if (gc >= cols) gc -= cols;

            smem[dy * shared_w + dx] = world[gr * cols + gc];
        }
    }
    __syncthreads();

    const int out_col = out_col0 + tx;
    const int out_row = out_row0 + ty;

    if (out_row < rows && out_col < cols) {
        double sum = 0.0;
        for (int ki = ksize - 1, si = 0; ki >= 0; ki--, si++) {
            for (int kj = ksize - 1, sj = 0; kj >= 0; kj--, sj++) {
                sum += d_kernel_const_v2[ki * ksize + kj] *
                       smem[(ty + si) * shared_w + (tx + sj)];
            }
        }
        tmp[out_row * cols + out_col] = sum;
    }
}

double *evolve_lenia_v2(const unsigned int rows, const unsigned int cols,
                     const unsigned int steps, const double dt,
                     const unsigned int kernel_size,
                     const struct orbium_coo *orbiums,
                     const unsigned int num_orbiums)
{
    if (kernel_size > MAX_KERNEL_SIZE) exit(EXIT_FAILURE);

    size_t world_bytes  = rows * cols * sizeof(double);
    size_t kernel_bytes = kernel_size * kernel_size * sizeof(double);

    double *h_world  = (double *)calloc(rows * cols, sizeof(double));
    double *h_kernel = (double *)calloc(kernel_size * kernel_size, sizeof(double));

    generate_kernel(h_kernel, kernel_size);

    for (unsigned int o = 0; o < num_orbiums; o++)
        h_world = place_orbium(h_world, rows, cols, orbiums[o].row, orbiums[o].col, orbiums[o].angle);

    double *d_world, *d_tmp;
    CUDA_CHECK(cudaMalloc(&d_world,  world_bytes));
    CUDA_CHECK(cudaMalloc(&d_tmp,    world_bytes));
    CUDA_CHECK(cudaMemcpy(d_world,  h_world,  world_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel_const_v2, h_kernel, kernel_bytes));

    dim3 block(TILE, TILE);
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
    int halo = kernel_size / 2;
    size_t smem_bytes = (TILE + 2*halo) * (TILE + 2*halo) * sizeof(double);

    for (unsigned int step = 0; step < steps; step++) {
        convolve2d_kernel_v2<<<grid, block, smem_bytes>>>(d_world, d_tmp, (int)rows, (int)cols, (int)kernel_size);
        evolve_kernel_v1<<<grid, block>>>(d_world, d_tmp, (int)rows, (int)cols, dt);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_world, d_world, world_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_world); cudaFree(d_tmp); free(h_kernel);
    return h_world;
}

// -------------------------------------------------------------------------
// 5. GPU V3: OPTIMIZED FLOAT (FP32 precision + Constant Mem + If/Else)
// -------------------------------------------------------------------------

static float *generate_kernel_float(float *K, unsigned int size) {
    float mu = 0.5f, sigma = 0.15f;
    int r = size / 2;
    float sum = 0.0f;

    for (int y = -r; y < r; y++) {
        for (int x = -r; x < r; x++) {
            float d = sqrtf((1.0f+x)*(1.0f+x) + (1.0f+y)*(1.0f+y)) / (float)r;
            float v = (d <= 1.0f) ? expf(-0.5f * powf((d - mu) / sigma, 2)) : 0.0f;
            K[(y+r)*size + (x+r)] = v;
            sum += v;
        }
    }
    for (unsigned int i = 0; i < size * size; i++) K[i] /= sum;
    return K;
}

__constant__ float d_kernel_const_v3[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

__device__ inline float gauss_f(float x, float mu, float sigma) {
    float z = (x - mu) / sigma;
    return expf(-0.5f * z * z); 
}

__device__ inline float growth_lenia_f(float u) {
    return -1.0f + 2.0f * gauss_f(u, 0.15f, 0.015f);
}

__global__ void convolve2d_kernel_v3(const float * __restrict__ world,
                                  float       * __restrict__ tmp,
                                  int rows, int cols, int ksize) 
{
    const int halo    = ksize / 2;
    const int shared_w = TILE + 2 * halo;
    const int shared_h = TILE + 2 * halo;

    extern __shared__ float smem_f[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int out_col0 = blockIdx.x * TILE;
    const int out_row0 = blockIdx.y * TILE;

    for (int dy = ty; dy < shared_h; dy += TILE) {
        for (int dx = tx; dx < shared_w; dx += TILE) {
            int gr = out_row0 - halo + dy;
            if (gr < 0) gr += rows;
            else if (gr >= rows) gr -= rows;

            int gc = out_col0 - halo + dx;
            if (gc < 0) gc += cols;
            else if (gc >= cols) gc -= cols;

            smem_f[dy * shared_w + dx] = world[gr * cols + gc];
        }
    }
    __syncthreads();

    const int out_col = out_col0 + tx;
    const int out_row = out_row0 + ty;

    if (out_row < rows && out_col < cols) {
        float sum = 0.0f;
        for (int ki = ksize - 1, si = 0; ki >= 0; ki--, si++) {
            for (int kj = ksize - 1, sj = 0; kj >= 0; kj--, sj++) {
                sum += d_kernel_const_v3[ki * ksize + kj] *
                       smem_f[(ty + si) * shared_w + (tx + sj)];
            }
        }
        tmp[out_row * cols + out_col] = sum;
    }
}

__global__ void evolve_kernel_v3(float * __restrict__ world, const float * __restrict__ tmp, int rows, int cols, float dt) {
    const int c = blockIdx.x * TILE + threadIdx.x;
    const int r = blockIdx.y * TILE + threadIdx.y;

    if (r < rows && c < cols) {
        int idx  = r * cols + c;
        float v = world[idx] + dt * growth_lenia_f(tmp[idx]);
        world[idx] = fminf(1.0f, fmaxf(0.0f, v)); 
    }
}

float *evolve_lenia_v3(const unsigned int rows, const unsigned int cols,
                     const unsigned int steps, const float dt,
                     const unsigned int kernel_size,
                     const struct orbium_coo *orbiums,
                     const unsigned int num_orbiums)
{
    if (kernel_size > MAX_KERNEL_SIZE) exit(EXIT_FAILURE);

    size_t world_bytes_f  = rows * cols * sizeof(float);
    size_t kernel_bytes_f = kernel_size * kernel_size * sizeof(float);

    double *h_world_d = (double *)calloc(rows * cols, sizeof(double));
    float *h_kernel_f = (float *)calloc(kernel_size * kernel_size, sizeof(float));
    float *h_world_f  = (float *)calloc(rows * cols, sizeof(float));

    generate_kernel_float(h_kernel_f, kernel_size);

    // Because place_orbium likely expects a double array, we populate double first
    for (unsigned int o = 0; o < num_orbiums; o++) {
        h_world_d = place_orbium(h_world_d, rows, cols, orbiums[o].row, orbiums[o].col, orbiums[o].angle);
    }

    // Cast double to float for the device upload
    for (unsigned int i = 0; i < rows * cols; i++) {
        h_world_f[i] = (float)h_world_d[i];
    }
    free(h_world_d);

    float *d_world, *d_tmp;
    CUDA_CHECK(cudaMalloc(&d_world,  world_bytes_f));
    CUDA_CHECK(cudaMalloc(&d_tmp,    world_bytes_f));
    CUDA_CHECK(cudaMemcpy(d_world,  h_world_f,  world_bytes_f,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel_const_v3, h_kernel_f, kernel_bytes_f));

    dim3 block(TILE, TILE);
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
    int halo = kernel_size / 2;
    size_t smem_bytes = (TILE + 2*halo) * (TILE + 2*halo) * sizeof(float);

    for (unsigned int step = 0; step < steps; step++) {
        convolve2d_kernel_v3<<<grid, block, smem_bytes>>>(d_world, d_tmp, (int)rows, (int)cols, (int)kernel_size);
        evolve_kernel_v3<<<grid, block>>>(d_world, d_tmp, (int)rows, (int)cols, dt);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_world_f, d_world, world_bytes_f, cudaMemcpyDeviceToHost));

    cudaFree(d_world); cudaFree(d_tmp); free(h_kernel_f);
    return h_world_f;
}