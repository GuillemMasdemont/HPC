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

// -------------------------------------------------------------------------
// 5. GPU V4: Block-shape experiment (FP32 + constant memory)
// Reuses d_kernel_const_v3 and growth_lenia_f / gauss_f from V3 above.
//
// Thread-block configs (all <= 1024 threads):
//   0: 32x32 = 1024  square baseline
//   1: 64x16 = 1024  wide rect  -- better coalescing along columns
//   2: 16x64 = 1024  tall rect  -- more threads per row
//   3: 32x16 =  512  half-warp  -- smaller block; helps when the shared
//                                  memory tile is large enough to prevent
//                                  two 1024-thread blocks fitting on one SM

template<int TILE_X, int TILE_Y>
__global__ void convolve_v5(const float * __restrict__ world,
                              float       * __restrict__ tmp,
                              int rows, int cols, int ksize)
{
    const int halo     = ksize / 2;
    const int shared_w = TILE_X + 2 * halo;
    const int shared_h = TILE_Y + 2 * halo;

    extern __shared__ float smem_f[];

    const int tx       = threadIdx.x;
    const int ty       = threadIdx.y;
    const int out_col0 = blockIdx.x * TILE_X;
    const int out_row0 = blockIdx.y * TILE_Y;

    for (int dy = ty; dy < shared_h; dy += TILE_Y) {
        for (int dx = tx; dx < shared_w; dx += TILE_X) {
            int gr = out_row0 - halo + dy;
            if (gr < 0)          gr += rows;
            else if (gr >= rows) gr -= rows;
            int gc = out_col0 - halo + dx;
            if (gc < 0)          gc += cols;
            else if (gc >= cols) gc -= cols;
            smem_f[dy * shared_w + dx] = world[gr * cols + gc];
        }
    }
    __syncthreads();

    const int out_col = out_col0 + tx;
    const int out_row = out_row0 + ty;
    if (out_row < rows && out_col < cols) {
        float sum = 0.f;
        for (int ki = ksize - 1, si = 0; ki >= 0; ki--, si++)
            for (int kj = ksize - 1, sj = 0; kj >= 0; kj--, sj++)
                sum += d_kernel_const_v3[ki * ksize + kj] *
                       smem_f[(ty + si) * shared_w + (tx + sj)];
        tmp[out_row * cols + out_col] = sum;
    }
}

template<int TILE_X, int TILE_Y>
__global__ void evolve_v5(float * __restrict__ world,
                           const float * __restrict__ tmp,
                           int rows, int cols, float dt)
{
    const int c = blockIdx.x * TILE_X + threadIdx.x;
    const int r = blockIdx.y * TILE_Y + threadIdx.y;
    if (r < rows && c < cols) {
        int idx = r * cols + c;
        float v = world[idx] + dt * growth_lenia_f(tmp[idx]);
        world[idx] = fminf(1.f, fmaxf(0.f, v));
    }
}

#define BENCH_STEPS 10

static double g_v5_last_autotune_seconds = 0.0;

double lenia_v5_get_last_autotune_seconds(void)
{
    return g_v5_last_autotune_seconds;
}

static float bench_v5(int cfg,
                       float *d_world, float *d_tmp,
                       int rows, int cols, int ksize, float dt,
                       cudaEvent_t ev0, cudaEvent_t ev1)
{
    int bx, by;
    switch (cfg) {
        case 0: bx = 32; by = 32; break;   // 1024 threads
        case 1: bx = 64; by = 16; break;   // 1024 threads
        case 2: bx = 16; by = 64; break;   // 1024 threads
        default: bx = 32; by = 16; break;  //  512 threads
    }
    int halo  = ksize / 2;
    size_t smem = (size_t)(bx + 2*halo) * (by + 2*halo) * sizeof(float);
    dim3 block(bx, by);
    dim3 grid((cols + bx - 1) / bx, (rows + by - 1) / by);

    CUDA_CHECK(cudaEventRecord(ev0));
    for (int s = 0; s < BENCH_STEPS; s++) {
        switch (cfg) {
            case 0:
                convolve_v5<32,32><<<grid,block,smem>>>(d_world,d_tmp,rows,cols,ksize);
                evolve_v5  <32,32><<<grid,block   >>>(d_world,d_tmp,rows,cols,dt);
                break;
            case 1:
                convolve_v5<64,16><<<grid,block,smem>>>(d_world,d_tmp,rows,cols,ksize);
                evolve_v5  <64,16><<<grid,block   >>>(d_world,d_tmp,rows,cols,dt);
                break;
            case 2:
                convolve_v5<16,64><<<grid,block,smem>>>(d_world,d_tmp,rows,cols,ksize);
                evolve_v5  <16,64><<<grid,block   >>>(d_world,d_tmp,rows,cols,dt);
                break;
            default:
                convolve_v5<32,16><<<grid,block,smem>>>(d_world,d_tmp,rows,cols,ksize);
                evolve_v5  <32,16><<<grid,block   >>>(d_world,d_tmp,rows,cols,dt);
                break;
        }
    }
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    return ms;
}

float *evolve_lenia_v5(const unsigned int rows, const unsigned int cols,
                        const unsigned int steps, const float dt,
                        const unsigned int kernel_size,
                        const struct orbium_coo *orbiums,
                        const unsigned int num_orbiums)
{
    if (kernel_size > MAX_KERNEL_SIZE) exit(EXIT_FAILURE);

#ifdef GENERATE_GIF
    ensure_gif_output_dir();
    ge_GIF *gif = ge_new_gif("gifs/lenia_v5.gif", cols, rows, inferno_pallete, 8, -1, 0);
#endif

    size_t world_bytes = rows * cols * sizeof(float);
    size_t kern_bytes  = kernel_size * kernel_size * sizeof(float);

    double *h_world_d = (double *)calloc(rows * cols, sizeof(double));
    float  *h_kernel  = (float  *)calloc(kernel_size * kernel_size, sizeof(float));
    float  *h_world_f = (float  *)calloc(rows * cols, sizeof(float));

    generate_kernel_float(h_kernel, kernel_size);   // defined in V3 section above

    for (unsigned int o = 0; o < num_orbiums; o++)
        h_world_d = place_orbium(h_world_d, rows, cols,
                                  orbiums[o].row, orbiums[o].col, orbiums[o].angle);
    for (unsigned int i = 0; i < rows * cols; i++)
        h_world_f[i] = (float)h_world_d[i];
    free(h_world_d);

    float *d_world, *d_tmp;
    CUDA_CHECK(cudaMalloc(&d_world, world_bytes));
    CUDA_CHECK(cudaMalloc(&d_tmp,   world_bytes));
    CUDA_CHECK(cudaMemcpy(d_world, h_world_f, world_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel_const_v3, h_kernel, kern_bytes));
    free(h_kernel);

    // ---- Benchmark ----
    g_v5_last_autotune_seconds = 0.0;
    double autotune_start = omp_get_wtime();

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    static const char *names[] = {"32x32 (1024)", "64x16 (1024)", "16x64 (1024)", "32x16 (512)"};
    int   best_cfg  = 0;
    float best_time = FLT_MAX;

    printf("[v5] Benchmarking block configs (%d steps, %ux%u grid):\n",
           BENCH_STEPS, rows, cols);
    for (int cfg = 0; cfg < 4; cfg++) {
        CUDA_CHECK(cudaMemcpy(d_world, h_world_f, world_bytes, cudaMemcpyHostToDevice));
        float ms = bench_v5(cfg, d_world, d_tmp,
                            (int)rows, (int)cols, (int)kernel_size, dt, ev0, ev1);
        printf("  config %d (%s): %.3f ms/step\n", cfg, names[cfg], ms / BENCH_STEPS);
        if (ms < best_time) { best_time = ms; best_cfg = cfg; }
    }
    printf("  -> chosen: config %d (%s)\n\n", best_cfg, names[best_cfg]);

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));

    g_v5_last_autotune_seconds = omp_get_wtime() - autotune_start;

    // ---- Full simulation with best config ----
    CUDA_CHECK(cudaMemcpy(d_world, h_world_f, world_bytes, cudaMemcpyHostToDevice));

    int bx, by;
    switch (best_cfg) {
        case 0: bx = 32; by = 32; break;
        case 1: bx = 64; by = 16; break;
        case 2: bx = 16; by = 64; break;
        default: bx = 32; by = 16; break;
    }
    int halo = (int)kernel_size / 2;
    size_t smem = (size_t)(bx + 2*halo) * (by + 2*halo) * sizeof(float);
    dim3 block(bx, by);
    dim3 grid((cols + bx - 1) / bx, (rows + by - 1) / by);

    for (unsigned int step = 0; step < steps; step++) {
        switch (best_cfg) {
            case 0:
                convolve_v5<32,32><<<grid,block,smem>>>(d_world,d_tmp,(int)rows,(int)cols,(int)kernel_size);
                evolve_v5  <32,32><<<grid,block   >>>(d_world,d_tmp,(int)rows,(int)cols,dt);
                break;
            case 1:
                convolve_v5<64,16><<<grid,block,smem>>>(d_world,d_tmp,(int)rows,(int)cols,(int)kernel_size);
                evolve_v5  <64,16><<<grid,block   >>>(d_world,d_tmp,(int)rows,(int)cols,dt);
                break;
            case 2:
                convolve_v5<16,64><<<grid,block,smem>>>(d_world,d_tmp,(int)rows,(int)cols,(int)kernel_size);
                evolve_v5  <16,64><<<grid,block   >>>(d_world,d_tmp,(int)rows,(int)cols,dt);
                break;
            default:
                convolve_v5<32,16><<<grid,block,smem>>>(d_world,d_tmp,(int)rows,(int)cols,(int)kernel_size);
                evolve_v5  <32,16><<<grid,block   >>>(d_world,d_tmp,(int)rows,(int)cols,dt);
                break;
        }

#ifdef GENERATE_GIF
        CUDA_CHECK(cudaMemcpy(h_world_f, d_world, world_bytes, cudaMemcpyDeviceToHost));
        for (unsigned int i = 0; i < rows * cols; i++)
            gif->frame[i] = h_world_f[i] * 255;
        ge_add_frame(gif, 5);
#endif
    }

#ifdef GENERATE_GIF
    ge_close_gif(gif);
#endif

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_world_f, d_world, world_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_world);
    cudaFree(d_tmp);
    return h_world_f;
}

// -------------------------------------------------------------------------
// 6. GPU V5: Dual-GPU simulation
//    Splits the world horizontally: GPU 0 owns rows [0 .. half-1],
//    GPU 1 owns rows [half .. rows-1].  Each device allocates a padded slab
//    (halo rows on both sides) and the two exchange boundary rows via
//    cudaMemcpyPeer after every step (falls back to host-staged copy if
//    peer access is unavailable).
// -------------------------------------------------------------------------

static void try_enable_peer(int from, int to)
{
    int can = 0;
    cudaDeviceCanAccessPeer(&can, from, to);
    if (can) {
        CUDA_CHECK(cudaSetDevice(from));
        cudaError_t e = cudaDeviceEnablePeerAccess(to, 0);
        if (e != cudaErrorPeerAccessAlreadyEnabled && e != cudaSuccess)
            CUDA_CHECK(e);
    }
}

// Convolution over a vertically-padded slab.
// The slab has `full_rows = halo + local_rows + halo` rows in memory.
// Output is written only to the interior rows [halo .. halo+local_rows-1].
__global__ void convolve_v6(const float * __restrict__ slab,
                              float       * __restrict__ tmp,
                              int local_rows, int cols,
                              int ksize, int halo)
{
    const int lc = blockIdx.x * blockDim.x + threadIdx.x;
    const int lr = blockIdx.y * blockDim.y + threadIdx.y;
    if (lr >= local_rows || lc >= cols) return;

    const int sr = lr + halo;   // row in the padded slab
    float sum = 0.f;
    for (int ki = ksize - 1; ki >= 0; ki--) {
        for (int kj = ksize - 1; kj >= 0; kj--) {
            int si = sr - halo + (ksize - 1 - ki);   // always in [0, full_rows)
            int sc = lc - halo + (ksize - 1 - kj);
            if (sc < 0)          sc += cols;
            else if (sc >= cols) sc -= cols;
            sum += d_kernel_const_v3[ki * ksize + kj] * slab[si * cols + sc];
        }
    }
    tmp[sr * cols + lc] = sum;
}

__global__ void evolve_v6(float * __restrict__ slab,
                           const float * __restrict__ tmp,
                           int local_rows, int cols,
                           int halo, float dt)
{
    const int lc = blockIdx.x * blockDim.x + threadIdx.x;
    const int lr = blockIdx.y * blockDim.y + threadIdx.y;
    if (lr >= local_rows || lc >= cols) return;
    int idx = (lr + halo) * cols + lc;
    float v = slab[idx] + dt * growth_lenia_f(tmp[idx]);
    slab[idx] = fminf(1.f, fmaxf(0.f, v));
}

static void exchange_halos_v6(float *d_slab0, float *d_slab1,
                               int half0, int half1,
                               int cols, int halo,
                               bool p2p, float *h_bounce)
{
    size_t halo_bytes = (size_t)halo * cols * sizeof(float);

    // Convenience pointers into each slab
    float *g0_data  = d_slab0 + (size_t)halo * cols;          // first data row, GPU0
    float *g0_dtail = g0_data + (size_t)(half0 - halo) * cols;// last halo of data, GPU0
    float *g0_top   = d_slab0;                                 // top ghost of GPU0
    float *g0_bot   = g0_data + (size_t)half0 * cols;          // bottom ghost of GPU0

    float *g1_data  = d_slab1 + (size_t)halo * cols;
    float *g1_dtail = g1_data + (size_t)(half1 - halo) * cols;
    float *g1_top   = d_slab1;
    float *g1_bot   = g1_data + (size_t)half1 * cols;

    if (p2p) {
        // Neighbour exchange
        CUDA_CHECK(cudaMemcpyPeer(g1_top, 1, g0_dtail, 0, halo_bytes)); // GPU0 tail -> GPU1 top ghost
        CUDA_CHECK(cudaMemcpyPeer(g0_bot, 0, g1_data,  1, halo_bytes)); // GPU1 head -> GPU0 bot ghost
        // Torus wrap
        CUDA_CHECK(cudaMemcpyPeer(g0_top, 0, g1_dtail, 1, halo_bytes)); // GPU1 tail -> GPU0 top ghost
        CUDA_CHECK(cudaMemcpyPeer(g1_bot, 1, g0_data,  0, halo_bytes)); // GPU0 head -> GPU1 bot ghost
    } else {
        // Neighbour: GPU0 tail -> GPU1 top ghost
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMemcpy(h_bounce, g0_dtail, halo_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaMemcpy(g1_top, h_bounce, halo_bytes, cudaMemcpyHostToDevice));

        // Neighbour: GPU1 head -> GPU0 bot ghost
        CUDA_CHECK(cudaMemcpy(h_bounce, g1_data, halo_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMemcpy(g0_bot, h_bounce, halo_bytes, cudaMemcpyHostToDevice));

        // Wrap: GPU1 tail -> GPU0 top ghost
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaMemcpy(h_bounce, g1_dtail, halo_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMemcpy(g0_top, h_bounce, halo_bytes, cudaMemcpyHostToDevice));

        // Wrap: GPU0 head -> GPU1 bot ghost
        CUDA_CHECK(cudaMemcpy(h_bounce, g0_data, halo_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaMemcpy(g1_bot, h_bounce, halo_bytes, cudaMemcpyHostToDevice));
    }
}

double *evolve_lenia_v6(const unsigned int rows, const unsigned int cols,
                         const unsigned int steps, const double dt,
                         const unsigned int kernel_size,
                         const struct orbium_coo *orbiums,
                         const unsigned int num_orbiums)
{
    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    if (num_gpus < 2) {
        fprintf(stderr, "[v6] Need >= 2 CUDA GPUs, found %d.\n", num_gpus);
        return NULL;
    }
    if (kernel_size > MAX_KERNEL_SIZE) exit(EXIT_FAILURE);

#ifdef GENERATE_GIF
    ensure_gif_output_dir();
    ge_GIF *gif = ge_new_gif("gifs/lenia_v6.gif", cols, rows, inferno_pallete, 8, -1, 0);
#endif

    int half0 = (int)(rows + 1) / 2;
    int half1 = (int)rows - half0;
    int halo  = (int)kernel_size / 2;
    float fdt = (float)dt;

    printf("[v6] Grid %ux%u, kernel %u, halo %d\n", rows, cols, kernel_size, halo);
    printf("[v6] GPU0: %d rows, GPU1: %d rows\n", half0, half1);

    // ---- Host world (double -> float) ----
    double *h_world_d = (double *)calloc(rows * cols, sizeof(double));
    float  *h_world_f = (float  *)calloc(rows * cols, sizeof(float));
    float  *h_kernel  = (float  *)calloc(kernel_size * kernel_size, sizeof(float));

    generate_kernel_float(h_kernel, kernel_size);
    for (unsigned int o = 0; o < num_orbiums; o++)
        h_world_d = place_orbium(h_world_d, rows, cols,
                                  orbiums[o].row, orbiums[o].col, orbiums[o].angle);
    for (unsigned int i = 0; i < rows * cols; i++)
        h_world_f[i] = (float)h_world_d[i];
    free(h_world_d);

    // ---- Peer access ----
    try_enable_peer(0, 1);
    try_enable_peer(1, 0);
    int can01 = 0, can10 = 0;
    cudaDeviceCanAccessPeer(&can01, 0, 1);
    cudaDeviceCanAccessPeer(&can10, 1, 0);
    bool p2p = (can01 && can10);
    printf("[v6] P2P: %s\n", p2p ? "enabled" : "via host");

    // ---- Padded slabs ----
    size_t slab0_bytes = (size_t)(halo + half0 + halo) * cols * sizeof(float);
    size_t slab1_bytes = (size_t)(halo + half1 + halo) * cols * sizeof(float);
    size_t kern_bytes  = (size_t)kernel_size * kernel_size * sizeof(float);

    float *d_slab0, *d_tmp0;
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc(&d_slab0, slab0_bytes));
    CUDA_CHECK(cudaMalloc(&d_tmp0,  slab0_bytes));
    CUDA_CHECK(cudaMemset(d_slab0, 0, slab0_bytes));
    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel_const_v3, h_kernel, kern_bytes));
    CUDA_CHECK(cudaMemcpy(d_slab0 + (size_t)halo * cols,
                           h_world_f,
                           (size_t)half0 * cols * sizeof(float),
                           cudaMemcpyHostToDevice));

    float *d_slab1, *d_tmp1;
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaMalloc(&d_slab1, slab1_bytes));
    CUDA_CHECK(cudaMalloc(&d_tmp1,  slab1_bytes));
    CUDA_CHECK(cudaMemset(d_slab1, 0, slab1_bytes));
    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel_const_v3, h_kernel, kern_bytes));
    CUDA_CHECK(cudaMemcpy(d_slab1 + (size_t)halo * cols,
                           h_world_f + (size_t)half0 * cols,
                           (size_t)half1 * cols * sizeof(float),
                           cudaMemcpyHostToDevice));

    free(h_kernel);

    float *h_bounce = NULL;
    if (!p2p) {
        h_bounce = (float *)malloc((size_t)halo * cols * sizeof(float));
        if (!h_bounce) { fprintf(stderr, "OOM\n"); exit(EXIT_FAILURE); }
    }

    dim3 block(32, 16);
    dim3 grid0((cols + 31) / 32, (half0 + 15) / 16);
    dim3 grid1((cols + 31) / 32, (half1 + 15) / 16);

    // Initial halo fill before step 0
    exchange_halos_v6(d_slab0, d_slab1, half0, half1, (int)cols, halo,
                      p2p, h_bounce);

    for (unsigned int step = 0; step < steps; step++) {
        CUDA_CHECK(cudaSetDevice(0));
        convolve_v6<<<grid0, block>>>(d_slab0, d_tmp0, half0, (int)cols, (int)kernel_size, halo);

        CUDA_CHECK(cudaSetDevice(1));
        convolve_v6<<<grid1, block>>>(d_slab1, d_tmp1, half1, (int)cols, (int)kernel_size, halo);

        CUDA_CHECK(cudaSetDevice(0));
        evolve_v6<<<grid0, block>>>(d_slab0, d_tmp0, half0, (int)cols, halo, fdt);

        CUDA_CHECK(cudaSetDevice(1));
        evolve_v6<<<grid1, block>>>(d_slab1, d_tmp1, half1, (int)cols, halo, fdt);

        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaDeviceSynchronize());

        exchange_halos_v6(d_slab0, d_slab1, half0, half1, (int)cols, halo,
                          p2p, h_bounce);

#ifdef GENERATE_GIF
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMemcpy(h_world_f,
                               d_slab0 + (size_t)halo * cols,
                               (size_t)half0 * cols * sizeof(float),
                               cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaMemcpy(h_world_f + (size_t)half0 * cols,
                               d_slab1 + (size_t)halo * cols,
                               (size_t)half1 * cols * sizeof(float),
                               cudaMemcpyDeviceToHost));
        for (unsigned int i = 0; i < rows * cols; i++)
            gif->frame[i] = h_world_f[i] * 255;
        ge_add_frame(gif, 5);
#endif
    }

#ifdef GENERATE_GIF
    ge_close_gif(gif);
#endif

    // ---- Gather ----
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpy(h_world_f,
                           d_slab0 + (size_t)halo * cols,
                           (size_t)half0 * cols * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaMemcpy(h_world_f + (size_t)half0 * cols,
                           d_slab1 + (size_t)halo * cols,
                           (size_t)half1 * cols * sizeof(float),
                           cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaSetDevice(0)); cudaFree(d_slab0); cudaFree(d_tmp0);
    CUDA_CHECK(cudaSetDevice(1)); cudaFree(d_slab1); cudaFree(d_tmp1);
    free(h_bounce);

    // Return as double* to match the other evolve_lenia_* functions
    double *h_result = (double *)malloc(rows * cols * sizeof(double));
    for (unsigned int i = 0; i < rows * cols; i++)
        h_result[i] = (double)h_world_f[i];
    free(h_world_f);
    return h_result;
}

// -------------------------------------------------------------------------
//  CPU IMPRovements
// -------------------------------------------------------------------------

inline double gauss_cpu_v2(double x, double mu, double sigma)
{
    double z = (x - mu) / sigma;
    return exp(-0.5 * z * z);
}

double growth_lenia_cpu_v2(double u)
{
    double mu    = 0.15;
    double sigma = 0.015;
    return -1 + 2 * gauss_cpu_v2(u, mu, sigma);
}

void set_openmp_threads(int num_threads)
{
    if (num_threads > 0)
    {
        omp_set_dynamic(0);
        omp_set_num_threads(num_threads);
    }
}

double *generate_kernel_cpu_v2(double *K, const unsigned int size)
{
    if (K == NULL)
    {
        fprintf(stderr, "generate_kernel: K is NULL\n");
        return NULL;
    }

    double mu    = 0.5;
    double sigma = 0.15;
    int    r     = size / 2;
    double sum   = 0.0;

    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int y = -r; y < r; y++)
    {
        for (int x = -r; x < r; x++)
        {
            double distance = sqrt((1 + x) * (1 + x) + (1 + y) * (1 + y)) / r;

            // CHANGE 3 (previous): skip exp() entirely for cells outside the circle.
            K[(y + r) * size + x + r] = (distance <= 1.0) ? gauss_cpu_v2(distance, mu, sigma) : 0.0;
            sum += K[(y + r) * size + x + r];
        }
    }

    #pragma omp parallel for schedule(static)
    for (unsigned int y = 0; y < size; y++)
        for (unsigned int x = 0; x < size; x++)
            K[y * size + x] /= sum;

    return K;
}

inline double *convolve2d_cpu_v2(
    double *result,
    const double *input,
    const double *w,
    const unsigned int rows,
    const unsigned int cols,
    const unsigned int w_rows,
    const unsigned int w_cols)
{
    unsigned int *row_lut = (unsigned int *)malloc(rows * w_rows * sizeof(unsigned int));
    unsigned int *col_lut = (unsigned int *)malloc(cols * w_cols * sizeof(unsigned int));

    if (!row_lut || !col_lut)
    {
        fprintf(stderr, "convolve2d: out of memory for lookup tables\n");
        free(row_lut);
        free(col_lut);
        return NULL;
    }

    const int half_r = (int)w_rows / 2;
    const int half_c = (int)w_cols / 2;

    #pragma omp parallel for schedule(static)
    for (unsigned int i = 0; i < rows; i++)
        for (unsigned int ki = 0; ki < w_rows; ki++)
            row_lut[i * w_rows + ki] = ((int)i - half_r + (int)rows + (int)ki) % rows;
    
    #pragma omp parallel for schedule(static)
    for (unsigned int j = 0; j < cols; j++)
        for (unsigned int kj = 0; kj < w_cols; kj++)
            col_lut[j * w_cols + kj] = ((int)j - half_c + (int)cols + (int)kj) % cols;

    #pragma omp parallel for schedule(static)
    for (unsigned int i = 0; i < rows; i++)
    {
        for (unsigned int j = 0; j < cols; j++)
        {
            double sum = 0.0;

            for (unsigned int ki = 0; ki < w_rows; ki++)
            {
                const double *w_row   = &w[(w_rows - 1 - ki) * w_cols];
                const double *src_row = &input[row_lut[i * w_rows + ki] * cols];

                for (unsigned int kj = 0; kj < w_cols; kj++)
                    sum += w_row[w_cols - 1 - kj] * src_row[col_lut[j * w_cols + kj]];
            }

            result[i * cols + j] = sum;
        }
    }

    free(row_lut);
    free(col_lut);
    return result;
}

double *evolve_lenia_seq_v2(
    const unsigned int rows,
    const unsigned int cols,
    const unsigned int steps,
    const double dt,
    const unsigned int kernel_size,
    const struct orbium_coo *orbiums,
    const unsigned int num_orbiums)
{
    printf("evolve_lenia: using %d OpenMP thread(s)\n", omp_get_max_threads());

#ifdef GENERATE_GIF
    ge_GIF *gif = ge_new_gif(
        "gifs/lenia_openmp_v2.gif",
        cols, rows,
        inferno_pallete,
        8,
        -1,
        0);
#endif

    double *w     = (double *)calloc(kernel_size * kernel_size, sizeof(double));
    double *world = (double *)calloc(rows * cols, sizeof(double));
    double *tmp   = (double *)calloc(rows * cols, sizeof(double));

    if (!w || !world || !tmp)
    {
        fprintf(stderr, "evolve_lenia: memory allocation failed\n");
        free(w); free(world); free(tmp);
        return NULL;
    }

    w = generate_kernel_cpu_v2(w, kernel_size);

    for (unsigned int o = 0; o < num_orbiums; o++)
        world = place_orbium(world, rows, cols,
                             orbiums[o].row, orbiums[o].col, orbiums[o].angle);

    for (unsigned int step = 0; step < steps; step++)
    {
        tmp = convolve2d_cpu_v2(tmp, world, w, rows, cols, kernel_size, kernel_size);

        #pragma omp parallel for schedule(static)
        for (unsigned int i = 0; i < rows; i++)
        {
            for (unsigned int j = 0; j < cols; j++)
            {
                const unsigned int idx = i * cols + j;

                double val = world[idx] + dt * growth_lenia_cpu_v2(tmp[idx]);

                if      (val < 0.0) val = 0.0;
                else if (val > 1.0) val = 1.0;

                world[idx] = val;

#ifdef GENERATE_GIF
                // Each thread writes to a unique gif->frame[idx] — no race condition.
                gif->frame[idx] = (unsigned char)(val * 255);
#endif
            }
        }

        // [OPENMP] ge_add_frame writes sequentially to the GIF file on disk.
        // It MUST stay outside the parallel region — calling it from multiple
        // threads simultaneously would corrupt the output file.
#ifdef GENERATE_GIF
        ge_add_frame(gif, 5);
#endif
    }

#ifdef GENERATE_GIF
    ge_close_gif(gif);
#endif

    free(w);
    free(tmp);
    return world;
}