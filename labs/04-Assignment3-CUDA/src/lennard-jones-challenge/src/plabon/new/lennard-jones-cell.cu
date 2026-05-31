// =============================================================================
//  Lennard-Jones — CELL-ONLY (linked-cell) GPU variant  [v8]
// -----------------------------------------------------------------------------
//  Self-contained translation unit. NO neighbour list, NO skin, NO rebuild
//  check, NO tuning. Cells are sized at R_CUT so a particle's cutoff sphere is
//  fully contained in its 3x3x3 cell stencil; cell lists are rebuilt every step
//  (O(N)). Forces are evaluated by traversing the 27 surrounding cells,
//  warp-per-particle, with a shuffle reduction and no atomics.
//
//  Shares nothing mutable with lennard-jones.cu: own device buffers, own
//  kernels (prefixed cok_/co_). Only the host-side CPU helpers compute_ke() and
//  compute_v_shift() are reused (declared extern via the header).
// =============================================================================

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include "lennard-jones.h"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d %s\n",                           \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CO_WARP             32u
#define CO_FORCE_BLOCK      256u   // 8 warps -> 8 particles per block
#define CO_SPATIAL_SORT_EVERY 4

// ─── device state (file-local) ───────────────────────────────────────────────
static double *c_x = nullptr, *c_y = nullptr, *c_z = nullptr;
static double *c_vx = nullptr, *c_vy = nullptr, *c_vz = nullptr;
static double *c_fx = nullptr, *c_fy = nullptr, *c_fz = nullptr;
static double *c_pe_arr = nullptr, *c_pe_total = nullptr;
static double *c_xs = nullptr, *c_ys = nullptr, *c_zs = nullptr;
static double *c_vxs = nullptr, *c_vys = nullptr, *c_vzs = nullptr;
static int *c_cell_id = nullptr, *c_cell_start = nullptr;
static int *c_cell_count = nullptr, *c_cell_part = nullptr;

static unsigned int c_n = 0;
static unsigned int c_n_cells_side = 0;
static unsigned int c_n_cells = 0;
static unsigned int c_step_counter = 0;
static double C_V_SHIFT = 0.0;


double random_double(void) { return (double)rand() / (double)RAND_MAX; }

double compute_ke(const Particle *particles, unsigned int n) {
    double ke = 0.0;
    for (unsigned int i = 0; i < n; ++i) {
        const Particle *p = &particles[i];
        ke += 0.5 * (p->vx * p->vx + p->vy * p->vy + p->vz * p->vz);
    }
    return ke;
}

double compute_v_shift(void) {
    return 4.0 * EPSILON * (pow(SIGMA / R_CUT, 12.0) - pow(SIGMA / R_CUT, 6.0));
}

int initialize_particles(Particle *particles, unsigned int n, double box_size,
                         double placement_fraction, unsigned int seed,
                         double temperature) {
    srand(seed);
    unsigned int n_side = (unsigned int)ceil(cbrt((double)n));
    double placement_size = placement_fraction * box_size;
    double offset = 0.5 * (box_size - placement_size);
    double delta = placement_size / (double)n_side;
    double mean_vx = 0.0, mean_vy = 0.0, mean_vz = 0.0;
    for (unsigned int k = 0; k < n; k++) {
        particles[k].id = k;
        unsigned int ix = k % n_side;
        unsigned int iy = (k / n_side) % n_side;
        unsigned int iz = k / (n_side * n_side);
        double x0 = offset + (0.5 + (double)ix) * delta;
        double y0 = offset + (0.5 + (double)iy) * delta;
        double z0 = offset + (0.5 + (double)iz) * delta;
        particles[k].x = x0 + (2.0 * random_double() - 1.0) * JITTER * delta;
        particles[k].y = y0 + (2.0 * random_double() - 1.0) * JITTER * delta;
        particles[k].z = z0 + (2.0 * random_double() - 1.0) * JITTER * delta;
        particles[k].vx = 2.0 * random_double() - 1.0;
        particles[k].vy = 2.0 * random_double() - 1.0;
        particles[k].vz = 2.0 * random_double() - 1.0;
        mean_vx += particles[k].vx;
        mean_vy += particles[k].vy;
        mean_vz += particles[k].vz;
    }
    mean_vx /= (double)n; mean_vy /= (double)n; mean_vz /= (double)n;
    double ke = 0.0;
    for (unsigned int k = 0; k < n; k++) {
        particles[k].vx -= mean_vx;
        particles[k].vy -= mean_vy;
        particles[k].vz -= mean_vz;
        ke += 0.5 * (particles[k].vx * particles[k].vx +
                     particles[k].vy * particles[k].vy +
                     particles[k].vz * particles[k].vz);
    }
    double cur_temp = ke / (double)n;
    if (cur_temp <= 0.0) return 0;
    double scale = sqrt(temperature / cur_temp);
    for (unsigned int k = 0; k < n; k++) {
        particles[k].vx *= scale;
        particles[k].vy *= scale;
        particles[k].vz *= scale;
    }
    return 1;
}


// =============================================================================
// KERNELS
// =============================================================================

__global__ void cok_cell_assign(
    const double * __restrict__ x, const double * __restrict__ y, const double * __restrict__ z,
    int * __restrict__ cell_id, int * __restrict__ cell_count,
    unsigned int n, double box_size, unsigned int n_cells_side)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double cell_size = box_size / (double)n_cells_side;
    int cx = (int)(x[i] / cell_size); if (cx >= (int)n_cells_side) cx = n_cells_side - 1; if (cx < 0) cx = 0;
    int cy = (int)(y[i] / cell_size); if (cy >= (int)n_cells_side) cy = n_cells_side - 1; if (cy < 0) cy = 0;
    int cz = (int)(z[i] / cell_size); if (cz >= (int)n_cells_side) cz = n_cells_side - 1; if (cz < 0) cz = 0;
    int cid = cx + cy * n_cells_side + cz * n_cells_side * n_cells_side;
    cell_id[i] = cid;
    atomicAdd(&cell_count[cid], 1);
}

__global__ void cok_cell_fill(
    const int * __restrict__ cell_id, const int * __restrict__ cell_start,
    int * __restrict__ cell_part, int * __restrict__ cell_cursor, unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int cid = cell_id[i];
    int slot = atomicAdd(&cell_cursor[cid], 1);
    cell_part[cell_start[cid] + slot] = (int)i;
}

__global__ void cok_gather6(
    const int *idx,
    const double *xi, const double *yi, const double *zi,
    const double *vxi, const double *vyi, const double *vzi,
    double *xo, double *yo, double *zo,
    double *vxo, double *vyo, double *vzo, unsigned int n)
{
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    int j = idx[k];
    xo[k] = xi[j];  yo[k] = yi[j];  zo[k] = zi[j];
    vxo[k] = vxi[j]; vyo[k] = vyi[j]; vzo[k] = vzi[j];
}

__global__ void cok_kick_drift(
    double * __restrict__ x, double * __restrict__ y, double * __restrict__ z,
    double * __restrict__ vx, double * __restrict__ vy, double * __restrict__ vz,
    const double * __restrict__ fx, const double * __restrict__ fy, const double * __restrict__ fz,
    unsigned int n, double box_size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    vx[i] += 0.5 * DT * fx[i];
    vy[i] += 0.5 * DT * fy[i];
    vz[i] += 0.5 * DT * fz[i];
    x[i] += DT * vx[i];
    y[i] += DT * vy[i];
    z[i] += DT * vz[i];
    double wx = fmod(x[i], box_size); if (wx < 0.0) wx += box_size; x[i] = wx;
    double wy = fmod(y[i], box_size); if (wy < 0.0) wy += box_size; y[i] = wy;
    double wz = fmod(z[i], box_size); if (wz < 0.0) wz += box_size; z[i] = wz;
}

__global__ void cok_kick(
    double * __restrict__ vx, double * __restrict__ vy, double * __restrict__ vz,
    const double * __restrict__ fx, const double * __restrict__ fy, const double * __restrict__ fz,
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    vx[i] += 0.5 * DT * fx[i];
    vy[i] += 0.5 * DT * fy[i];
    vz[i] += 0.5 * DT * fz[i];
}

__global__ void cok_reduce_pe(
    const double * __restrict__ pe_arr, double * __restrict__ pe_total, unsigned int n)
{
    extern __shared__ double s_pe[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    s_pe[tid] = (i < n) ? pe_arr[i] : 0.0;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_pe[tid] += s_pe[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(pe_total, s_pe[0]);
}

// Force kernel: warp per particle, traverse 27 surrounding cells, no atomics.
__global__ void cok_forces(
    const double * __restrict__ x, const double * __restrict__ y, const double * __restrict__ z,
    double * __restrict__ fx, double * __restrict__ fy, double * __restrict__ fz,
    double * __restrict__ pe_out,
    const int * __restrict__ cell_id,
    const int * __restrict__ cell_start, const int * __restrict__ cell_count,
    const int * __restrict__ cell_part,
    unsigned int n, double box_size, double v_shift, unsigned int n_cells_side)
{
    const unsigned int i    = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; // warp == particle
    const unsigned int lane = threadIdx.x & 31u;
    if (i >= n) return;

    const double xi = x[i], yi = y[i], zi = z[i];
    const double r_cut_sq = R_CUT * R_CUT;
    const double sig2 = SIGMA * SIGMA;
    const double inv_box = 1.0 / box_size;

    const int cid = cell_id[i];
    const int n2 = n_cells_side * n_cells_side;
    const int cz_i = cid / n2;
    const int cy_i = (cid % n2) / n_cells_side;
    const int cx_i = cid % n_cells_side;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0, Pe = 0.0;

    for (int dz = -1; dz <= 1; dz++)
    for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++) {
        int nx = cx_i + dx; if (nx < 0) nx += n_cells_side; if (nx >= (int)n_cells_side) nx -= n_cells_side;
        int ny = cy_i + dy; if (ny < 0) ny += n_cells_side; if (ny >= (int)n_cells_side) ny -= n_cells_side;
        int nz = cz_i + dz; if (nz < 0) nz += n_cells_side; if (nz >= (int)n_cells_side) nz -= n_cells_side;
        int ncid = nx + ny * n_cells_side + nz * n2;
        int start = cell_start[ncid];
        int cnt = cell_count[ncid];
        for (int s = (int)lane; s < cnt; s += (int)CO_WARP) { // lanes split this cell's members
            int j = cell_part[start + s];
            if (j == (int)i) continue;
            double ddx = xi - x[j];
            double ddy = yi - y[j];
            double ddz = zi - z[j];
            ddx -= box_size * rint(ddx * inv_box);
            ddy -= box_size * rint(ddy * inv_box);
            ddz -= box_size * rint(ddz * inv_box);
            double r2 = ddx * ddx + ddy * ddy + ddz * ddz;
            if (r2 >= r_cut_sq || r2 == 0.0) continue;
            double inv_r2 = 1.0 / r2;
            double sr2 = sig2 * inv_r2;
            double sr6 = sr2 * sr2 * sr2;
            double sr12 = sr6 * sr6;
            double fij = 24.0 * EPSILON * (2.0 * sr12 - sr6) * inv_r2;
            Fx += fij * ddx;
            Fy += fij * ddy;
            Fz += fij * ddz;
            Pe += 0.5 * (4.0 * EPSILON * (sr12 - sr6) - v_shift); // pair seen from i and j
        }
    }

    for (int off = 16; off > 0; off >>= 1) {
        Fx += __shfl_down_sync(0xffffffffu, Fx, off);
        Fy += __shfl_down_sync(0xffffffffu, Fy, off);
        Fz += __shfl_down_sync(0xffffffffu, Fz, off);
        Pe += __shfl_down_sync(0xffffffffu, Pe, off);
    }
    if (lane == 0u) {
        fx[i] = Fx;  fy[i] = Fy;  fz[i] = Fz;  pe_out[i] = Pe;
    }
}

// =============================================================================
// HOST HELPERS
// =============================================================================

static void co_alloc(unsigned int n, double box_size) {
    if (c_n == n) return;

    c_n_cells_side = (unsigned int)(box_size / R_CUT); // cell edge = box/side >= R_CUT
    if (c_n_cells_side < 3) {
        fprintf(stderr,
                "[CELL-ONLY] box too small: %u cells/side (need >=3 for a safe "
                "27-cell stencil with PBC). Use the Verlet method for this N.\n",
                c_n_cells_side);
        exit(1);
    }
    c_n_cells = c_n_cells_side * c_n_cells_side * c_n_cells_side;
    fprintf(stdout,
            "[CELL-ONLY] %u cells/side  %u cells  cell_edge=%.4f (>= R_CUT=%.2f)  "
            "no skin, no neighbour list\n",
            c_n_cells_side, c_n_cells, box_size / (double)c_n_cells_side, (double)R_CUT);

    const size_t sz = (size_t)n * sizeof(double);
    CUDA_CHECK(cudaMalloc(&c_x, sz));  CUDA_CHECK(cudaMalloc(&c_y, sz));  CUDA_CHECK(cudaMalloc(&c_z, sz));
    CUDA_CHECK(cudaMalloc(&c_vx, sz)); CUDA_CHECK(cudaMalloc(&c_vy, sz)); CUDA_CHECK(cudaMalloc(&c_vz, sz));
    CUDA_CHECK(cudaMalloc(&c_fx, sz)); CUDA_CHECK(cudaMalloc(&c_fy, sz)); CUDA_CHECK(cudaMalloc(&c_fz, sz));
    CUDA_CHECK(cudaMalloc(&c_pe_arr, sz));
    CUDA_CHECK(cudaMalloc(&c_pe_total, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c_xs, sz));  CUDA_CHECK(cudaMalloc(&c_ys, sz));  CUDA_CHECK(cudaMalloc(&c_zs, sz));
    CUDA_CHECK(cudaMalloc(&c_vxs, sz)); CUDA_CHECK(cudaMalloc(&c_vys, sz)); CUDA_CHECK(cudaMalloc(&c_vzs, sz));
    CUDA_CHECK(cudaMalloc(&c_cell_id, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&c_cell_count, (size_t)c_n_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&c_cell_start, (size_t)c_n_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&c_cell_part, (size_t)n * sizeof(int)));

    c_n = n;
}

static void co_upload(const Particle *p, unsigned int n) {
    const size_t sz = n * sizeof(double);
    double *hx = (double*)malloc(sz), *hy = (double*)malloc(sz), *hz = (double*)malloc(sz);
    double *hvx = (double*)malloc(sz), *hvy = (double*)malloc(sz), *hvz = (double*)malloc(sz);
    for (unsigned int i = 0; i < n; ++i) {
        hx[i] = p[i].x;  hy[i] = p[i].y;  hz[i] = p[i].z;
        hvx[i] = p[i].vx; hvy[i] = p[i].vy; hvz[i] = p[i].vz;
    }
    CUDA_CHECK(cudaMemcpy(c_x, hx, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c_y, hy, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c_z, hz, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c_vx, hvx, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c_vy, hvy, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c_vz, hvz, sz, cudaMemcpyHostToDevice));
    free(hx); free(hy); free(hz); free(hvx); free(hvy); free(hvz);
}

static void co_download(Particle *p, unsigned int n) {
    const size_t sz = n * sizeof(double);
    double *hx = (double*)malloc(sz), *hy = (double*)malloc(sz), *hz = (double*)malloc(sz);
    double *hvx = (double*)malloc(sz), *hvy = (double*)malloc(sz), *hvz = (double*)malloc(sz);
    double *hfx = (double*)malloc(sz), *hfy = (double*)malloc(sz), *hfz = (double*)malloc(sz);
    CUDA_CHECK(cudaMemcpy(hx, c_x, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hy, c_y, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hz, c_z, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hvx, c_vx, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hvy, c_vy, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hvz, c_vz, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hfx, c_fx, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hfy, c_fy, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hfz, c_fz, sz, cudaMemcpyDeviceToHost));
    for (unsigned int i = 0; i < n; ++i) {
        p[i].x = hx[i];  p[i].y = hy[i];  p[i].z = hz[i];
        p[i].vx = hvx[i]; p[i].vy = hvy[i]; p[i].vz = hvz[i];
        p[i].fx = hfx[i]; p[i].fy = hfy[i]; p[i].fz = hfz[i];
    }
    free(hx); free(hy); free(hz); free(hvx); free(hvy); free(hvz);
    free(hfx); free(hfy); free(hfz);
}

static void co_build_cell_lists(unsigned int n, unsigned int blocks, unsigned int threads,
                                double box_size) {
    CUDA_CHECK(cudaMemset(c_cell_count, 0, c_n_cells * sizeof(int)));
    cok_cell_assign<<<blocks, threads>>>(
        c_x, c_y, c_z, c_cell_id, c_cell_count, n, box_size, c_n_cells_side);
    thrust::exclusive_scan(thrust::device, c_cell_count, c_cell_count + c_n_cells, c_cell_start);
    CUDA_CHECK(cudaMemset(c_cell_count, 0, c_n_cells * sizeof(int))); // reuse as cursor
    cok_cell_fill<<<blocks, threads>>>(c_cell_id, c_cell_start, c_cell_part, c_cell_count, n);
    // after fill, c_cell_count[c] == #particles in cell c again
}

static double co_forces(unsigned int n, double box_size, int want_pe) {
    const unsigned int fblocks = (n * CO_WARP + CO_FORCE_BLOCK - 1) / CO_FORCE_BLOCK;
    cok_forces<<<fblocks, CO_FORCE_BLOCK>>>(
        c_x, c_y, c_z, c_fx, c_fy, c_fz, c_pe_arr,
        c_cell_id, c_cell_start, c_cell_count, c_cell_part,
        n, box_size, C_V_SHIFT, c_n_cells_side);

    if (!want_pe) return 0.0;
    CUDA_CHECK(cudaMemsetAsync(c_pe_total, 0, sizeof(double)));
    const unsigned int rth = 256;
    cok_reduce_pe<<<(n + rth - 1) / rth, rth, rth * sizeof(double)>>>(c_pe_arr, c_pe_total, n);
    double pe = 0.0;
    CUDA_CHECK(cudaMemcpy(&pe, c_pe_total, sizeof(double), cudaMemcpyDeviceToHost));
    return pe;
}

static double co_step(unsigned int n, double box_size, int want_pe) {
    const unsigned int threads = 256;
    const unsigned int blocks = (n + threads - 1) / threads;

    cok_kick_drift<<<blocks, threads>>>(
        c_x, c_y, c_z, c_vx, c_vy, c_vz, c_fx, c_fy, c_fz, n, box_size);

    c_step_counter++;
    if ((c_step_counter % CO_SPATIAL_SORT_EVERY) == 0) {
        cok_gather6<<<blocks, threads>>>(
            c_cell_part, c_x, c_y, c_z, c_vx, c_vy, c_vz,
            c_xs, c_ys, c_zs, c_vxs, c_vys, c_vzs, n);
        CUDA_CHECK(cudaMemcpy(c_x, c_xs, n * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(c_y, c_ys, n * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(c_z, c_zs, n * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(c_vx, c_vxs, n * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(c_vy, c_vys, n * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(c_vz, c_vzs, n * sizeof(double), cudaMemcpyDeviceToDevice));
    }

    co_build_cell_lists(n, blocks, threads, box_size);
    double pe = co_forces(n, box_size, want_pe);

    cok_kick<<<blocks, threads>>>(c_vx, c_vy, c_vz, c_fx, c_fy, c_fz, n);
    CUDA_CHECK(cudaGetLastError());
    return pe;
}

// =============================================================================
// PUBLIC ENTRY POINT
// =============================================================================

SimulationResult run_simulation_gpu_v8_3d(Particle *particles, unsigned int n,
                                          unsigned int nsteps, double box_size,
                                          int log_steps) {
    co_alloc(n, box_size);
    co_upload(particles, n);
    C_V_SHIFT = compute_v_shift();   // reused from lennard-jones.cu
    c_step_counter = 0;

    const unsigned int threads = 256;
    const unsigned int blocks = (n + threads - 1) / threads;

    co_build_cell_lists(n, blocks, threads, box_size);

    SimulationResult out;
    out.start_potential = co_forces(n, box_size, 1);
    out.start_kinetic   = compute_ke(particles, n);
    out.start_total     = out.start_kinetic + out.start_potential;
    out.final_potential = out.start_potential;

    // ---- timed region: integration loop only --------------------------------
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));

    for (unsigned int step = 0; step < nsteps; ++step) {
        const int want_pe = log_steps || (step + 1u == nsteps);
        double pe = co_step(n, box_size, want_pe);
        if (want_pe) out.final_potential = pe;
        if (log_steps) {
            CUDA_CHECK(cudaDeviceSynchronize());
            co_download(particles, n);
            out.final_kinetic = compute_ke(particles, n);
            out.final_total   = out.final_kinetic + out.final_potential;
            printf("step=%6u KE=%12.6f PE=%12.6f E=%12.6f\n",
                   step, out.final_kinetic, out.final_potential, out.final_total);
        }
    }

    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    // -------------------------------------------------------------------------

    co_download(particles, n);
    out.final_kinetic = compute_ke(particles, n);
    out.final_total   = out.final_kinetic + out.final_potential;
    out.n = n;
    out.particles = particles;

    printf("[CELL-ONLY] cell lists rebuilt every step (%u steps)\n", nsteps);
    printf("Simulation time %u steps: %.3f seconds\n", nsteps, ms / 1000.0f);
    return out;
}
