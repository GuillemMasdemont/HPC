// =============================================================================
//  Lennard-Jones — VERLET+CELL hybrid with CUDA GRAPHS  [graph variant]
// -----------------------------------------------------------------------------
//  Goal: remove per-step kernel-launch overhead and per-step host syncs.
//
//  How it differs from lennard-jones.cu (v6/v7):
//   * The steady step (kick_drift -> force -> kick) is captured ONCE into a
//     cudaGraph and replayed with a single cudaGraphLaunch per step -> no
//     per-kernel launch latency, no host interaction.
//   * A captured graph bakes in buffer POINTERS, so we must NOT pointer-swap
//     the spatial sort here; we copy sorted data back into the fixed buffers.
//   * To make the step fully host-free we drop the per-step rebuild CHECK and
//     instead rebuild on a FIXED PERIOD (GR_REBUILD_EVERY). The Verlet skin is
//     sized so the list stays valid across that many steps. Rebuild steps and
//     PE/logging steps fall back to a manual launch path (they are rare).
//
//  Self-contained: own device state + kernels (prefixed grk_/gr_). Reuses only
//  compute_ke() / compute_v_shift() from lennard-jones.cu via the header.
// =============================================================================

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>

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

// ─── tunables ─────────────────────────────────────────────────────────────────
#define GR_WARP               32u
#define GR_FORCE_BLOCK        256u
#define GR_SPATIAL_SORT_EVERY 4
#define GR_REBUILD_EVERY      8      // fixed rebuild period (steps)
#define GR_SKIN_SAFETY        1.5    // margin on the analytical skin
#define GR_VEL_SIGMA          3.0    // skin sized off GR_VEL_SIGMA * v_rms (tail/heating)
#define GR_NEIGH_SAFETY       4
#define GR_NEIGH_MIN          32

// ─── device state (file-local) ───────────────────────────────────────────────
static double *grd_x = nullptr, *grd_y = nullptr, *grd_z = nullptr;
static double *grd_vx = nullptr, *grd_vy = nullptr, *grd_vz = nullptr;
static double *grd_fx = nullptr, *grd_fy = nullptr, *grd_fz = nullptr;
static double *grd_pe_arr = nullptr, *grd_pe_total = nullptr;
static double *grd_xs = nullptr, *grd_ys = nullptr, *grd_zs = nullptr;
static double *grd_vxs = nullptr, *grd_vys = nullptr, *grd_vzs = nullptr;
static int *grd_cell_id = nullptr, *grd_cell_start = nullptr;
static int *grd_cell_count = nullptr, *grd_cell_part = nullptr;
static int *grd_nl = nullptr;
static unsigned int *grd_nl_count = nullptr;
static unsigned int *grd_overflow = nullptr;

static unsigned int grd_n = 0;
static unsigned int gr_n_cells_side = 0, gr_n_cells = 0;
static unsigned int gr_max_neighbours = 0;
static unsigned int gr_sort_counter = 0;
static double gr_skin = 0.0;
static double GR_V_SHIFT = 0.0;

static cudaStream_t  gr_stream = nullptr;
static cudaGraph_t   gr_graph = nullptr;
static cudaGraphExec_t gr_exec = nullptr;

#define GR_R_LIST_SQ ((R_CUT + gr_skin) * (R_CUT + gr_skin))


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

__global__ void grk_cell_assign(
    const double * __restrict__ x, const double * __restrict__ y, const double * __restrict__ z,
    int * __restrict__ cell_id, int * __restrict__ cell_count,
    unsigned int n, double box_size, unsigned int n_cells_side)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double cs = box_size / (double)n_cells_side;
    int cx = (int)(x[i] / cs); if (cx >= (int)n_cells_side) cx = n_cells_side - 1; if (cx < 0) cx = 0;
    int cy = (int)(y[i] / cs); if (cy >= (int)n_cells_side) cy = n_cells_side - 1; if (cy < 0) cy = 0;
    int cz = (int)(z[i] / cs); if (cz >= (int)n_cells_side) cz = n_cells_side - 1; if (cz < 0) cz = 0;
    int cid = cx + cy * n_cells_side + cz * n_cells_side * n_cells_side;
    cell_id[i] = cid;
    atomicAdd(&cell_count[cid], 1);
}

__global__ void grk_cell_fill(
    const int * __restrict__ cell_id, const int * __restrict__ cell_start,
    int * __restrict__ cell_part, int * __restrict__ cursor, unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int cid = cell_id[i];
    int slot = atomicAdd(&cursor[cid], 1);
    cell_part[cell_start[cid] + slot] = (int)i;
}

__global__ void grk_gather6(
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

__global__ void grk_build_nl_cell(
    const double * __restrict__ x, const double * __restrict__ y, const double * __restrict__ z,
    const int * __restrict__ cell_id,
    const int * __restrict__ cell_start, const int * __restrict__ cell_count,
    const int * __restrict__ cell_part,
    int * __restrict__ nl, unsigned int * __restrict__ nl_count,
    unsigned int * __restrict__ overflow,
    unsigned int n, unsigned int max_neighbours,
    double box_size, double r_list_sq, unsigned int n_cells_side)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double xi = x[i], yi = y[i], zi = z[i];
    int cid = cell_id[i];
    int n2 = n_cells_side * n_cells_side;
    int cz_i = cid / n2, cy_i = (cid % n2) / n_cells_side, cx_i = cid % n_cells_side;
    unsigned int count = 0;
    const double inv_box = 1.0 / box_size;

    for (int dz = -1; dz <= 1; dz++)
    for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++) {
        int nx = cx_i + dx; if (nx < 0) nx += n_cells_side; if (nx >= (int)n_cells_side) nx -= n_cells_side;
        int ny = cy_i + dy; if (ny < 0) ny += n_cells_side; if (ny >= (int)n_cells_side) ny -= n_cells_side;
        int nz = cz_i + dz; if (nz < 0) nz += n_cells_side; if (nz >= (int)n_cells_side) nz -= n_cells_side;
        int ncid = nx + ny * n_cells_side + nz * n2;
        int start = cell_start[ncid];
        int cnt = cell_count[ncid];
        for (int s = 0; s < cnt; s++) {
            int j = cell_part[start + s];
            if (j == (int)i) continue;                 // FULL list
            double ddx = xi - x[j], ddy = yi - y[j], ddz = zi - z[j];
            ddx -= box_size * rint(ddx * inv_box);
            ddy -= box_size * rint(ddy * inv_box);
            ddz -= box_size * rint(ddz * inv_box);
            if (ddx*ddx + ddy*ddy + ddz*ddz < r_list_sq) {
                if (count < max_neighbours) nl[i * max_neighbours + count] = j;
                count++;
            }
        }
    }
    if (count > max_neighbours) { nl_count[i] = max_neighbours; atomicOr(overflow, i + 1u); }
    else                        { nl_count[i] = count; }
}

__global__ void grk_forces(
    const double * __restrict__ x, const double * __restrict__ y, const double * __restrict__ z,
    double * __restrict__ fx, double * __restrict__ fy, double * __restrict__ fz,
    double * __restrict__ pe_out,
    const int * __restrict__ nl, const unsigned int * __restrict__ nl_count,
    unsigned int n, unsigned int max_neighbours, double box_size, double v_shift)
{
    const unsigned int i    = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    if (i >= n) return;

    const double xi = x[i], yi = y[i], zi = z[i];
    const double r_cut_sq = R_CUT * R_CUT;
    const double sig2 = SIGMA * SIGMA;
    const double inv_box = 1.0 / box_size;
    const unsigned int nn = nl_count[i];

    double Fx = 0.0, Fy = 0.0, Fz = 0.0, Pe = 0.0;
    for (unsigned int k = lane; k < nn; k += GR_WARP) {
        const unsigned int j = (unsigned int)nl[i * max_neighbours + k];
        double dx = xi - x[j], dy = yi - y[j], dz = zi - z[j];
        dx -= box_size * rint(dx * inv_box);
        dy -= box_size * rint(dy * inv_box);
        dz -= box_size * rint(dz * inv_box);
        double r2 = dx*dx + dy*dy + dz*dz;
        if (r2 >= r_cut_sq || r2 == 0.0) continue;
        double inv_r2 = 1.0 / r2;
        double sr2 = sig2 * inv_r2;
        double sr6 = sr2 * sr2 * sr2;
        double sr12 = sr6 * sr6;
        double fij = 24.0 * EPSILON * (2.0 * sr12 - sr6) * inv_r2;
        Fx += fij * dx; Fy += fij * dy; Fz += fij * dz;
        Pe += 0.5 * (4.0 * EPSILON * (sr12 - sr6) - v_shift);
    }
    for (int off = 16; off > 0; off >>= 1) {
        Fx += __shfl_down_sync(0xffffffffu, Fx, off);
        Fy += __shfl_down_sync(0xffffffffu, Fy, off);
        Fz += __shfl_down_sync(0xffffffffu, Fz, off);
        Pe += __shfl_down_sync(0xffffffffu, Pe, off);
    }
    if (lane == 0u) { fx[i] = Fx; fy[i] = Fy; fz[i] = Fz; pe_out[i] = Pe; }
}

__global__ void grk_reduce_pe(
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

__global__ void grk_kick_drift(
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

__global__ void grk_kick(
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

// =============================================================================
// HOST HELPERS
// =============================================================================

static void gr_alloc(unsigned int n, double box_size) {
    if (grd_n == n) return;

    const double box_vol = box_size * box_size * box_size;
    const double density = (double)n / box_vol;
    const double r = R_CUT + gr_skin;
    const double expected = density * (4.0 / 3.0) * M_PI * r * r * r;
    const unsigned int computed = (unsigned int)(GR_NEIGH_SAFETY * expected) + 1;
    gr_max_neighbours = (computed > GR_NEIGH_MIN) ? computed : GR_NEIGH_MIN;

    gr_n_cells_side = (unsigned int)(box_size / (R_CUT + gr_skin));
    if (gr_n_cells_side < 1) gr_n_cells_side = 1;
    gr_n_cells = gr_n_cells_side * gr_n_cells_side * gr_n_cells_side;

    fprintf(stdout,
            "[GRAPH] skin=%.4f (%.3f*R_CUT)  rebuild_every=%d  density=%.4f  "
            "expected_nn=%.1f  max_neighbours=%u  cells=%u(%u/side)\n",
            gr_skin, gr_skin / R_CUT, GR_REBUILD_EVERY, density, expected,
            gr_max_neighbours, gr_n_cells, gr_n_cells_side);

    const size_t sz = (size_t)n * sizeof(double);
    CUDA_CHECK(cudaMalloc(&grd_x, sz));  CUDA_CHECK(cudaMalloc(&grd_y, sz));  CUDA_CHECK(cudaMalloc(&grd_z, sz));
    CUDA_CHECK(cudaMalloc(&grd_vx, sz)); CUDA_CHECK(cudaMalloc(&grd_vy, sz)); CUDA_CHECK(cudaMalloc(&grd_vz, sz));
    CUDA_CHECK(cudaMalloc(&grd_fx, sz)); CUDA_CHECK(cudaMalloc(&grd_fy, sz)); CUDA_CHECK(cudaMalloc(&grd_fz, sz));
    CUDA_CHECK(cudaMalloc(&grd_pe_arr, sz));
    CUDA_CHECK(cudaMalloc(&grd_pe_total, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&grd_xs, sz));  CUDA_CHECK(cudaMalloc(&grd_ys, sz));  CUDA_CHECK(cudaMalloc(&grd_zs, sz));
    CUDA_CHECK(cudaMalloc(&grd_vxs, sz)); CUDA_CHECK(cudaMalloc(&grd_vys, sz)); CUDA_CHECK(cudaMalloc(&grd_vzs, sz));
    CUDA_CHECK(cudaMalloc(&grd_cell_id, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&grd_cell_count, (size_t)gr_n_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&grd_cell_start, (size_t)gr_n_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&grd_cell_part, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&grd_nl_count, (size_t)n * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&grd_nl, (size_t)n * gr_max_neighbours * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&grd_overflow, sizeof(unsigned int)));

    CUDA_CHECK(cudaStreamCreateWithFlags(&gr_stream, cudaStreamNonBlocking));
    grd_n = n;
}

static void gr_upload(const Particle *p, unsigned int n) {
    const size_t sz = n * sizeof(double);
    double *hx = (double*)malloc(sz), *hy = (double*)malloc(sz), *hz = (double*)malloc(sz);
    double *hvx = (double*)malloc(sz), *hvy = (double*)malloc(sz), *hvz = (double*)malloc(sz);
    for (unsigned int i = 0; i < n; ++i) {
        hx[i] = p[i].x;  hy[i] = p[i].y;  hz[i] = p[i].z;
        hvx[i] = p[i].vx; hvy[i] = p[i].vy; hvz[i] = p[i].vz;
    }
    CUDA_CHECK(cudaMemcpy(grd_x, hx, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(grd_y, hy, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(grd_z, hz, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(grd_vx, hvx, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(grd_vy, hvy, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(grd_vz, hvz, sz, cudaMemcpyHostToDevice));
    free(hx); free(hy); free(hz); free(hvx); free(hvy); free(hvz);
}

static void gr_download(Particle *p, unsigned int n) {
    const size_t sz = n * sizeof(double);
    double *hx = (double*)malloc(sz), *hy = (double*)malloc(sz), *hz = (double*)malloc(sz);
    double *hvx = (double*)malloc(sz), *hvy = (double*)malloc(sz), *hvz = (double*)malloc(sz);
    double *hfx = (double*)malloc(sz), *hfy = (double*)malloc(sz), *hfz = (double*)malloc(sz);
    CUDA_CHECK(cudaMemcpy(hx, grd_x, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hy, grd_y, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hz, grd_z, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hvx, grd_vx, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hvy, grd_vy, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hvz, grd_vz, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hfx, grd_fx, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hfy, grd_fy, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hfz, grd_fz, sz, cudaMemcpyDeviceToHost));
    for (unsigned int i = 0; i < n; ++i) {
        p[i].x = hx[i];  p[i].y = hy[i];  p[i].z = hz[i];
        p[i].vx = hvx[i]; p[i].vy = hvy[i]; p[i].vz = hvz[i];
        p[i].fx = hfx[i]; p[i].fy = hfy[i]; p[i].fz = hfz[i];
    }
    free(hx); free(hy); free(hz); free(hvx); free(hvy); free(hvz);
    free(hfx); free(hfy); free(hfz);
}

static void gr_build_cell_lists(unsigned int n, unsigned int blocks, unsigned int threads,
                                double box_size) {
    CUDA_CHECK(cudaMemsetAsync(grd_cell_count, 0, gr_n_cells * sizeof(int), gr_stream));
    grk_cell_assign<<<blocks, threads, 0, gr_stream>>>(
        grd_x, grd_y, grd_z, grd_cell_id, grd_cell_count, n, box_size, gr_n_cells_side);
    thrust::exclusive_scan(thrust::cuda::par.on(gr_stream),
                           grd_cell_count, grd_cell_count + gr_n_cells, grd_cell_start);
    CUDA_CHECK(cudaMemsetAsync(grd_cell_count, 0, gr_n_cells * sizeof(int), gr_stream));
    grk_cell_fill<<<blocks, threads, 0, gr_stream>>>(
        grd_cell_id, grd_cell_start, grd_cell_part, grd_cell_count, n);
}

// Full neighbour-list rebuild. Uses copy-back (NOT pointer swap) so the captured
// graph keeps referencing the same buffers.
static void gr_build_nl(unsigned int n, double box_size) {
    const unsigned int threads = 256;
    const unsigned int blocks = (n + threads - 1) / threads;

    gr_build_cell_lists(n, blocks, threads, box_size);

    gr_sort_counter++;
    if ((gr_sort_counter % GR_SPATIAL_SORT_EVERY) == 0) {
        grk_gather6<<<blocks, threads, 0, gr_stream>>>(
            grd_cell_part, grd_x, grd_y, grd_z, grd_vx, grd_vy, grd_vz,
            grd_xs, grd_ys, grd_zs, grd_vxs, grd_vys, grd_vzs, n);
        CUDA_CHECK(cudaMemcpyAsync(grd_x, grd_xs, n * sizeof(double), cudaMemcpyDeviceToDevice, gr_stream));
        CUDA_CHECK(cudaMemcpyAsync(grd_y, grd_ys, n * sizeof(double), cudaMemcpyDeviceToDevice, gr_stream));
        CUDA_CHECK(cudaMemcpyAsync(grd_z, grd_zs, n * sizeof(double), cudaMemcpyDeviceToDevice, gr_stream));
        CUDA_CHECK(cudaMemcpyAsync(grd_vx, grd_vxs, n * sizeof(double), cudaMemcpyDeviceToDevice, gr_stream));
        CUDA_CHECK(cudaMemcpyAsync(grd_vy, grd_vys, n * sizeof(double), cudaMemcpyDeviceToDevice, gr_stream));
        CUDA_CHECK(cudaMemcpyAsync(grd_vz, grd_vzs, n * sizeof(double), cudaMemcpyDeviceToDevice, gr_stream));
        gr_build_cell_lists(n, blocks, threads, box_size);
    }

    CUDA_CHECK(cudaMemsetAsync(grd_overflow, 0, sizeof(unsigned int), gr_stream));
    grk_build_nl_cell<<<blocks, threads, 0, gr_stream>>>(
        grd_x, grd_y, grd_z, grd_cell_id, grd_cell_start, grd_cell_count, grd_cell_part,
        grd_nl, grd_nl_count, grd_overflow, n, gr_max_neighbours,
        box_size, GR_R_LIST_SQ, gr_n_cells_side);

    unsigned int overflow = 0;
    CUDA_CHECK(cudaMemcpyAsync(&overflow, grd_overflow, sizeof(unsigned int),
                               cudaMemcpyDeviceToHost, gr_stream));
    CUDA_CHECK(cudaStreamSynchronize(gr_stream));
    if (overflow > 0) {
        fprintf(stderr, "[GRAPH] OVERFLOW: particle %u exceeds max_neighbours=%u. "
                        "Increase skin/GR_NEIGH_SAFETY.\n", overflow - 1u, gr_max_neighbours);
        exit(1);
    }
}

// launch the force kernel; reduce + copy PE only when requested
static double gr_forces(unsigned int n, double box_size, int want_pe) {
    const unsigned int fblocks = (n * GR_WARP + GR_FORCE_BLOCK - 1) / GR_FORCE_BLOCK;
    grk_forces<<<fblocks, GR_FORCE_BLOCK, 0, gr_stream>>>(
        grd_x, grd_y, grd_z, grd_fx, grd_fy, grd_fz, grd_pe_arr,
        grd_nl, grd_nl_count, n, gr_max_neighbours, box_size, GR_V_SHIFT);
    if (!want_pe) return 0.0;
    CUDA_CHECK(cudaMemsetAsync(grd_pe_total, 0, sizeof(double), gr_stream));
    const unsigned int rth = 256;
    grk_reduce_pe<<<(n + rth - 1) / rth, rth, rth * sizeof(double), gr_stream>>>(
        grd_pe_arr, grd_pe_total, n);
    double pe = 0.0;
    CUDA_CHECK(cudaMemcpyAsync(&pe, grd_pe_total, sizeof(double), cudaMemcpyDeviceToHost, gr_stream));
    CUDA_CHECK(cudaStreamSynchronize(gr_stream));
    return pe;
}

// Capture the steady step (kick_drift -> force[no PE] -> kick) into a graph.
static void gr_capture_steady(unsigned int n, double box_size) {
    const unsigned int threads = 256;
    const unsigned int blocks = (n + threads - 1) / threads;
    const unsigned int fblocks = (n * GR_WARP + GR_FORCE_BLOCK - 1) / GR_FORCE_BLOCK;

    CUDA_CHECK(cudaStreamBeginCapture(gr_stream, cudaStreamCaptureModeGlobal));
    grk_kick_drift<<<blocks, threads, 0, gr_stream>>>(
        grd_x, grd_y, grd_z, grd_vx, grd_vy, grd_vz, grd_fx, grd_fy, grd_fz, n, box_size);
    grk_forces<<<fblocks, GR_FORCE_BLOCK, 0, gr_stream>>>(
        grd_x, grd_y, grd_z, grd_fx, grd_fy, grd_fz, grd_pe_arr,
        grd_nl, grd_nl_count, n, gr_max_neighbours, box_size, GR_V_SHIFT);
    grk_kick<<<blocks, threads, 0, gr_stream>>>(grd_vx, grd_vy, grd_vz, grd_fx, grd_fy, grd_fz, n);
    CUDA_CHECK(cudaStreamEndCapture(gr_stream, &gr_graph));

#if CUDART_VERSION >= 12000
    CUDA_CHECK(cudaGraphInstantiate(&gr_exec, gr_graph, 0));
#else
    CUDA_CHECK(cudaGraphInstantiate(&gr_exec, gr_graph, nullptr, nullptr, 0));
#endif
}

// manual (non-graph) step: kick_drift -> [rebuild] -> force -> kick
static double gr_manual_step(unsigned int n, double box_size, int do_rebuild, int want_pe) {
    const unsigned int threads = 256;
    const unsigned int blocks = (n + threads - 1) / threads;
    grk_kick_drift<<<blocks, threads, 0, gr_stream>>>(
        grd_x, grd_y, grd_z, grd_vx, grd_vy, grd_vz, grd_fx, grd_fy, grd_fz, n, box_size);
    if (do_rebuild) gr_build_nl(n, box_size);
    double pe = gr_forces(n, box_size, want_pe);
    grk_kick<<<blocks, threads, 0, gr_stream>>>(grd_vx, grd_vy, grd_vz, grd_fx, grd_fy, grd_fz, n);
    return pe;
}

// =============================================================================
// PUBLIC ENTRY POINT
// =============================================================================

SimulationResult run_simulation_gpu_graph_3d(Particle *particles, unsigned int n,
                                             unsigned int nsteps, double box_size,
                                             int log_steps) {
    // Size the skin so the list survives GR_REBUILD_EVERY steps without a check.
    // IMPORTANT: size off the RMS speed (set by temperature, INDEPENDENT of N),
    // not the per-particle MAX. The max of n samples is an extreme-value
    // statistic that grows ~sqrt(ln n) with N, which would inflate the skin and
    // make the force kernel heavier at large N. We instead use v_rms times a
    // fixed tail/heating multiplier (GR_VEL_SIGMA), giving an N-independent skin.
    // Safety condition (fixed period, no per-step check): max single-particle
    // drift over P steps <= skin/2  =>  skin >= 2 * v_char * DT * P.
    double v2sum = 0.0;
    for (unsigned int i = 0; i < n; ++i) {
        v2sum += particles[i].vx * particles[i].vx
               + particles[i].vy * particles[i].vy
               + particles[i].vz * particles[i].vz;
    }
    double v_rms  = (n > 0) ? sqrt(v2sum / (double)n) : 0.0;
    double v_char = GR_VEL_SIGMA * v_rms;   // cover the fast tail + heating
    gr_skin = GR_SKIN_SAFETY * 2.0 * v_char * DT * (double)GR_REBUILD_EVERY;
    if (gr_skin < 0.15 * R_CUT) gr_skin = 0.15 * R_CUT;
    if (gr_skin > 1.00 * R_CUT) gr_skin = 1.00 * R_CUT;

    gr_alloc(n, box_size);
    gr_upload(particles, n);
    GR_V_SHIFT = compute_v_shift();
    gr_sort_counter = 0;

    gr_build_nl(n, box_size);

    SimulationResult out;
    out.start_potential = gr_forces(n, box_size, 1);
    out.start_kinetic   = compute_ke(particles, n);
    out.start_total     = out.start_kinetic + out.start_potential;
    out.final_potential = out.start_potential;

    gr_capture_steady(n, box_size);

    // ---- timed region: integration loop only --------------------------------
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0, gr_stream));

    for (unsigned int step = 0; step < nsteps; ++step) {
        const int do_rebuild = (step > 0) && (step % GR_REBUILD_EVERY == 0);
        const int want_pe = log_steps || (step + 1u == nsteps);

        if (do_rebuild || want_pe || log_steps) {
            double pe = gr_manual_step(n, box_size, do_rebuild, want_pe);   // rare path
            if (want_pe) out.final_potential = pe;
            if (log_steps) {
                CUDA_CHECK(cudaStreamSynchronize(gr_stream));
                gr_download(particles, n);
                out.final_kinetic = compute_ke(particles, n);
                out.final_total   = out.final_kinetic + out.final_potential;
                printf("step=%6u KE=%12.6f PE=%12.6f E=%12.6f\n",
                       step, out.final_kinetic, out.final_potential, out.final_total);
            }
        } else {
            CUDA_CHECK(cudaGraphLaunch(gr_exec, gr_stream));   // common path: 1 launch
        }
    }

    CUDA_CHECK(cudaEventRecord(t1, gr_stream));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    // -------------------------------------------------------------------------

    CUDA_CHECK(cudaStreamSynchronize(gr_stream));
    gr_download(particles, n);
    out.final_kinetic = compute_ke(particles, n);
    out.final_total   = out.final_kinetic + out.final_potential;
    out.n = n;
    out.particles = particles;

    printf("[GRAPH] steady step captured; rebuild every %d steps (skin %.3f*R_CUT)\n",
           GR_REBUILD_EVERY, gr_skin / R_CUT);
    printf("Simulation time %u steps: %.3f seconds\n", nsteps, ms / 1000.0f);
    return out;
}
