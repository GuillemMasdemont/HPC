// =============================================================================
//  Lennard-Jones — VERLET+CELL hybrid with WARP-PER-PARTICLE list build
// -----------------------------------------------------------------------------
//  Same as lennard-jones.cu (v6/v7): tuning-free analytical skin, drift-
//  triggered rebuild, warp-per-particle FULL-list force kernel, PE computed
//  only when needed, ping-pong spatial sort.
//
//  ONLY DIFFERENCE: the neighbour-list BUILD (wnk_build_nl_cell) assigns one
//  WARP (32 lanes) per particle instead of one thread. Lanes stride over the
//  ~500 candidates in the 27-cell stencil (coalesced reads) and append hits
//  with a warp-aggregated prefix sum (__ballot_sync + __popc), giving coalesced
//  list writes and no atomics. Correctness note: all 32 lanes execute the inner
//  loop the same number of times so __ballot_sync is called by the full warp.
//
//  Self-contained: own state + kernels (prefixed wn_/wnk_/wnd_). Reuses only
//  compute_ke() / compute_v_shift() from lennard-jones.cu via the header.
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

#define WN_WARP               32u
#define WN_FORCE_BLOCK        256u
#define WN_BUILD_BLOCK        256u   // 8 warps -> 8 particles per block
#define WN_SPATIAL_SORT_EVERY 4
#define WN_NEIGH_SAFETY       4
#define WN_NEIGH_MIN          32

// ─── device state (file-local) ───────────────────────────────────────────────
static double *wnd_x = nullptr, *wnd_y = nullptr, *wnd_z = nullptr;
static double *wnd_vx = nullptr, *wnd_vy = nullptr, *wnd_vz = nullptr;
static double *wnd_fx = nullptr, *wnd_fy = nullptr, *wnd_fz = nullptr;
static double *wnd_pe_arr = nullptr, *wnd_pe_total = nullptr;
static double *wnd_xs = nullptr, *wnd_ys = nullptr, *wnd_zs = nullptr;
static double *wnd_vxs = nullptr, *wnd_vys = nullptr, *wnd_vzs = nullptr;
static double *wnd_x_ref = nullptr, *wnd_y_ref = nullptr, *wnd_z_ref = nullptr;
static int *wnd_cell_id = nullptr, *wnd_cell_start = nullptr;
static int *wnd_cell_count = nullptr, *wnd_cell_part = nullptr;
static int *wnd_nl = nullptr;
static unsigned int *wnd_nl_count = nullptr;
static unsigned int *wnd_flag = nullptr, *wnd_overflow = nullptr;
static unsigned int *wn_flag_pinned = nullptr;

static unsigned int wnd_n = 0;
static unsigned int wn_n_cells_side = 0, wn_n_cells = 0;
static unsigned int wn_max_neighbours = 0;
static unsigned int wn_sort_counter = 0;
static double wn_skin = 0.0;
static double WN_V_SHIFT = 0.0;

#define WN_R_LIST_SQ    ((R_CUT + wn_skin) * (R_CUT + wn_skin))
#define WN_HALF_SKIN_SQ ((wn_skin * 0.5) * (wn_skin * 0.5))

static double wn_skin_frac(unsigned int n) {
    double f = 0.15 + 6.0 / sqrt((double)n);
    if (f < 0.10) f = 0.10;
    if (f > 0.60) f = 0.60;
    return f;
}

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

__global__ void wnk_cell_assign(
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

__global__ void wnk_cell_fill(
    const int * __restrict__ cell_id, const int * __restrict__ cell_start,
    int * __restrict__ cell_part, int * __restrict__ cursor, unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int cid = cell_id[i];
    int slot = atomicAdd(&cursor[cid], 1);
    cell_part[cell_start[cid] + slot] = (int)i;
}

__global__ void wnk_gather6(
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

// ---- WARP-PER-PARTICLE neighbour-list build --------------------------------
__global__ void wnk_build_nl_cell(
    const double * __restrict__ x, const double * __restrict__ y, const double * __restrict__ z,
    const int * __restrict__ cell_id,
    const int * __restrict__ cell_start, const int * __restrict__ cell_count,
    const int * __restrict__ cell_part,
    int * __restrict__ nl, unsigned int * __restrict__ nl_count,
    unsigned int * __restrict__ overflow,
    unsigned int n, unsigned int max_neighbours,
    double box_size, double r_list_sq, unsigned int n_cells_side)
{
    const unsigned int i    = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; // warp == particle
    const unsigned int lane = threadIdx.x & 31u;
    if (i >= n) return;

    const double xi = x[i], yi = y[i], zi = z[i];
    const int cid = cell_id[i];
    const int n2 = n_cells_side * n_cells_side;
    const int cz_i = cid / n2;
    const int cy_i = (cid % n2) / n_cells_side;
    const int cx_i = cid % n_cells_side;
    const double inv_box = 1.0 / box_size;

    unsigned int base = 0;   // warp-uniform running count of stored neighbours
    int overflowed = 0;

    for (int dz = -1; dz <= 1; dz++)
    for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++) {
        int nx = cx_i + dx; if (nx < 0) nx += n_cells_side; if (nx >= (int)n_cells_side) nx -= n_cells_side;
        int ny = cy_i + dy; if (ny < 0) ny += n_cells_side; if (ny >= (int)n_cells_side) ny -= n_cells_side;
        int nz = cz_i + dz; if (nz < 0) nz += n_cells_side; if (nz >= (int)n_cells_side) nz -= n_cells_side;
        int ncid = nx + ny * n_cells_side + nz * n2;
        int start = cell_start[ncid];
        int cnt = cell_count[ncid];

        // Uniform loop bound (cnt) -> every lane iterates the same # of times,
        // so __ballot_sync below is executed by the full warp.
        for (int s0 = 0; s0 < cnt; s0 += (int)WN_WARP) {
            int s = s0 + (int)lane;
            int j = -1;
            int hit = 0;
            if (s < cnt) {
                j = cell_part[start + s];
                if (j != (int)i) {
                    double ddx = xi - x[j], ddy = yi - y[j], ddz = zi - z[j];
                    ddx -= box_size * rint(ddx * inv_box);
                    ddy -= box_size * rint(ddy * inv_box);
                    ddz -= box_size * rint(ddz * inv_box);
                    if (ddx*ddx + ddy*ddy + ddz*ddz < r_list_sq) hit = 1;
                }
            }
            unsigned int mask = __ballot_sync(0xffffffffu, hit);
            unsigned int offset = __popc(mask & ((1u << lane) - 1u)); // hits in lower lanes
            if (hit) {
                unsigned int slot = base + offset;
                if (slot < max_neighbours) nl[i * max_neighbours + slot] = j;
                else overflowed = 1;
            }
            base += __popc(mask);   // same increment on every lane -> base stays uniform
        }
    }

    if (lane == 0u)
        nl_count[i] = (base <= max_neighbours) ? base : max_neighbours;
    if (__any_sync(0xffffffffu, overflowed) && lane == 0u)
        atomicOr(overflow, i + 1u);
}

__global__ void wnk_forces(
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
    for (unsigned int k = lane; k < nn; k += WN_WARP) {
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

__global__ void wnk_reduce_pe(
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

__global__ void wnk_kick_drift(
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

__global__ void wnk_kick(
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

__global__ void wnk_check_rebuild(
    const double * __restrict__ x, const double * __restrict__ y, const double * __restrict__ z,
    const double * __restrict__ x_ref, const double * __restrict__ y_ref, const double * __restrict__ z_ref,
    unsigned int n, double box_size, double half_skin_sq, unsigned int *flag)
{
    extern __shared__ unsigned int s_flag[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    s_flag[tid] = 0;
    if (i < n) {
        double dx = x[i] - x_ref[i], dy = y[i] - y_ref[i], dz = z[i] - z_ref[i];
        dx -= box_size * nearbyint(dx / box_size);
        dy -= box_size * nearbyint(dy / box_size);
        dz -= box_size * nearbyint(dz / box_size);
        if (dx*dx + dy*dy + dz*dz > half_skin_sq) s_flag[tid] = 1;
    }
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_flag[tid] |= s_flag[tid + stride];
        __syncthreads();
    }
    if (tid == 0 && s_flag[0]) atomicOr(flag, 1u);
}

// =============================================================================
// HOST HELPERS
// =============================================================================

static void wn_alloc(unsigned int n, double box_size) {
    if (wnd_n == n) return;

    wn_skin = wn_skin_frac(n) * R_CUT;

    const double box_vol = box_size * box_size * box_size;
    const double density = (double)n / box_vol;
    const double r = R_CUT + wn_skin;
    const double expected = density * (4.0 / 3.0) * M_PI * r * r * r;
    const unsigned int computed = (unsigned int)(WN_NEIGH_SAFETY * expected) + 1;
    wn_max_neighbours = (computed > WN_NEIGH_MIN) ? computed : WN_NEIGH_MIN;

    wn_n_cells_side = (unsigned int)(box_size / (R_CUT + wn_skin));
    if (wn_n_cells_side < 1) wn_n_cells_side = 1;
    wn_n_cells = wn_n_cells_side * wn_n_cells_side * wn_n_cells_side;

    fprintf(stdout,
            "[WARP-NL] skin=%.4f (%.3f*R_CUT)  density=%.4f  expected_nn=%.1f  "
            "max_neighbours=%u  cells=%u(%u/side)  WARP-per-particle build\n",
            wn_skin, wn_skin / R_CUT, density, expected,
            wn_max_neighbours, wn_n_cells, wn_n_cells_side);

    const size_t sz = (size_t)n * sizeof(double);
    CUDA_CHECK(cudaMalloc(&wnd_x, sz));  CUDA_CHECK(cudaMalloc(&wnd_y, sz));  CUDA_CHECK(cudaMalloc(&wnd_z, sz));
    CUDA_CHECK(cudaMalloc(&wnd_vx, sz)); CUDA_CHECK(cudaMalloc(&wnd_vy, sz)); CUDA_CHECK(cudaMalloc(&wnd_vz, sz));
    CUDA_CHECK(cudaMalloc(&wnd_fx, sz)); CUDA_CHECK(cudaMalloc(&wnd_fy, sz)); CUDA_CHECK(cudaMalloc(&wnd_fz, sz));
    CUDA_CHECK(cudaMalloc(&wnd_pe_arr, sz));
    CUDA_CHECK(cudaMalloc(&wnd_pe_total, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&wnd_xs, sz));  CUDA_CHECK(cudaMalloc(&wnd_ys, sz));  CUDA_CHECK(cudaMalloc(&wnd_zs, sz));
    CUDA_CHECK(cudaMalloc(&wnd_vxs, sz)); CUDA_CHECK(cudaMalloc(&wnd_vys, sz)); CUDA_CHECK(cudaMalloc(&wnd_vzs, sz));
    CUDA_CHECK(cudaMalloc(&wnd_x_ref, sz)); CUDA_CHECK(cudaMalloc(&wnd_y_ref, sz)); CUDA_CHECK(cudaMalloc(&wnd_z_ref, sz));
    CUDA_CHECK(cudaMalloc(&wnd_cell_id, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&wnd_cell_count, (size_t)wn_n_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&wnd_cell_start, (size_t)wn_n_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&wnd_cell_part, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&wnd_nl_count, (size_t)n * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&wnd_nl, (size_t)n * wn_max_neighbours * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&wnd_flag, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&wnd_overflow, sizeof(unsigned int)));
    CUDA_CHECK(cudaMallocHost(&wn_flag_pinned, sizeof(unsigned int)));

    wnd_n = n;
}

static void wn_upload(const Particle *p, unsigned int n) {
    const size_t sz = n * sizeof(double);
    double *hx = (double*)malloc(sz), *hy = (double*)malloc(sz), *hz = (double*)malloc(sz);
    double *hvx = (double*)malloc(sz), *hvy = (double*)malloc(sz), *hvz = (double*)malloc(sz);
    for (unsigned int i = 0; i < n; ++i) {
        hx[i] = p[i].x;  hy[i] = p[i].y;  hz[i] = p[i].z;
        hvx[i] = p[i].vx; hvy[i] = p[i].vy; hvz[i] = p[i].vz;
    }
    CUDA_CHECK(cudaMemcpy(wnd_x, hx, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(wnd_y, hy, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(wnd_z, hz, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(wnd_vx, hvx, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(wnd_vy, hvy, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(wnd_vz, hvz, sz, cudaMemcpyHostToDevice));
    free(hx); free(hy); free(hz); free(hvx); free(hvy); free(hvz);
}

static void wn_download(Particle *p, unsigned int n) {
    const size_t sz = n * sizeof(double);
    double *hx = (double*)malloc(sz), *hy = (double*)malloc(sz), *hz = (double*)malloc(sz);
    double *hvx = (double*)malloc(sz), *hvy = (double*)malloc(sz), *hvz = (double*)malloc(sz);
    double *hfx = (double*)malloc(sz), *hfy = (double*)malloc(sz), *hfz = (double*)malloc(sz);
    CUDA_CHECK(cudaMemcpy(hx, wnd_x, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hy, wnd_y, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hz, wnd_z, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hvx, wnd_vx, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hvy, wnd_vy, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hvz, wnd_vz, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hfx, wnd_fx, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hfy, wnd_fy, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hfz, wnd_fz, sz, cudaMemcpyDeviceToHost));
    for (unsigned int i = 0; i < n; ++i) {
        p[i].x = hx[i];  p[i].y = hy[i];  p[i].z = hz[i];
        p[i].vx = hvx[i]; p[i].vy = hvy[i]; p[i].vz = hvz[i];
        p[i].fx = hfx[i]; p[i].fy = hfy[i]; p[i].fz = hfz[i];
    }
    free(hx); free(hy); free(hz); free(hvx); free(hvy); free(hvz);
    free(hfx); free(hfy); free(hfz);
}

static void wn_build_cell_lists(unsigned int n, unsigned int blocks, unsigned int threads,
                                double box_size) {
    CUDA_CHECK(cudaMemset(wnd_cell_count, 0, wn_n_cells * sizeof(int)));
    wnk_cell_assign<<<blocks, threads>>>(
        wnd_x, wnd_y, wnd_z, wnd_cell_id, wnd_cell_count, n, box_size, wn_n_cells_side);
    thrust::exclusive_scan(thrust::device, wnd_cell_count, wnd_cell_count + wn_n_cells, wnd_cell_start);
    CUDA_CHECK(cudaMemset(wnd_cell_count, 0, wn_n_cells * sizeof(int)));
    wnk_cell_fill<<<blocks, threads>>>(wnd_cell_id, wnd_cell_start, wnd_cell_part, wnd_cell_count, n);
}

static void wn_build_nl(unsigned int n, double box_size) {
    const unsigned int threads = 256;
    const unsigned int blocks = (n + threads - 1) / threads;

    wn_build_cell_lists(n, blocks, threads, box_size);

    wn_sort_counter++;
    if ((wn_sort_counter % WN_SPATIAL_SORT_EVERY) == 0) {
        wnk_gather6<<<blocks, threads>>>(
            wnd_cell_part, wnd_x, wnd_y, wnd_z, wnd_vx, wnd_vy, wnd_vz,
            wnd_xs, wnd_ys, wnd_zs, wnd_vxs, wnd_vys, wnd_vzs, n);
        double *t;
        t = wnd_x;  wnd_x  = wnd_xs;  wnd_xs  = t;
        t = wnd_y;  wnd_y  = wnd_ys;  wnd_ys  = t;
        t = wnd_z;  wnd_z  = wnd_zs;  wnd_zs  = t;
        t = wnd_vx; wnd_vx = wnd_vxs; wnd_vxs = t;
        t = wnd_vy; wnd_vy = wnd_vys; wnd_vys = t;
        t = wnd_vz; wnd_vz = wnd_vzs; wnd_vzs = t;
        wn_build_cell_lists(n, blocks, threads, box_size);
    }

    CUDA_CHECK(cudaMemset(wnd_overflow, 0, sizeof(unsigned int)));
    // WARP per particle: total warps = n -> total threads = n*32
    const unsigned int wblocks = (n * WN_WARP + WN_BUILD_BLOCK - 1) / WN_BUILD_BLOCK;
    wnk_build_nl_cell<<<wblocks, WN_BUILD_BLOCK>>>(
        wnd_x, wnd_y, wnd_z, wnd_cell_id, wnd_cell_start, wnd_cell_count, wnd_cell_part,
        wnd_nl, wnd_nl_count, wnd_overflow, n, wn_max_neighbours,
        box_size, WN_R_LIST_SQ, wn_n_cells_side);

    unsigned int overflow = 0;
    CUDA_CHECK(cudaMemcpy(&overflow, wnd_overflow, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    if (overflow > 0) {
        fprintf(stderr, "[WARP-NL] OVERFLOW: particle %u exceeds max_neighbours=%u. Aborting.\n",
                overflow - 1u, wn_max_neighbours);
        exit(1);
    }

    CUDA_CHECK(cudaMemcpy(wnd_x_ref, wnd_x, n * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(wnd_y_ref, wnd_y, n * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(wnd_z_ref, wnd_z, n * sizeof(double), cudaMemcpyDeviceToDevice));
}

static int wn_needs_rebuild(unsigned int n, double box_size) {
    CUDA_CHECK(cudaMemsetAsync(wnd_flag, 0, sizeof(unsigned int)));
    const unsigned int threads = 128;
    const unsigned int blocks = (n + threads - 1) / threads;
    wnk_check_rebuild<<<blocks, threads, threads * sizeof(unsigned int)>>>(
        wnd_x, wnd_y, wnd_z, wnd_x_ref, wnd_y_ref, wnd_z_ref,
        n, box_size, WN_HALF_SKIN_SQ, wnd_flag);
    CUDA_CHECK(cudaMemcpy(wn_flag_pinned, wnd_flag, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    return (int)(*wn_flag_pinned);
}

static double wn_forces(unsigned int n, double box_size, int want_pe) {
    const unsigned int fblocks = (n * WN_WARP + WN_FORCE_BLOCK - 1) / WN_FORCE_BLOCK;
    wnk_forces<<<fblocks, WN_FORCE_BLOCK>>>(
        wnd_x, wnd_y, wnd_z, wnd_fx, wnd_fy, wnd_fz, wnd_pe_arr,
        wnd_nl, wnd_nl_count, n, wn_max_neighbours, box_size, WN_V_SHIFT);
    if (!want_pe) return 0.0;
    CUDA_CHECK(cudaMemsetAsync(wnd_pe_total, 0, sizeof(double)));
    const unsigned int rth = 256;
    wnk_reduce_pe<<<(n + rth - 1) / rth, rth, rth * sizeof(double)>>>(wnd_pe_arr, wnd_pe_total, n);
    double pe = 0.0;
    CUDA_CHECK(cudaMemcpy(&pe, wnd_pe_total, sizeof(double), cudaMemcpyDeviceToHost));
    return pe;
}

static double wn_leapfrog_step(unsigned int n, double box_size, int want_pe) {
    const unsigned int threads = 256;
    const unsigned int blocks = (n + threads - 1) / threads;

    wnk_kick_drift<<<blocks, threads>>>(
        wnd_x, wnd_y, wnd_z, wnd_vx, wnd_vy, wnd_vz, wnd_fx, wnd_fy, wnd_fz, n, box_size);
    CUDA_CHECK(cudaGetLastError());

    if (wn_needs_rebuild(n, box_size))
        wn_build_nl(n, box_size);

    const double pe = wn_forces(n, box_size, want_pe);

    wnk_kick<<<blocks, threads>>>(wnd_vx, wnd_vy, wnd_vz, wnd_fx, wnd_fy, wnd_fz, n);
    CUDA_CHECK(cudaGetLastError());
    return pe;
}

// =============================================================================
// PUBLIC ENTRY POINT
// =============================================================================

SimulationResult run_simulation_gpu_warpnl_3d(Particle *particles, unsigned int n,
                                              unsigned int nsteps, double box_size,
                                              int log_steps) {
    wn_alloc(n, box_size);
    wn_upload(particles, n);
    WN_V_SHIFT = compute_v_shift();
    wn_sort_counter = 0;

    wn_build_nl(n, box_size);

    SimulationResult out;
    out.start_potential = wn_forces(n, box_size, 1);
    out.start_kinetic   = compute_ke(particles, n);
    out.start_total     = out.start_kinetic + out.start_potential;
    out.final_potential = out.start_potential;

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));

    for (unsigned int step = 0; step < nsteps; ++step) {
        const int want_pe = log_steps || (step + 1u == nsteps);
        double pe = wn_leapfrog_step(n, box_size, want_pe);
        if (want_pe) out.final_potential = pe;
        if (log_steps) {
            CUDA_CHECK(cudaDeviceSynchronize());
            wn_download(particles, n);
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

    wn_download(particles, n);
    out.final_kinetic = compute_ke(particles, n);
    out.final_total   = out.final_kinetic + out.final_potential;
    out.n = n;
    out.particles = particles;

    printf("[WARP-NL] warp-per-particle neighbour-list build\n");
    printf("Simulation time %u steps: %.3f seconds\n", nsteps, ms / 1000.0f);
    return out;
}
