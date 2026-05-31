#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Include CUDA headers
#include <cuda_runtime.h>
#include <cuda.h>

#include "lennard-jones.h"

#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d %s\n",                           \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// =============================================================================
//  GRAPH VARIANT
// =============================================================================

#define GR_WARP               32u
#define GR_FORCE_BLOCK        256u
#define GR_SPATIAL_SORT_EVERY 4
#define GR_REBUILD_EVERY      8      // fixed rebuild period (steps)
#define GR_SKIN_SAFETY        1.5    // margin on the skin
#define GR_VEL_SIGMA          3.0    // skin sized off GR_VEL_SIGMA * v_rms
#define GR_NEIGH_SAFETY       4
#define GR_NEIGH_MIN          32

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


double random_double(void) {
    return (double)rand() / (double)RAND_MAX;
}

// compute kinetic energy of the system
double compute_ke(const Particle *particles, unsigned int n) {
    double ke = 0.0;
    for (unsigned int i = 0; i < n; ++i) {
        const Particle *p = &particles[i];
        ke += 0.5 * (p->vx * p->vx + p->vy * p->vy + p->vz * p->vz);
    }
    return ke;
}

int initialize_particles(Particle *particles, unsigned int n, double box_size, double placement_fraction, unsigned int seed, double temperature) {
    
    srand(seed);
    unsigned int n_side = (unsigned int)ceil(cbrt((double)n));
    double placement_size = placement_fraction * box_size;
    double offset = 0.5 * (box_size - placement_size);
    double delta = placement_size / (double)n_side;

    double mean_vx = 0.0;
    double mean_vy = 0.0;
    double mean_vz = 0.0;
    // place particles int he middle of the grid with some random jitter and assign random velocities
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

    mean_vx /= (double)n;
    mean_vy /= (double)n;
    mean_vz /= (double)n;
    double ke = 0.0;
    // subtract mean velocity to ensure zero net momentum and compute initial kinetic energy
    for (unsigned int k = 0; k < n; k++) {
        particles[k].vx -= mean_vx;
        particles[k].vy -= mean_vy;
        particles[k].vz -= mean_vz;
        ke += 0.5 * (
            particles[k].vx * particles[k].vx +
            particles[k].vy * particles[k].vy +
            particles[k].vz * particles[k].vz
        );
    }

    double current_temperature = ke / (double)n;
    if (current_temperature <= 0.0) {
        return 0;
    }

    // scale velocities to match the desired initial temperature of the system
    double scale = sqrt(temperature / current_temperature);
    for (unsigned int k = 0; k < n; k++) {
        particles[k].vx *= scale;
        particles[k].vy *= scale;
        particles[k].vz *= scale;
    }

    return 1;
}

// shift potential to ensure it goes to zero at the cutoff distance, improving energy conservation
double compute_v_shift(void) {
    return 4.0 * EPSILON * (pow(SIGMA / R_CUT, 12.0) - pow(SIGMA / R_CUT, 6.0));
}


// Assign each particle to its cell and count cell occupancy.
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

// Scatter particle indices into cell_part using the prefix-summed cell starts.
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

// Reorder positions+velocities by a permutation (the spatial sort).
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

// Build each particle's full Verlet list by scanning its 27 cells (one thread per particle).
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
            if (j == (int)i) continue;
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

// LJ forces + PE for each particle, one warp per particle, with a warp-shuffle reduction.
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

// Block reduction of per-particle PE into a single global total.
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

// Leapfrog half-kick + drift + periodic wrap.
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

// Leapfrog second half-kick (velocity update only).
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

// Allocate device buffers + cell grid (sized for the skin) and create the stream.
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

// Copy particle positions/velocities host -> device.
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

// Copy positions/velocities/forces device -> host.
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

// Bin particles into cells: count, exclusive-scan to starts, then fill.
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

// Rebuild cell + neighbour lists; spatial sort uses copy-back so graph buffer pointers stay fixed.
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

// Launch the force kernel; reduce + copy PE back only when requested.
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

// Capture the steady step (kick_drift -> force[no PE] -> kick) into a replayable graph.
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

// One step outside the graph: kick_drift, optional neighbour rebuild, force, kick.
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

// Graph variant entry: size skin from v_rms, fixed-period rebuild, steady step replayed via graph.
SimulationResult run_simulation_gpu_graph_3d(Particle *particles, unsigned int n,
                                             unsigned int nsteps, double box_size,
                                             int log_steps) {
    // Skin from RMS speed (N-independent), big enough to survive GR_REBUILD_EVERY steps.
    double v2sum = 0.0;
    for (unsigned int i = 0; i < n; ++i) {
        v2sum += particles[i].vx * particles[i].vx
               + particles[i].vy * particles[i].vy
               + particles[i].vz * particles[i].vz;
    }
    double v_rms  = (n > 0) ? sqrt(v2sum / (double)n) : 0.0;
    double v_char = GR_VEL_SIGMA * v_rms;
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

    cudaEvent_t t0, t1;                 // time the integration loop only
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
            CUDA_CHECK(cudaGraphLaunch(gr_exec, gr_stream));   // common path: one launch
        }
    }

    CUDA_CHECK(cudaEventRecord(t1, gr_stream));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));

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

// =============================================================================
//  WARPNL VARIANT  (drift-triggered rebuild; neighbour list built one warp/particle)
// =============================================================================

#define WN_WARP               32u
#define WN_FORCE_BLOCK        256u
#define WN_BUILD_BLOCK        256u   // 8 warps -> 8 particles per block
#define WN_SPATIAL_SORT_EVERY 4
#define WN_NEIGH_SAFETY       4
#define WN_NEIGH_MIN          32

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

// Analytical Verlet skin fraction as a function of N (same form as lennard-jones.cu).
static double wn_skin_frac(unsigned int n) {
    double f = 0.15 + 6.0 / sqrt((double)n);
    if (f < 0.10) f = 0.10;
    if (f > 0.60) f = 0.60;
    return f;
}

// Assign each particle to its cell and count cell occupancy.
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

// Scatter particle indices into cell_part using the prefix-summed cell starts.
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

// Reorder positions+velocities by a permutation (the spatial sort).
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

// Build each particle's neighbour list with one WARP per particle; warp-aggregated compaction.
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

        // Uniform loop bound (cnt) so all 32 lanes reach __ballot_sync together.
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

// LJ forces + PE for each particle, one warp per particle, with a warp-shuffle reduction.
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

// Block reduction of per-particle PE into a single global total.
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

// Leapfrog half-kick + drift + periodic wrap.
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

// Leapfrog second half-kick (velocity update only).
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

// Flag a rebuild if any particle has drifted more than half the skin since the last build.
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

// Allocate device buffers + cell grid sized for the analytical skin.
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

// Copy particle positions/velocities host -> device.
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

// Copy positions/velocities/forces device -> host.
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

// Bin particles into cells: count, exclusive-scan to starts, then fill.
static void wn_build_cell_lists(unsigned int n, unsigned int blocks, unsigned int threads,
                                double box_size) {
    CUDA_CHECK(cudaMemset(wnd_cell_count, 0, wn_n_cells * sizeof(int)));
    wnk_cell_assign<<<blocks, threads>>>(
        wnd_x, wnd_y, wnd_z, wnd_cell_id, wnd_cell_count, n, box_size, wn_n_cells_side);
    thrust::exclusive_scan(thrust::device, wnd_cell_count, wnd_cell_count + wn_n_cells, wnd_cell_start);
    CUDA_CHECK(cudaMemset(wnd_cell_count, 0, wn_n_cells * sizeof(int)));
    wnk_cell_fill<<<blocks, threads>>>(wnd_cell_id, wnd_cell_start, wnd_cell_part, wnd_cell_count, n);
}

// Rebuild cell + neighbour lists (pointer-swap sort) and refresh reference positions.
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

// Run the drift-check kernel and read back whether a rebuild is needed.
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

// Launch the force kernel; reduce + copy PE back only when requested.
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

// One drift-triggered leapfrog step: kick_drift, check/rebuild, force, kick.
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

// Warpnl variant entry: drift-triggered rebuild with a warp-built neighbour list.
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

    cudaEvent_t t0, t1;                 // time the integration loop only
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

// =============================================================================
//  DISPATCHER
// =============================================================================

#define AUTO_N_THRESHOLD 43000u

// Pick the faster variant by size: graph for N<=threshold, warpnl above.
SimulationResult run_simulation(Particle *particles, unsigned int n,
                                    unsigned int nsteps, double box_size,
                                    int log_steps) {
    if (n <= AUTO_N_THRESHOLD) {
        fprintf(stdout, "[AUTO] N=%u <= %u  ->  graph variant\n", n, AUTO_N_THRESHOLD);
        return run_simulation_gpu_graph_3d(particles, n, nsteps, box_size, log_steps);
    } else {
        fprintf(stdout, "[AUTO] N=%u >  %u  ->  warpnl variant\n", n, AUTO_N_THRESHOLD);
        return run_simulation_gpu_warpnl_3d(particles, n, nsteps, box_size, log_steps);
    }
}
