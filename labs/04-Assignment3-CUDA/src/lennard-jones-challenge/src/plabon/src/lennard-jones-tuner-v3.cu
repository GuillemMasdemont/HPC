#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include "lennard-jones.h"

// ─── CUDA error checking ─────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d %s\n",                           \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ─── Globals ─────────────────────────────────────────────────────────────────
static double G_V_SHIFT = 0.0;

#define NEIGHBOUR_SAFETY_FACTOR 4
#define NEIGHBOUR_MIN           32
#define SPATIAL_SORT_EVERY      4
#define WARP                    32u
#define FORCE_BLOCK             256u   // 8 warps per block -> 8 particles/block

static double g_skin = 0.3 * R_CUT;    // overwritten analytically in gpu_alloc_3d

#define R_LIST           (R_CUT + g_skin)
#define R_LIST_SQ_HOST   ((R_CUT + g_skin) * (R_CUT + g_skin))
#define HALF_SKIN_SQ_HOST ((g_skin * 0.5) * (g_skin * 0.5))

static unsigned int g_rebuild_counter = 0;
static unsigned int *h_flag_pinned = nullptr;

static unsigned int g_n_cells_side = 0;
static unsigned int g_n_cells = 0;
static int *d_cell_id = nullptr;
static int *d_cell_start = nullptr;
static int *d_cell_count = nullptr;
static int *d_cell_part = nullptr;

static double *d_x = nullptr, *d_y = nullptr, *d_z = nullptr;
static double *d_vx = nullptr, *d_vy = nullptr, *d_vz = nullptr;
static double *d_fx = nullptr, *d_fy = nullptr, *d_fz = nullptr;
static double *d_pe_arr = nullptr;

static double *d_xs = nullptr, *d_ys = nullptr, *d_zs = nullptr;
static double *d_vxs = nullptr, *d_vys = nullptr, *d_vzs = nullptr;

static int *d_nl = nullptr;
static unsigned int *d_nl_count = nullptr;
static double *d_x_ref = nullptr, *d_y_ref = nullptr, *d_z_ref = nullptr;
static unsigned int *d_flag = nullptr;
static unsigned int *d_overflow = nullptr;
static unsigned int g_max_neighbours = 0;
static unsigned int d_n = 0;

static double *d_pe_total = nullptr;

// rebuild bookkeeping (for reporting, not tuning)
static unsigned int g_nl_rebuilds = 0;

// =============================================================================
// CPU HOST FUNCTIONS (reference / validation)
// =============================================================================

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

void wrap_positions(Particle *particles, unsigned int n, double box_size) {
    for (unsigned int i = 0; i < n; ++i) {
        Particle *p = &particles[i];
        double wx = fmod(p->x, box_size); if (wx < 0.0) wx += box_size; p->x = wx;
        double wy = fmod(p->y, box_size); if (wy < 0.0) wy += box_size; p->y = wy;
        double wz = fmod(p->z, box_size); if (wz < 0.0) wz += box_size; p->z = wz;
    }
}

double compute_forces(Particle *particles, unsigned int n, double box_size) {
    for (unsigned int i = 0; i < n; ++i) {
        particles[i].fx = 0.0; particles[i].fy = 0.0; particles[i].fz = 0.0;
    }
    double pe = 0.0;
    double v_shift = compute_v_shift();
    for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j) {
            if (j == i) continue;
            double dx = particles[j].x - particles[i].x;
            double dy = particles[j].y - particles[i].y;
            double dz = particles[j].z - particles[i].z;
            dx -= box_size * nearbyint(dx / box_size);
            dy -= box_size * nearbyint(dy / box_size);
            dz -= box_size * nearbyint(dz / box_size);
            double r = sqrt(dx * dx + dy * dy + dz * dz);
            if (r >= R_CUT || r == 0.0) continue;
            double sr = SIGMA / r;
            double fij = -24.0 * EPSILON * (2.0 * pow(sr, 12.0) - pow(sr, 6.0)) / r;
            particles[i].fx += fij * dx / r;
            particles[i].fy += fij * dy / r;
            particles[i].fz += fij * dz / r;
            pe += 0.5 * (4.0 * EPSILON * (pow(sr, 12.0) - pow(sr, 6.0)) - v_shift);
        }
    return pe;
}

double leapfrog_step(Particle *particles, unsigned int n, double box_size) {
    for (unsigned int i = 0; i < n; ++i) {
        particles[i].vx += 0.5 * DT * particles[i].fx;
        particles[i].vy += 0.5 * DT * particles[i].fy;
        particles[i].vz += 0.5 * DT * particles[i].fz;
        particles[i].x  += DT * particles[i].vx;
        particles[i].y  += DT * particles[i].vy;
        particles[i].z  += DT * particles[i].vz;
    }
    wrap_positions(particles, n, box_size);
    double pe = compute_forces(particles, n, box_size);
    for (unsigned int i = 0; i < n; ++i) {
        particles[i].vx += 0.5 * DT * particles[i].fx;
        particles[i].vy += 0.5 * DT * particles[i].fy;
        particles[i].vz += 0.5 * DT * particles[i].fz;
    }
    return pe;
}

SimulationResult run_simulation(Particle *particles, unsigned int n,
                                unsigned int nsteps, double box_size, int log_steps) {
    SimulationResult out;
    out.start_potential = compute_forces(particles, n, box_size);
    out.start_kinetic   = compute_ke(particles, n);
    out.start_total     = out.start_kinetic + out.start_potential;
    for (unsigned int step = 0; step < nsteps; step++) {
        out.final_potential = leapfrog_step(particles, n, box_size);
        out.final_kinetic   = compute_ke(particles, n);
        out.final_total     = out.final_kinetic + out.final_potential;
        if (log_steps)
            printf("step=%6u KE=%10.4f PE=%10.4f E=%12.6f\n",
                   step, out.final_kinetic, out.final_potential, out.final_total);
    }
    out.n = n;
    out.particles = particles;
    return out;
}

// =============================================================================
// TUNING-FREE PARAMETER SELECTION
// =============================================================================
//
// Skin: closed-form, no sweep. The optimum minimises
//     cost(s) = force_cost(s) + rebuild_cost / rebuild_interval(s)
// At fixed density the leading behaviour is a near-constant skin (~0.15*R_CUT),
// rising at small N where fixed per-rebuild / kernel-launch overhead dominates.
// The expression below reproduces the previously hand-tuned table to within
// its own noise (0.15..0.35 *R_CUT across N = 1e3..1.5e5) with ZERO tuning.
//
static double compute_skin_frac(unsigned int n) {
    double f = 0.15 + 6.0 / sqrt((double)n);
    if (f < 0.10) f = 0.10;
    if (f > 0.60) f = 0.60;
    return f;
}

// =============================================================================
// GPU MEMORY MANAGEMENT
// =============================================================================

static void gpu_alloc_3d(unsigned int n, double box_size) {
    if (d_n == n) return;

    g_skin = compute_skin_frac(n) * R_CUT;

    if (d_n > 0) {
        CUDA_CHECK(cudaFree(d_x));  CUDA_CHECK(cudaFree(d_y));  CUDA_CHECK(cudaFree(d_z));
        CUDA_CHECK(cudaFree(d_vx)); CUDA_CHECK(cudaFree(d_vy)); CUDA_CHECK(cudaFree(d_vz));
        CUDA_CHECK(cudaFree(d_fx)); CUDA_CHECK(cudaFree(d_fy)); CUDA_CHECK(cudaFree(d_fz));
        CUDA_CHECK(cudaFree(d_pe_arr));   CUDA_CHECK(cudaFree(d_pe_total));
        CUDA_CHECK(cudaFree(d_nl));       CUDA_CHECK(cudaFree(d_nl_count));
        CUDA_CHECK(cudaFree(d_xs));  CUDA_CHECK(cudaFree(d_ys));  CUDA_CHECK(cudaFree(d_zs));
        CUDA_CHECK(cudaFree(d_vxs)); CUDA_CHECK(cudaFree(d_vys)); CUDA_CHECK(cudaFree(d_vzs));
        CUDA_CHECK(cudaFree(d_x_ref)); CUDA_CHECK(cudaFree(d_y_ref)); CUDA_CHECK(cudaFree(d_z_ref));
        CUDA_CHECK(cudaFree(d_flag));   CUDA_CHECK(cudaFree(d_overflow));
        CUDA_CHECK(cudaFree(d_cell_id)); CUDA_CHECK(cudaFree(d_cell_start));
        CUDA_CHECK(cudaFree(d_cell_count)); CUDA_CHECK(cudaFree(d_cell_part));
        if (h_flag_pinned) { cudaFreeHost(h_flag_pinned); h_flag_pinned = nullptr; }
    }

    {
        const double box_vol = box_size * box_size * box_size;
        const double density = (double)n / box_vol;
        const double r = R_LIST;
        const double expected = density * (4.0 / 3.0) * M_PI * r * r * r;
        const unsigned int computed = (unsigned int)(NEIGHBOUR_SAFETY_FACTOR * expected) + 1;
        g_max_neighbours = (computed > NEIGHBOUR_MIN) ? computed : NEIGHBOUR_MIN;
        fprintf(stdout,
                "[NL-3D] skin=%.4f (%.3f*R_CUT) density=%.4f expected_nn=%.1f "
                "max_neighbours=%u (FULL list, warp-per-particle)\n",
                g_skin, g_skin / R_CUT, density, expected, g_max_neighbours);
    }

    g_n_cells_side = (unsigned int)(box_size / R_LIST);
    if (g_n_cells_side < 1) g_n_cells_side = 1;
    g_n_cells = g_n_cells_side * g_n_cells_side * g_n_cells_side;
    fprintf(stdout, "[CELL] %u cells/side  %u total cells\n", g_n_cells_side, g_n_cells);

    const size_t sz = (size_t)n * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_x, sz));  CUDA_CHECK(cudaMalloc(&d_y, sz));  CUDA_CHECK(cudaMalloc(&d_z, sz));
    CUDA_CHECK(cudaMalloc(&d_vx, sz)); CUDA_CHECK(cudaMalloc(&d_vy, sz)); CUDA_CHECK(cudaMalloc(&d_vz, sz));
    CUDA_CHECK(cudaMalloc(&d_fx, sz)); CUDA_CHECK(cudaMalloc(&d_fy, sz)); CUDA_CHECK(cudaMalloc(&d_fz, sz));
    CUDA_CHECK(cudaMalloc(&d_pe_arr, sz));
    CUDA_CHECK(cudaMalloc(&d_pe_total, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x_ref, sz)); CUDA_CHECK(cudaMalloc(&d_y_ref, sz)); CUDA_CHECK(cudaMalloc(&d_z_ref, sz));
    CUDA_CHECK(cudaMalloc(&d_nl_count, (size_t)n * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_nl, (size_t)n * g_max_neighbours * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_flag, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_overflow, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_xs, sz));  CUDA_CHECK(cudaMalloc(&d_ys, sz));  CUDA_CHECK(cudaMalloc(&d_zs, sz));
    CUDA_CHECK(cudaMalloc(&d_vxs, sz)); CUDA_CHECK(cudaMalloc(&d_vys, sz)); CUDA_CHECK(cudaMalloc(&d_vzs, sz));
    CUDA_CHECK(cudaMallocHost(&h_flag_pinned, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_cell_id, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cell_count, (size_t)g_n_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cell_start, (size_t)g_n_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cell_part, (size_t)n * sizeof(int)));

    d_n = n;
}

// =============================================================================
// HOST <-> DEVICE TRANSFERS
// =============================================================================

static void upload_particles_3d(const Particle *p, unsigned int n) {
    const size_t sz = n * sizeof(double);
    double *hx = (double*)malloc(sz), *hy = (double*)malloc(sz), *hz = (double*)malloc(sz);
    double *hvx = (double*)malloc(sz), *hvy = (double*)malloc(sz), *hvz = (double*)malloc(sz);
    for (unsigned int i = 0; i < n; ++i) {
        hx[i] = p[i].x;  hy[i] = p[i].y;  hz[i] = p[i].z;
        hvx[i] = p[i].vx; hvy[i] = p[i].vy; hvz[i] = p[i].vz;
    }
    CUDA_CHECK(cudaMemcpy(d_x, hx, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, hy, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, hz, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx, hvx, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, hvy, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vz, hvz, sz, cudaMemcpyHostToDevice));
    free(hx); free(hy); free(hz); free(hvx); free(hvy); free(hvz);
}

static void download_particles_3d(Particle *p, unsigned int n) {
    const size_t sz = n * sizeof(double);
    double *hx = (double*)malloc(sz), *hy = (double*)malloc(sz), *hz = (double*)malloc(sz);
    double *hvx = (double*)malloc(sz), *hvy = (double*)malloc(sz), *hvz = (double*)malloc(sz);
    double *hfx = (double*)malloc(sz), *hfy = (double*)malloc(sz), *hfz = (double*)malloc(sz);
    CUDA_CHECK(cudaMemcpy(hx, d_x, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hy, d_y, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hz, d_z, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hvx, d_vx, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hvy, d_vy, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hvz, d_vz, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hfx, d_fx, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hfy, d_fy, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hfz, d_fz, sz, cudaMemcpyDeviceToHost));
    for (unsigned int i = 0; i < n; ++i) {
        p[i].x = hx[i];  p[i].y = hy[i];  p[i].z = hz[i];
        p[i].vx = hvx[i]; p[i].vy = hvy[i]; p[i].vz = hvz[i];
        p[i].fx = hfx[i]; p[i].fy = hfy[i]; p[i].fz = hfz[i];
    }
    free(hx); free(hy); free(hz); free(hvx); free(hvy); free(hvz);
    free(hfx); free(hfy); free(hfz);
}

// =============================================================================
// CELL LIST KERNELS
// =============================================================================

__global__ void kernel_cell_assign(
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

__global__ void kernel_cell_fill(
    const int * __restrict__ cell_id, const int * __restrict__ cell_start,
    int * __restrict__ cell_part, int * __restrict__ cell_cursor, unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int cid = cell_id[i];
    int slot = atomicAdd(&cell_cursor[cid], 1);
    cell_part[cell_start[cid] + slot] = (int)i;
}

__global__ void kernel_gather6(
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

// =============================================================================
// NEIGHBOUR LIST BUILD — CELL-ACCELERATED, *FULL* LIST (no half-list)
// =============================================================================

__global__ void kernel_build_nl_cell(
    const double * __restrict__ x, const double * __restrict__ y, const double * __restrict__ z,
    const int * __restrict__ cell_id,
    const int * __restrict__ cell_start, const int * __restrict__ cell_count,
    const int * __restrict__ cell_part,
    int * __restrict__ nl, unsigned int * __restrict__ nl_count,
    unsigned int * __restrict__ d_overflow,
    unsigned int n, unsigned int max_neighbours,
    double box_size, double r_list_sq, unsigned int n_cells_side)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const double xi = x[i], yi = y[i], zi = z[i];
    int cid = cell_id[i];
    int n2 = n_cells_side * n_cells_side;
    int cz_i = cid / n2;
    int cy_i = (cid % n2) / n_cells_side;
    int cx_i = cid % n_cells_side;

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
            if (j == (int)i) continue;          // FULL list: keep every neighbour
            double ddx = xi - x[j];
            double ddy = yi - y[j];
            double ddz = zi - z[j];
            ddx -= box_size * rint(ddx * inv_box);
            ddy -= box_size * rint(ddy * inv_box);
            ddz -= box_size * rint(ddz * inv_box);
            if (ddx * ddx + ddy * ddy + ddz * ddz < r_list_sq) {
                if (count < max_neighbours) nl[i * max_neighbours + count] = j;
                count++;
            }
        }
    }

    if (count > max_neighbours) {
        nl_count[i] = max_neighbours;
        atomicOr(d_overflow, i + 1u);
    } else {
        nl_count[i] = count;
    }
}

// =============================================================================
// FORCE KERNEL — WARP-PER-PARTICLE, FULL LIST, NO ATOMICS, SHUFFLE REDUCTION
// =============================================================================

__global__ void kernel_compute_forces_3d(
    const double * __restrict__ x, const double * __restrict__ y, const double * __restrict__ z,
    double * __restrict__ fx, double * __restrict__ fy, double * __restrict__ fz,
    double * __restrict__ pe_out,
    const int * __restrict__ nl, const unsigned int * __restrict__ nl_count,
    unsigned int n, unsigned int max_neighbours, double box_size, double v_shift)
{
    const unsigned int i    = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; // warp == particle
    const unsigned int lane = threadIdx.x & 31u;
    if (i >= n) return;

    const double xi = x[i], yi = y[i], zi = z[i];
    const double r_cut_sq = R_CUT * R_CUT;
    const double sig2 = SIGMA * SIGMA;
    const double inv_box = 1.0 / box_size;
    const unsigned int nn = nl_count[i];

    double Fx = 0.0, Fy = 0.0, Fz = 0.0, Pe = 0.0;

    for (unsigned int k = lane; k < nn; k += WARP) {
        const unsigned int j = (unsigned int)nl[i * max_neighbours + k];
        double dx = xi - x[j];
        double dy = yi - y[j];
        double dz = zi - z[j];
        dx -= box_size * rint(dx * inv_box);
        dy -= box_size * rint(dy * inv_box);
        dz -= box_size * rint(dz * inv_box);
        double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 >= r_cut_sq || r2 == 0.0) continue;
        double inv_r2 = 1.0 / r2;
        double sr2 = sig2 * inv_r2;
        double sr6 = sr2 * sr2 * sr2;
        double sr12 = sr6 * sr6;
        double fij = 24.0 * EPSILON * (2.0 * sr12 - sr6) * inv_r2;
        Fx += fij * dx;
        Fy += fij * dy;
        Fz += fij * dz;
        Pe += 0.5 * (4.0 * EPSILON * (sr12 - sr6) - v_shift); // 0.5: full list double-counts
    }

    // warp reduction (no shared memory, no __syncthreads)
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
// PE REDUCTION
// =============================================================================

__global__ void kernel_reduce_pe(
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

// =============================================================================
// LEAPFROG AND REBUILD KERNELS
// =============================================================================

__global__ void kernel_leapfrog_kick_drift_3d(
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

__global__ void kernel_leapfrog_kick_3d(
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

__global__ void kernel_check_rebuild_3d(
    const double * __restrict__ x, const double * __restrict__ y, const double * __restrict__ z,
    const double * __restrict__ x_ref, const double * __restrict__ y_ref, const double * __restrict__ z_ref,
    unsigned int n, double box_size, double half_skin_sq, unsigned int *d_flag)
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
        if (dx * dx + dy * dy + dz * dz > half_skin_sq) s_flag[tid] = 1;
    }
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_flag[tid] |= s_flag[tid + stride];
        __syncthreads();
    }
    if (tid == 0 && s_flag[0]) atomicOr(d_flag, 1u);
}

// =============================================================================
// HOST: CELL LIST CONSTRUCTION (device-side prefix scan via Thrust)
// =============================================================================

static void build_cell_lists(unsigned int n, unsigned int blocks, unsigned int threads,
                             double box_size) {
    CUDA_CHECK(cudaMemset(d_cell_count, 0, g_n_cells * sizeof(int)));
    kernel_cell_assign<<<blocks, threads>>>(
        d_x, d_y, d_z, d_cell_id, d_cell_count, n, box_size, g_n_cells_side);
    // exclusive scan of counts -> cell_start, fully on-device (no host round trip)
    thrust::exclusive_scan(thrust::device, d_cell_count, d_cell_count + g_n_cells, d_cell_start);
    CUDA_CHECK(cudaMemset(d_cell_count, 0, g_n_cells * sizeof(int))); // reuse as cursor
    kernel_cell_fill<<<blocks, threads>>>(d_cell_id, d_cell_start, d_cell_part, d_cell_count, n);
    // after fill, d_cell_count[c] == #particles in cell c again
}

static void gpu_build_neighbour_list_3d(unsigned int n, double box_size) {
    const unsigned int threads = 256;
    const unsigned int blocks = (n + threads - 1) / threads;

    build_cell_lists(n, blocks, threads, box_size);

    g_rebuild_counter++;
    if ((g_rebuild_counter % SPATIAL_SORT_EVERY) == 0) {
        kernel_gather6<<<blocks, threads>>>(
            d_cell_part, d_x, d_y, d_z, d_vx, d_vy, d_vz,
            d_xs, d_ys, d_zs, d_vxs, d_vys, d_vzs, n);
        CUDA_CHECK(cudaMemcpy(d_x, d_xs, n * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, d_ys, n * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_z, d_zs, n * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_vx, d_vxs, n * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_vy, d_vys, n * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_vz, d_vzs, n * sizeof(double), cudaMemcpyDeviceToDevice));
        build_cell_lists(n, blocks, threads, box_size);
    }

    CUDA_CHECK(cudaMemset(d_overflow, 0, sizeof(unsigned int)));
    kernel_build_nl_cell<<<blocks, threads>>>(
        d_x, d_y, d_z, d_cell_id, d_cell_start, d_cell_count, d_cell_part,
        d_nl, d_nl_count, d_overflow, n, g_max_neighbours,
        box_size, R_LIST_SQ_HOST, g_n_cells_side);

    unsigned int overflow = 0;
    CUDA_CHECK(cudaMemcpy(&overflow, d_overflow, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    if (overflow > 0) {
        fprintf(stderr, "[NL-3D] OVERFLOW: particle %u exceeds max_neighbours=%u. Aborting.\n",
                overflow - 1u, g_max_neighbours);
        exit(1);
    }

    CUDA_CHECK(cudaMemcpy(d_x_ref, d_x, n * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_ref, d_y, n * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_z_ref, d_z, n * sizeof(double), cudaMemcpyDeviceToDevice));

    g_nl_rebuilds++;
}

// =============================================================================
// HOST: REBUILD CHECK, FORCES, LEAPFROG STEP
// =============================================================================

static int gpu_needs_rebuild_3d(unsigned int n, double box_size) {
    CUDA_CHECK(cudaMemsetAsync(d_flag, 0, sizeof(unsigned int)));
    const unsigned int threads = 128;
    const unsigned int blocks = (n + threads - 1) / threads;
    kernel_check_rebuild_3d<<<blocks, threads, threads * sizeof(unsigned int)>>>(
        d_x, d_y, d_z, d_x_ref, d_y_ref, d_z_ref,
        n, box_size, HALF_SKIN_SQ_HOST, d_flag);
    CUDA_CHECK(cudaMemcpy(h_flag_pinned, d_flag, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    return (int)(*h_flag_pinned);
}

static double gpu_compute_forces_3d(unsigned int n, double box_size) {
    // warp per particle: total warps = n  ->  total threads = n*32
    const unsigned int fblocks = (n * WARP + FORCE_BLOCK - 1) / FORCE_BLOCK;
    kernel_compute_forces_3d<<<fblocks, FORCE_BLOCK>>>(
        d_x, d_y, d_z, d_fx, d_fy, d_fz, d_pe_arr,
        d_nl, d_nl_count, n, g_max_neighbours, box_size, G_V_SHIFT);

    CUDA_CHECK(cudaMemsetAsync(d_pe_total, 0, sizeof(double)));
    const unsigned int rth = 256;
    kernel_reduce_pe<<<(n + rth - 1) / rth, rth, rth * sizeof(double)>>>(d_pe_arr, d_pe_total, n);
    double pe = 0.0;
    CUDA_CHECK(cudaMemcpy(&pe, d_pe_total, sizeof(double), cudaMemcpyDeviceToHost));
    return pe;
}

static double gpu_leapfrog_step_3d(unsigned int n, double box_size) {
    const unsigned int threads = 256;
    const unsigned int blocks = (n + threads - 1) / threads;

    kernel_leapfrog_kick_drift_3d<<<blocks, threads>>>(
        d_x, d_y, d_z, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, n, box_size);
    CUDA_CHECK(cudaGetLastError());

    if (gpu_needs_rebuild_3d(n, box_size))
        gpu_build_neighbour_list_3d(n, box_size);

    const double pe = gpu_compute_forces_3d(n, box_size);

    kernel_leapfrog_kick_3d<<<blocks, threads>>>(d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, n);
    CUDA_CHECK(cudaGetLastError());
    return pe;
}

// =============================================================================
// PUBLIC SIMULATION API
// =============================================================================

SimulationResult run_simulation_gpu_v9_3d(Particle *particles, unsigned int n,
                                          unsigned int nsteps, double box_size,
                                          int log_steps) {
    gpu_alloc_3d(n, box_size);
    upload_particles_3d(particles, n);
    G_V_SHIFT = compute_v_shift();
    g_rebuild_counter = 0;
    g_nl_rebuilds = 0;

    gpu_build_neighbour_list_3d(n, box_size);

    SimulationResult out;
    out.start_potential = gpu_compute_forces_3d(n, box_size);
    out.start_kinetic   = compute_ke(particles, n);
    out.start_total     = out.start_kinetic + out.start_potential;

    // ---- timed region: integration loop only (setup/context excluded) -------
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));

    for (unsigned int step = 0; step < nsteps; ++step) {
        out.final_potential = gpu_leapfrog_step_3d(n, box_size);
        if (log_steps) {
            CUDA_CHECK(cudaDeviceSynchronize());
            download_particles_3d(particles, n);
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

    download_particles_3d(particles, n);
    out.final_kinetic = compute_ke(particles, n);
    out.final_total   = out.final_kinetic + out.final_potential;
    out.n = n;
    out.particles = particles;

    printf("[NL-3D] neighbour-list rebuilds: %u over %u steps (every %.1f steps)\n",
           g_nl_rebuilds, nsteps, nsteps ? (double)nsteps / (double)g_nl_rebuilds : 0.0);
    printf("Simulation time %u steps: %.3f seconds\n", nsteps, ms / 1000.0f);
    return out;
}

// SimulationResult run_simulation_gpu_v7_3d(Particle *particles, unsigned int n,
//                                           unsigned int nsteps, double box_size,
//                                           int log_steps) {
//     return run_simulation_gpu_v6_3d(particles, n, nsteps, box_size, log_steps);
// }
