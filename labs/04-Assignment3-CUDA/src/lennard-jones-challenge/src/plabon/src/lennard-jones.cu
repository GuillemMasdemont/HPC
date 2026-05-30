#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include <cuda_runtime.h>
#include <cuda.h>

#include "lennard-jones.h"

// ─── CUDA error checking ───────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ─── Globals ──────────────────────────────────────────────────────────────────
static double G_V_SHIFT = 0.0;

#define NEIGHBOUR_SAFETY_FACTOR  4
#define NEIGHBOUR_MIN            32
#define SPATIAL_SORT_EVERY       4

static double        g_skin       = 0.3 * R_CUT;
static unsigned int  g_block_size = 32;

#define R_LIST            (R_CUT + g_skin)
#define R_LIST_SQ_HOST    ((R_CUT + g_skin) * (R_CUT + g_skin))
#define HALF_SKIN_SQ_HOST ((g_skin * 0.5) * (g_skin * 0.5))

static double       *d_xs = nullptr, *d_ys = nullptr, *d_zs = nullptr;
static double       *d_vxs = nullptr, *d_vys = nullptr, *d_vzs = nullptr;
static unsigned int  g_rebuild_counter = 0;

static unsigned int *h_flag_pinned = nullptr;

static unsigned int  g_n_cells_side = 0;
static unsigned int  g_n_cells      = 0;
static int          *d_cell_id      = nullptr;
static int          *d_cell_start   = nullptr;
static int          *d_cell_count   = nullptr;
static int          *d_cell_part    = nullptr;

static double       *d_x  = nullptr, *d_y  = nullptr, *d_z  = nullptr;
static double       *d_vx = nullptr, *d_vy = nullptr, *d_vz = nullptr;
static double       *d_fx = nullptr, *d_fy = nullptr, *d_fz = nullptr;
static double       *d_pe_arr = nullptr;

static int          *d_nl       = nullptr;
static unsigned int *d_nl_count = nullptr;
static double       *d_x_ref   = nullptr, *d_y_ref = nullptr, *d_z_ref = nullptr;
static unsigned int *d_flag     = nullptr;
static unsigned int *d_overflow = nullptr;
static unsigned int  g_max_neighbours = 0;
static unsigned int  d_n = 0;

static double       *d_pe_total = nullptr;
static double       *d_ke_total = nullptr;

// =============================================================================
// CPU HOST FUNCTIONS
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
    unsigned int n_side   = (unsigned int)ceil(cbrt((double)n));
    double placement_size = placement_fraction * box_size;
    double offset         = 0.5 * (box_size - placement_size);
    double delta          = placement_size / (double)n_side;

    double mean_vx = 0.0, mean_vy = 0.0, mean_vz = 0.0;
    for (unsigned int k = 0; k < n; k++) {
        particles[k].id = k;
        unsigned int ix = k % n_side;
        unsigned int iy = (k / n_side) % n_side;
        unsigned int iz = k / (n_side * n_side);
        double x0 = offset + (0.5 + (double)ix) * delta;
        double y0 = offset + (0.5 + (double)iy) * delta;
        double z0 = offset + (0.5 + (double)iz) * delta;
        particles[k].x  = x0 + (2.0*random_double()-1.0)*JITTER*delta;
        particles[k].y  = y0 + (2.0*random_double()-1.0)*JITTER*delta;
        particles[k].z  = z0 + (2.0*random_double()-1.0)*JITTER*delta;
        particles[k].vx = 2.0*random_double()-1.0;
        particles[k].vy = 2.0*random_double()-1.0;
        particles[k].vz = 2.0*random_double()-1.0;
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
        ke += 0.5*(particles[k].vx*particles[k].vx +
                   particles[k].vy*particles[k].vy +
                   particles[k].vz*particles[k].vz);
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
            dx -= box_size*nearbyint(dx/box_size);
            dy -= box_size*nearbyint(dy/box_size);
            dz -= box_size*nearbyint(dz/box_size);
            double r = sqrt(dx*dx+dy*dy+dz*dz);
            if (r >= R_CUT || r == 0.0) continue;
            double sr  = SIGMA/r;
            double fij = -24.0*EPSILON*(2.0*pow(sr,12.0)-pow(sr,6.0))/r;
            particles[i].fx += fij*dx/r;
            particles[i].fy += fij*dy/r;
            particles[i].fz += fij*dz/r;
            pe += 0.5*(4.0*EPSILON*(pow(sr,12.0)-pow(sr,6.0))-v_shift);
        }
    return pe;
}

double leapfrog_step(Particle *particles, unsigned int n, double box_size) {
    for (unsigned int i = 0; i < n; ++i) {
        particles[i].vx += 0.5*DT*particles[i].fx;
        particles[i].vy += 0.5*DT*particles[i].fy;
        particles[i].vz += 0.5*DT*particles[i].fz;
        particles[i].x  += DT*particles[i].vx;
        particles[i].y  += DT*particles[i].vy;
        particles[i].z  += DT*particles[i].vz;
    }
    wrap_positions(particles, n, box_size);
    double pe = compute_forces(particles, n, box_size);
    for (unsigned int i = 0; i < n; ++i) {
        particles[i].vx += 0.5*DT*particles[i].fx;
        particles[i].vy += 0.5*DT*particles[i].fy;
        particles[i].vz += 0.5*DT*particles[i].fz;
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
            printf("step=%6u  KE=%10.4f  PE=%10.4f  E=%12.6f\n",
                   step, out.final_kinetic, out.final_potential, out.final_total);
    }
    out.n = n; out.particles = particles;
    return out;
}

// =============================================================================
// PRETUNE LOOKUP TABLE
// =============================================================================
// Generated by tune_sweep_and_print. Paste updated entries here and rebuild.
// When LJ_PRETUNE_COUNT is 0 the analytical default (g_skin=0.3*R_CUT) is used.
#define LJ_PRETUNE_COUNT 150
static const struct { unsigned int n; float skin_frac; unsigned int block; } LJ_PRETUNE[] = {
    {   1000, 0.35f,  64 },  // 0.048 ms/step
    {   2000, 0.35f,  32 },  // 0.055 ms/step
    {   3000, 0.50f,  32 },  // 0.057 ms/step
    {   4000, 0.40f,  32 },  // 0.056 ms/step
    {   5000, 0.50f,  32 },  // 0.062 ms/step
    {   6000, 0.40f,  32 },  // 0.065 ms/step
    {   7000, 0.40f,  32 },  // 0.067 ms/step
    {   8000, 0.45f,  32 },  // 0.067 ms/step
    {   9000, 0.25f,  32 },  // 0.070 ms/step
    {  10000, 0.40f,  32 },  // 0.076 ms/step
    {  11000, 0.25f,  32 },  // 0.079 ms/step
    {  12000, 0.15f,  32 },  // 0.076 ms/step
    {  13000, 0.15f,  32 },  // 0.082 ms/step
    {  14000, 0.20f,  32 },  // 0.086 ms/step
    {  15000, 0.20f,  32 },  // 0.083 ms/step
    {  16000, 0.15f,  32 },  // 0.090 ms/step
    {  17000, 0.20f,  32 },  // 0.088 ms/step
    {  18000, 0.25f,  32 },  // 0.099 ms/step
    {  19000, 0.10f,  32 },  // 0.096 ms/step
    {  20000, 0.20f,  32 },  // 0.102 ms/step
    {  21000, 0.20f,  32 },  // 0.100 ms/step
    {  22000, 0.15f,  32 },  // 0.108 ms/step
    {  23000, 0.15f,  32 },  // 0.110 ms/step
    {  24000, 0.20f,  32 },  // 0.106 ms/step
    {  25000, 0.25f,  32 },  // 0.118 ms/step
    {  26000, 0.15f,  32 },  // 0.113 ms/step
    {  27000, 0.15f,  32 },  // 0.118 ms/step
    {  28000, 0.15f,  32 },  // 0.129 ms/step
    {  29000, 0.15f,  32 },  // 0.124 ms/step
    {  30000, 0.20f,  32 },  // 0.129 ms/step
    {  31000, 0.15f,  32 },  // 0.131 ms/step
    {  32000, 0.20f,  32 },  // 0.125 ms/step
    {  33000, 0.15f,  32 },  // 0.136 ms/step
    {  34000, 0.15f,  32 },  // 0.142 ms/step
    {  35000, 0.15f,  32 },  // 0.136 ms/step
    {  36000, 0.10f,  32 },  // 0.151 ms/step
    {  37000, 0.20f,  32 },  // 0.154 ms/step
    {  38000, 0.15f,  32 },  // 0.148 ms/step
    {  39000, 0.15f,  32 },  // 0.148 ms/step
    {  40000, 0.15f,  32 },  // 0.155 ms/step
    {  41000, 0.15f,  32 },  // 0.149 ms/step
    {  42000, 0.15f,  32 },  // 0.153 ms/step
    {  43000, 0.15f,  32 },  // 0.161 ms/step
    {  44000, 0.15f,  32 },  // 0.163 ms/step
    {  45000, 0.20f,  32 },  // 0.160 ms/step
    {  46000, 0.15f,  32 },  // 0.164 ms/step
    {  47000, 0.15f,  32 },  // 0.178 ms/step
    {  48000, 0.15f,  32 },  // 0.179 ms/step
    {  49000, 0.15f,  32 },  // 0.169 ms/step
    {  50000, 0.15f,  32 },  // 0.172 ms/step
    {  51000, 0.15f,  32 },  // 0.189 ms/step
    {  52000, 0.15f,  32 },  // 0.191 ms/step
    {  53000, 0.15f,  32 },  // 0.181 ms/step
    {  54000, 0.15f,  32 },  // 0.185 ms/step
    {  55000, 0.15f,  32 },  // 0.198 ms/step
    {  56000, 0.15f,  32 },  // 0.197 ms/step
    {  57000, 0.15f,  32 },  // 0.195 ms/step
    {  58000, 0.15f,  32 },  // 0.196 ms/step
    {  59000, 0.15f,  32 },  // 0.198 ms/step
    {  60000, 0.15f,  32 },  // 0.206 ms/step
    {  61000, 0.15f,  32 },  // 0.201 ms/step
    {  62000, 0.15f,  32 },  // 0.203 ms/step
    {  63000, 0.15f,  32 },  // 0.208 ms/step
    {  64000, 0.15f,  32 },  // 0.210 ms/step
    {  65000, 0.15f,  32 },  // 0.224 ms/step
    {  66000, 0.15f,  32 },  // 0.220 ms/step
    {  67000, 0.15f,  32 },  // 0.222 ms/step
    {  68000, 0.15f,  32 },  // 0.224 ms/step
    {  69000, 0.15f,  32 },  // 0.237 ms/step
    {  70000, 0.15f,  32 },  // 0.243 ms/step
    {  71000, 0.15f,  32 },  // 0.229 ms/step
    {  72000, 0.15f,  32 },  // 0.236 ms/step
    {  73000, 0.20f,  32 },  // 0.236 ms/step
    {  74000, 0.15f,  32 },  // 0.242 ms/step
    {  75000, 0.15f,  32 },  // 0.252 ms/step
    {  76000, 0.15f,  32 },  // 0.247 ms/step
    {  77000, 0.15f,  32 },  // 0.247 ms/step
    {  78000, 0.15f,  32 },  // 0.247 ms/step
    {  79000, 0.15f,  32 },  // 0.255 ms/step
    {  80000, 0.15f,  32 },  // 0.271 ms/step
    {  81000, 0.15f,  32 },  // 0.261 ms/step
    {  82000, 0.15f,  32 },  // 0.257 ms/step
    {  83000, 0.15f,  32 },  // 0.268 ms/step
    {  84000, 0.15f,  32 },  // 0.266 ms/step
    {  85000, 0.15f,  32 },  // 0.275 ms/step
    {  86000, 0.10f,  32 },  // 0.301 ms/step
    {  87000, 0.15f,  32 },  // 0.287 ms/step
    {  88000, 0.15f,  32 },  // 0.281 ms/step
    {  89000, 0.15f,  32 },  // 0.287 ms/step
    {  90000, 0.15f,  32 },  // 0.291 ms/step
    {  91000, 0.15f,  32 },  // 0.297 ms/step
    {  92000, 0.15f,  32 },  // 0.310 ms/step
    {  93000, 0.15f,  32 },  // 0.300 ms/step
    {  94000, 0.15f,  32 },  // 0.300 ms/step
    {  95000, 0.15f,  32 },  // 0.308 ms/step
    {  96000, 0.15f,  32 },  // 0.310 ms/step
    {  97000, 0.15f,  32 },  // 0.316 ms/step
    {  98000, 0.15f,  32 },  // 0.335 ms/step
    {  99000, 0.20f,  32 },  // 0.325 ms/step
    { 100000, 0.20f,  32 },  // 0.325 ms/step
    { 101000, 0.20f,  32 },  // 0.332 ms/step
    { 102000, 0.20f,  32 },  // 0.330 ms/step
    { 103000, 0.20f,  32 },  // 0.335 ms/step
    { 104000, 0.15f,  32 },  // 0.355 ms/step
    { 105000, 0.15f,  32 },  // 0.354 ms/step
    { 106000, 0.15f,  32 },  // 0.341 ms/step
    { 107000, 0.15f,  32 },  // 0.346 ms/step
    { 108000, 0.15f,  32 },  // 0.348 ms/step
    { 109000, 0.15f,  32 },  // 0.354 ms/step
    { 110000, 0.15f,  32 },  // 0.355 ms/step
    { 111000, 0.15f,  32 },  // 0.390 ms/step
    { 112000, 0.15f,  32 },  // 0.366 ms/step
    { 113000, 0.15f,  32 },  // 0.370 ms/step
    { 114000, 0.15f,  32 },  // 0.373 ms/step
    { 115000, 0.15f,  32 },  // 0.377 ms/step
    { 116000, 0.15f,  32 },  // 0.380 ms/step
    { 117000, 0.15f,  32 },  // 0.385 ms/step
    { 118000, 0.15f,  32 },  // 0.408 ms/step
    { 119000, 0.15f,  32 },  // 0.388 ms/step
    { 120000, 0.15f,  32 },  // 0.396 ms/step
    { 121000, 0.15f,  32 },  // 0.393 ms/step
    { 122000, 0.15f,  32 },  // 0.401 ms/step
    { 123000, 0.15f,  32 },  // 0.398 ms/step
    { 124000, 0.15f,  32 },  // 0.407 ms/step
    { 125000, 0.15f,  32 },  // 0.407 ms/step
    { 126000, 0.15f,  32 },  // 0.432 ms/step
    { 127000, 0.15f,  32 },  // 0.415 ms/step
    { 128000, 0.15f,  32 },  // 0.418 ms/step
    { 129000, 0.15f,  32 },  // 0.421 ms/step
    { 130000, 0.15f,  32 },  // 0.424 ms/step
    { 131000, 0.15f,  32 },  // 0.427 ms/step
    { 132000, 0.15f,  32 },  // 0.431 ms/step
    { 133000, 0.15f,  32 },  // 0.453 ms/step
    { 134000, 0.15f,  32 },  // 0.438 ms/step
    { 135000, 0.15f,  32 },  // 0.443 ms/step
    { 136000, 0.15f,  32 },  // 0.444 ms/step
    { 137000, 0.15f,  32 },  // 0.446 ms/step
    { 138000, 0.15f,  32 },  // 0.450 ms/step
    { 139000, 0.15f,  32 },  // 0.454 ms/step
    { 140000, 0.15f,  32 },  // 0.456 ms/step
    { 141000, 0.15f,  32 },  // 0.486 ms/step
    { 142000, 0.15f,  32 },  // 0.462 ms/step
    { 143000, 0.15f,  32 },  // 0.468 ms/step
    { 144000, 0.15f,  32 },  // 0.470 ms/step
    { 145000, 0.15f,  32 },  // 0.473 ms/step
    { 146000, 0.15f,  32 },  // 0.475 ms/step
    { 147000, 0.15f,  32 },  // 0.477 ms/step
    { 148000, 0.15f,  32 },  // 0.481 ms/step
    { 149000, 0.15f,  32 },  // 0.514 ms/step
    { 150000, 0.15f,  32 },  // 0.488 ms/step
};

// Applies nearest-N entry from LJ_PRETUNE to set g_skin and g_block_size.
// Returns 1 if applied, 0 if table is empty (falls back to analytical default).
static int apply_pretune_3d(unsigned int n) {
    if (LJ_PRETUNE_COUNT == 0) return 0;
    unsigned int best      = 0;
    unsigned int best_diff = (n >= LJ_PRETUNE[0].n) ? n - LJ_PRETUNE[0].n
                                                      : LJ_PRETUNE[0].n - n;
    for (unsigned int i = 1; i < LJ_PRETUNE_COUNT; ++i) {
        unsigned int diff = (n >= LJ_PRETUNE[i].n) ? n - LJ_PRETUNE[i].n
                                                    : LJ_PRETUNE[i].n - n;
        if (diff < best_diff) { best_diff = diff; best = i; }
    }
    g_skin       = (double)LJ_PRETUNE[best].skin_frac * R_CUT;
    g_block_size = LJ_PRETUNE[best].block;
    fprintf(stdout, "[NL-3D] pretune N=%u (table N=%u): skin=%.4f (%.2f*R_CUT)  block=%u\n",
            n, LJ_PRETUNE[best].n, g_skin, (double)LJ_PRETUNE[best].skin_frac, g_block_size);
    return 1;
}

// =============================================================================
// GPU MEMORY MANAGEMENT
// =============================================================================

// Analytical block size fallback — only used when pretune table is empty.
static void configure_block_size_3d(unsigned int n, double box_size) {
    const double box_vol = box_size * box_size * box_size;
    const double density = (double)n / box_vol;
    const double r       = R_LIST;
    const double mean_nn = density * (4.0/3.0) * M_PI * r * r * r;
    const double target  = 1.4 * mean_nn + 4.0;
    unsigned int bs = 32;
    while ((double)bs < target && bs < 256) bs <<= 1;
    g_block_size = bs;
}

static void gpu_alloc_3d(unsigned int n, double box_size) {
    if (d_n == n) return;
    const int had_pretune = apply_pretune_3d(n);
    // ── ADD THIS ──────────────────────────────────────────────────────────
    fprintf(stdout, "[ALLOC] n=%u  had_pretune=%d  g_skin=%.6f (%.3f*R_CUT)  "
            "g_block_size=%u\n",
            n, had_pretune, g_skin, g_skin/R_CUT, g_block_size);
    // ─────────────────────────────────────────────────────────────────────
    if (d_n > 0) {
        CUDA_CHECK(cudaFree(d_x));  CUDA_CHECK(cudaFree(d_y));  CUDA_CHECK(cudaFree(d_z));
        CUDA_CHECK(cudaFree(d_vx)); CUDA_CHECK(cudaFree(d_vy)); CUDA_CHECK(cudaFree(d_vz));
        CUDA_CHECK(cudaFree(d_fx)); CUDA_CHECK(cudaFree(d_fy)); CUDA_CHECK(cudaFree(d_fz));
        CUDA_CHECK(cudaFree(d_pe_arr)); CUDA_CHECK(cudaFree(d_pe_total));
        if (d_ke_total) { CUDA_CHECK(cudaFree(d_ke_total)); d_ke_total = nullptr; }
        if (d_nl)       { CUDA_CHECK(cudaFree(d_nl));       d_nl       = nullptr; }
        CUDA_CHECK(cudaFree(d_xs));  CUDA_CHECK(cudaFree(d_ys));  CUDA_CHECK(cudaFree(d_zs));
        CUDA_CHECK(cudaFree(d_vxs)); CUDA_CHECK(cudaFree(d_vys)); CUDA_CHECK(cudaFree(d_vzs));
        if (h_flag_pinned) { cudaFreeHost(h_flag_pinned); h_flag_pinned = nullptr; }
        CUDA_CHECK(cudaFree(d_nl_count));
        CUDA_CHECK(cudaFree(d_x_ref)); CUDA_CHECK(cudaFree(d_y_ref)); CUDA_CHECK(cudaFree(d_z_ref));
        CUDA_CHECK(cudaFree(d_flag));  CUDA_CHECK(cudaFree(d_overflow));
        if (d_cell_id)    { CUDA_CHECK(cudaFree(d_cell_id));    d_cell_id    = nullptr; }
        if (d_cell_start) { CUDA_CHECK(cudaFree(d_cell_start)); d_cell_start = nullptr; }
        if (d_cell_count) { CUDA_CHECK(cudaFree(d_cell_count)); d_cell_count = nullptr; }
        if (d_cell_part)  { CUDA_CHECK(cudaFree(d_cell_part));  d_cell_part  = nullptr; }
    }

    {
        const double box_vol  = box_size * box_size * box_size;
        const double density  = (double)n / box_vol;
        const double r        = R_LIST;
        const double expected = density * (4.0/3.0) * M_PI * r * r * r;
        const unsigned int computed = (unsigned int)(NEIGHBOUR_SAFETY_FACTOR * expected) + 1;
        g_max_neighbours = (computed > NEIGHBOUR_MIN) ? computed : NEIGHBOUR_MIN;
        if (!had_pretune) configure_block_size_3d(n, box_size);
        fprintf(stdout, "[NL-3D] density=%.4f  expected_nn=%.1f  max_neighbours=%u  block=%u\n",
                density, expected, g_max_neighbours, g_block_size);
    }

    g_n_cells_side = (unsigned int)(box_size / R_LIST);
    if (g_n_cells_side < 1) g_n_cells_side = 1;
    g_n_cells = g_n_cells_side * g_n_cells_side * g_n_cells_side;
    fprintf(stdout, "[CELL] %u cells/side  %u total cells\n", g_n_cells_side, g_n_cells);

    const size_t sz = (size_t)n * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_x,  sz)); CUDA_CHECK(cudaMalloc(&d_y,  sz)); CUDA_CHECK(cudaMalloc(&d_z,  sz));
    CUDA_CHECK(cudaMalloc(&d_vx, sz)); CUDA_CHECK(cudaMalloc(&d_vy, sz)); CUDA_CHECK(cudaMalloc(&d_vz, sz));
    CUDA_CHECK(cudaMalloc(&d_fx, sz)); CUDA_CHECK(cudaMalloc(&d_fy, sz)); CUDA_CHECK(cudaMalloc(&d_fz, sz));
    CUDA_CHECK(cudaMalloc(&d_pe_arr,   sz));
    CUDA_CHECK(cudaMalloc(&d_pe_total, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ke_total, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x_ref, sz)); CUDA_CHECK(cudaMalloc(&d_y_ref, sz)); CUDA_CHECK(cudaMalloc(&d_z_ref, sz));
    CUDA_CHECK(cudaMalloc(&d_nl_count, (size_t)n * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_nl,       (size_t)n * g_max_neighbours * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_flag,     sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_overflow, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_xs,  sz)); CUDA_CHECK(cudaMalloc(&d_ys,  sz)); CUDA_CHECK(cudaMalloc(&d_zs,  sz));
    CUDA_CHECK(cudaMalloc(&d_vxs, sz)); CUDA_CHECK(cudaMalloc(&d_vys, sz)); CUDA_CHECK(cudaMalloc(&d_vzs, sz));
    CUDA_CHECK(cudaMallocHost(&h_flag_pinned, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_cell_id,    (size_t)n         * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cell_count, (size_t)g_n_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cell_start, (size_t)g_n_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cell_part,  (size_t)n         * sizeof(int)));
    d_n = n;
}

// =============================================================================
// HOST <-> DEVICE TRANSFERS
// =============================================================================

static void upload_particles_3d(const Particle *p, unsigned int n) {
    const size_t sz = n * sizeof(double);
    double *hx  = (double*)malloc(sz), *hy  = (double*)malloc(sz), *hz  = (double*)malloc(sz);
    double *hvx = (double*)malloc(sz), *hvy = (double*)malloc(sz), *hvz = (double*)malloc(sz);
    for (unsigned int i = 0; i < n; ++i) {
        hx[i]=p[i].x;   hy[i]=p[i].y;   hz[i]=p[i].z;
        hvx[i]=p[i].vx; hvy[i]=p[i].vy; hvz[i]=p[i].vz;
    }
    CUDA_CHECK(cudaMemcpy(d_x,  hx,  sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y,  hy,  sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z,  hz,  sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx, hvx, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, hvy, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vz, hvz, sz, cudaMemcpyHostToDevice));
    free(hx); free(hy); free(hz); free(hvx); free(hvy); free(hvz);
}

static void download_particles_3d(Particle *p, unsigned int n) {
    const size_t sz = n * sizeof(double);
    double *hx  = (double*)malloc(sz), *hy  = (double*)malloc(sz), *hz  = (double*)malloc(sz);
    double *hvx = (double*)malloc(sz), *hvy = (double*)malloc(sz), *hvz = (double*)malloc(sz);
    double *hfx = (double*)malloc(sz), *hfy = (double*)malloc(sz), *hfz = (double*)malloc(sz);
    CUDA_CHECK(cudaMemcpy(hx,  d_x,  sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hy,  d_y,  sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hz,  d_z,  sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hvx, d_vx, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hvy, d_vy, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hvz, d_vz, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hfx, d_fx, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hfy, d_fy, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hfz, d_fz, sz, cudaMemcpyDeviceToHost));
    for (unsigned int i = 0; i < n; ++i) {
        p[i].x=hx[i];   p[i].y=hy[i];   p[i].z=hz[i];
        p[i].vx=hvx[i]; p[i].vy=hvy[i]; p[i].vz=hvz[i];
        p[i].fx=hfx[i]; p[i].fy=hfy[i]; p[i].fz=hfz[i];
    }
    free(hx); free(hy); free(hz); free(hvx); free(hvy); free(hvz);
    free(hfx); free(hfy); free(hfz);
}

// =============================================================================
// CELL LIST KERNELS
// =============================================================================

__global__ void kernel_cell_assign(
    const double * __restrict__ x,
    const double * __restrict__ y,
    const double * __restrict__ z,
    int          * __restrict__ cell_id,
    int          * __restrict__ cell_count,
    unsigned int n, double box_size, unsigned int n_cells_side)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double cell_size = box_size / (double)n_cells_side;
    int cx = (int)(x[i] / cell_size); if (cx >= (int)n_cells_side) cx = n_cells_side-1; if (cx < 0) cx = 0;
    int cy = (int)(y[i] / cell_size); if (cy >= (int)n_cells_side) cy = n_cells_side-1; if (cy < 0) cy = 0;
    int cz = (int)(z[i] / cell_size); if (cz >= (int)n_cells_side) cz = n_cells_side-1; if (cz < 0) cz = 0;
    int cid = cx + cy * n_cells_side + cz * n_cells_side * n_cells_side;
    cell_id[i] = cid;
    atomicAdd(&cell_count[cid], 1);
}

__global__ void kernel_cell_fill(
    const int * __restrict__ cell_id,
    const int * __restrict__ cell_start,
    int       * __restrict__ cell_part,
    int       * __restrict__ cell_cursor,
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int cid  = cell_id[i];
    int slot = atomicAdd(&cell_cursor[cid], 1);
    cell_part[cell_start[cid] + slot] = (int)i;
}

// =============================================================================
// NEIGHBOUR LIST BUILD — CELL LIST ACCELERATED, HALF LIST
// =============================================================================

__global__ void kernel_build_nl_cell(
    const double * __restrict__ x,
    const double * __restrict__ y,
    const double * __restrict__ z,
    const int    * __restrict__ cell_id,
    const int    * __restrict__ cell_start,
    const int    * __restrict__ cell_count,
    const int    * __restrict__ cell_part,
    int          * __restrict__ nl,
    unsigned int * __restrict__ nl_count,
    unsigned int * __restrict__ d_overflow,
    unsigned int n, unsigned int max_neighbours,
    double box_size, double r_list_sq,
    unsigned int n_cells_side)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const double xi = x[i], yi = y[i], zi = z[i];
    int cid  = cell_id[i];
    int n2   = n_cells_side * n_cells_side;
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

        int ncid  = nx + ny * n_cells_side + nz * n2;
        int start = cell_start[ncid];
        int cnt   = cell_count[ncid];

        for (int s = 0; s < cnt; s++) {
            int j = cell_part[start + s];
            if (j <= (int)i) continue;

            double ddx = xi - x[j];
            double ddy = yi - y[j];
            double ddz = zi - z[j];
            ddx -= box_size * rint(ddx * inv_box);
            ddy -= box_size * rint(ddy * inv_box);
            ddz -= box_size * rint(ddz * inv_box);

            if (ddx*ddx + ddy*ddy + ddz*ddz < r_list_sq) {
                if (count < max_neighbours)
                    nl[i * max_neighbours + count] = j;
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
// FORCE KERNEL — half list + Newton's 3rd law via atomicAdd
// =============================================================================

__global__ void kernel_zero_forces(
    double *fx, double *fy, double *fz, double *pe, unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fx[i] = 0.0; fy[i] = 0.0; fz[i] = 0.0; pe[i] = 0.0;
}

__global__ void kernel_compute_forces_3d(
    const double * __restrict__ x,
    const double * __restrict__ y,
    const double * __restrict__ z,
    double       * __restrict__ fx,
    double       * __restrict__ fy,
    double       * __restrict__ fz,
    double       * __restrict__ pe_out,
    const int    * __restrict__ nl,
    const unsigned int * __restrict__ nl_count,
    unsigned int n, unsigned int max_neighbours,
    double box_size, double v_shift)
{
    extern __shared__ double s_shared[];
    const unsigned int bs = blockDim.x;
    double *s_fx = s_shared;
    double *s_fy = s_shared +     bs;
    double *s_fz = s_shared + 2 * bs;
    double *s_pe = s_shared + 3 * bs;

    const unsigned int i   = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (i >= n) return;

    const double xi       = x[i], yi = y[i], zi = z[i];
    const double r_cut_sq = R_CUT * R_CUT;
    const double sig2     = SIGMA * SIGMA;
    const unsigned int nn = nl_count[i];
    const double inv_box  = 1.0 / box_size;

    double lfx = 0.0, lfy = 0.0, lfz = 0.0, lpe = 0.0;

    for (unsigned int k = tid; k < nn; k += bs) {
        const unsigned int j = (unsigned int)nl[i * max_neighbours + k];

        double dx = xi - x[j];
        double dy = yi - y[j];
        double dz = zi - z[j];
        dx -= box_size * rint(dx * inv_box);
        dy -= box_size * rint(dy * inv_box);
        dz -= box_size * rint(dz * inv_box);

        double r2 = dx*dx + dy*dy + dz*dz;
        if (r2 >= r_cut_sq || r2 == 0.0) continue;

        double inv_r2 = 1.0 / r2;
        double sr2    = sig2 * inv_r2;
        double sr6    = sr2 * sr2 * sr2;
        double sr12   = sr6 * sr6;
        double fij    = 24.0 * EPSILON * (2.0*sr12 - sr6) * inv_r2;
        double fxij   = fij * dx;
        double fyij   = fij * dy;
        double fzij   = fij * dz;

        lfx += fxij;
        lfy += fyij;
        lfz += fzij;
        lpe += 4.0*EPSILON*(sr12 - sr6) - v_shift;

        atomicAdd(&fx[j], -fxij);
        atomicAdd(&fy[j], -fyij);
        atomicAdd(&fz[j], -fzij);
    }

    s_fx[tid] = lfx; s_fy[tid] = lfy; s_fz[tid] = lfz; s_pe[tid] = lpe;
    __syncthreads();

    for (unsigned int stride = bs / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_fx[tid] += s_fx[tid + stride];
            s_fy[tid] += s_fy[tid + stride];
            s_fz[tid] += s_fz[tid + stride];
            s_pe[tid] += s_pe[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&fx[i], s_fx[0]);
        atomicAdd(&fy[i], s_fy[0]);
        atomicAdd(&fz[i], s_fz[0]);
        pe_out[i] = s_pe[0];
    }
}

// =============================================================================
// GPU-SIDE PE AND KE REDUCTIONS
// =============================================================================

__global__ void kernel_reduce_pe(
    const double * __restrict__ pe_arr,
    double       * __restrict__ pe_total,
    unsigned int n)
{
    extern __shared__ double s_pe[];
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + tid;
    s_pe[tid] = (i < n) ? pe_arr[i] : 0.0;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_pe[tid] += s_pe[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(pe_total, s_pe[0]);
}

__global__ void kernel_reduce_ke(
    const double * __restrict__ vx,
    const double * __restrict__ vy,
    const double * __restrict__ vz,
    double       * __restrict__ ke_total,
    unsigned int n)
{
    extern __shared__ double s_ke[];
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + tid;
    s_ke[tid] = (i < n) ? 0.5*(vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]) : 0.0;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_ke[tid] += s_ke[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(ke_total, s_ke[0]);
}

// =============================================================================
// LEAPFROG AND REBUILD KERNELS
// =============================================================================

__global__ void kernel_leapfrog_kick_drift_3d(
    double * __restrict__ x,  double * __restrict__ y,  double * __restrict__ z,
    double * __restrict__ vx, double * __restrict__ vy, double * __restrict__ vz,
    const double * __restrict__ fx,
    const double * __restrict__ fy,
    const double * __restrict__ fz,
    unsigned int n, double box_size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    vx[i] += 0.5*DT*fx[i]; vy[i] += 0.5*DT*fy[i]; vz[i] += 0.5*DT*fz[i];
    x[i]  += DT*vx[i];     y[i]  += DT*vy[i];     z[i]  += DT*vz[i];
    double wx = fmod(x[i],box_size); if (wx<0.0) wx+=box_size; x[i]=wx;
    double wy = fmod(y[i],box_size); if (wy<0.0) wy+=box_size; y[i]=wy;
    double wz = fmod(z[i],box_size); if (wz<0.0) wz+=box_size; z[i]=wz;
}

__global__ void kernel_leapfrog_kick_3d(
    double * __restrict__ vx, double * __restrict__ vy, double * __restrict__ vz,
    const double * __restrict__ fx,
    const double * __restrict__ fy,
    const double * __restrict__ fz,
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    vx[i] += 0.5*DT*fx[i];
    vy[i] += 0.5*DT*fy[i];
    vz[i] += 0.5*DT*fz[i];
}

__global__ void kernel_check_rebuild_3d(
    const double * __restrict__ x,     const double * __restrict__ y,
    const double * __restrict__ z,     const double * __restrict__ x_ref,
    const double * __restrict__ y_ref, const double * __restrict__ z_ref,
    unsigned int n, double box_size, double half_skin_sq, unsigned int *d_flag)
{
    extern __shared__ unsigned int s_flag[];
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    s_flag[tid] = 0;
    if (i < n) {
        double dx = x[i]-x_ref[i], dy = y[i]-y_ref[i], dz = z[i]-z_ref[i];
        dx -= box_size*nearbyint(dx/box_size);
        dy -= box_size*nearbyint(dy/box_size);
        dz -= box_size*nearbyint(dz/box_size);
        if (dx*dx+dy*dy+dz*dz > half_skin_sq) s_flag[tid] = 1;
    }
    __syncthreads();
    for (unsigned int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) s_flag[tid] |= s_flag[tid+stride];
        __syncthreads();
    }
    if (tid == 0 && s_flag[0]) atomicOr(d_flag, 1u);
}

// =============================================================================
// HOST: BUILD NEIGHBOUR LIST (CELL-ACCELERATED, WITH SPATIAL SORT)
// =============================================================================

// static inline void swap_ptr(double **a, double **b) { double *t=*a; *a=*b; *b=t; }

__global__ void kernel_gather6(
    const int    *idx,
    const double *xi, const double *yi, const double *zi,
    const double *vxi, const double *vyi, const double *vzi,
    double *xo, double *yo, double *zo,
    double *vxo, double *vyo, double *vzo,
    unsigned int n)
{
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    int j = idx[k];
    xo[k]=xi[j];   yo[k]=yi[j];   zo[k]=zi[j];
    vxo[k]=vxi[j]; vyo[k]=vyi[j]; vzo[k]=vzi[j];
}

static void gpu_build_neighbour_list_3d(unsigned int n, double box_size) {
    const unsigned int threads = 256;
    const unsigned int blocks  = (n + threads - 1) / threads;

    CUDA_CHECK(cudaMemset(d_cell_count, 0, g_n_cells * sizeof(int)));
    kernel_cell_assign<<<blocks, threads>>>(
        d_x, d_y, d_z, d_cell_id, d_cell_count, n, box_size, g_n_cells_side);

    {
        int *h_count = (int*)malloc(g_n_cells * sizeof(int));
        int *h_start = (int*)malloc(g_n_cells * sizeof(int));
        CUDA_CHECK(cudaMemcpy(h_count, d_cell_count, g_n_cells*sizeof(int), cudaMemcpyDeviceToHost));
        int running = 0;
        for (unsigned int c = 0; c < g_n_cells; c++) { h_start[c] = running; running += h_count[c]; }
        CUDA_CHECK(cudaMemcpy(d_cell_start, h_start, g_n_cells*sizeof(int), cudaMemcpyHostToDevice));
        free(h_count); free(h_start);
    }

    CUDA_CHECK(cudaMemset(d_cell_count, 0, g_n_cells * sizeof(int)));
    kernel_cell_fill<<<blocks, threads>>>(d_cell_id, d_cell_start, d_cell_part, d_cell_count, n);

    g_rebuild_counter++;
    if ((g_rebuild_counter % SPATIAL_SORT_EVERY) == 0) {
        kernel_gather6<<<blocks, threads>>>(
            d_cell_part,
            d_x,  d_y,  d_z,  d_vx,  d_vy,  d_vz,
            d_xs, d_ys, d_zs, d_vxs, d_vys, d_vzs, n);

        // Copy sorted data back to primary buffers (avoids pointer aliasing issues)
        CUDA_CHECK(cudaMemcpy(d_x,  d_xs,  n*sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_y,  d_ys,  n*sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_z,  d_zs,  n*sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_vx, d_vxs, n*sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_vy, d_vys, n*sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_vz, d_vzs, n*sizeof(double), cudaMemcpyDeviceToDevice));

        CUDA_CHECK(cudaMemset(d_cell_count, 0, g_n_cells * sizeof(int)));
        kernel_cell_assign<<<blocks, threads>>>(
            d_x, d_y, d_z, d_cell_id, d_cell_count, n, box_size, g_n_cells_side);
        {
            int *h_count = (int*)malloc(g_n_cells * sizeof(int));
            int *h_start = (int*)malloc(g_n_cells * sizeof(int));
            CUDA_CHECK(cudaMemcpy(h_count, d_cell_count, g_n_cells*sizeof(int), cudaMemcpyDeviceToHost));
            int running = 0;
            for (unsigned int c = 0; c < g_n_cells; c++) { h_start[c] = running; running += h_count[c]; }
            CUDA_CHECK(cudaMemcpy(d_cell_start, h_start, g_n_cells*sizeof(int), cudaMemcpyHostToDevice));
            free(h_count); free(h_start);
        }
        CUDA_CHECK(cudaMemset(d_cell_count, 0, g_n_cells * sizeof(int)));
        kernel_cell_fill<<<blocks, threads>>>(d_cell_id, d_cell_start, d_cell_part, d_cell_count, n);
    }

    CUDA_CHECK(cudaMemset(d_overflow, 0, sizeof(unsigned int)));
    kernel_build_nl_cell<<<blocks, threads>>>(
        d_x, d_y, d_z,
        d_cell_id, d_cell_start, d_cell_count, d_cell_part,
        d_nl, d_nl_count, d_overflow,
        n, g_max_neighbours, box_size, R_LIST_SQ_HOST, g_n_cells_side);

    unsigned int overflow = 0;
    CUDA_CHECK(cudaMemcpy(&overflow, d_overflow, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    if (overflow > 0) {
        fprintf(stderr, "[NL-3D] OVERFLOW: particle %u exceeds max_neighbours=%u. Aborting.\n",
                overflow - 1u, g_max_neighbours);
        exit(1);
    }

    CUDA_CHECK(cudaMemcpy(d_x_ref, d_x, n*sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_ref, d_y, n*sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_z_ref, d_z, n*sizeof(double), cudaMemcpyDeviceToDevice));
}

// =============================================================================
// HOST: CHECK REBUILD, COMPUTE FORCES, LEAPFROG STEP
// =============================================================================

static int gpu_needs_rebuild_3d(unsigned int n, double box_size) {
    CUDA_CHECK(cudaMemsetAsync(d_flag, 0, sizeof(unsigned int)));
    const unsigned int threads = 128;
    const unsigned int blocks  = (n + threads - 1) / threads;
    kernel_check_rebuild_3d<<<blocks, threads, threads * sizeof(unsigned int)>>>(
        d_x, d_y, d_z, d_x_ref, d_y_ref, d_z_ref,
        n, box_size, HALF_SKIN_SQ_HOST, d_flag);
    CUDA_CHECK(cudaMemcpy(h_flag_pinned, d_flag, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    return (int)(*h_flag_pinned);
}

static double gpu_compute_forces_3d(unsigned int n, double box_size) {
    const unsigned int zth = 256;
    kernel_zero_forces<<<(n + zth - 1) / zth, zth>>>(d_fx, d_fy, d_fz, d_pe_arr, n);

    const size_t shm = 4 * g_block_size * sizeof(double);
    kernel_compute_forces_3d<<<n, g_block_size, shm>>>(
        d_x, d_y, d_z, d_fx, d_fy, d_fz, d_pe_arr,
        d_nl, d_nl_count, n, g_max_neighbours, box_size, G_V_SHIFT);

    CUDA_CHECK(cudaMemsetAsync(d_pe_total, 0, sizeof(double)));
    const unsigned int rth = 256;
    kernel_reduce_pe<<<(n + rth - 1) / rth, rth, rth*sizeof(double)>>>(d_pe_arr, d_pe_total, n);

    double pe = 0.0;
    CUDA_CHECK(cudaMemcpy(&pe, d_pe_total, sizeof(double), cudaMemcpyDeviceToHost));
    return pe;
}

static double gpu_leapfrog_step_3d(unsigned int n, double box_size) {
    const unsigned int threads = 256;
    const unsigned int blocks  = (n + threads - 1) / threads;

    kernel_leapfrog_kick_drift_3d<<<blocks, threads>>>(
        d_x, d_y, d_z, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, n, box_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (gpu_needs_rebuild_3d(n, box_size))
        gpu_build_neighbour_list_3d(n, box_size);

    const double pe = gpu_compute_forces_3d(n, box_size);

    kernel_leapfrog_kick_3d<<<blocks, threads>>>(d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return pe;
}

// =============================================================================
// PUBLIC SIMULATION API
// =============================================================================

SimulationResult run_simulation_gpu_v6_3d(Particle *particles, unsigned int n,
                                           unsigned int nsteps, double box_size,
                                           int log_steps) {
    gpu_alloc_3d(n, box_size);
    upload_particles_3d(particles, n);
    G_V_SHIFT = compute_v_shift();

    CUDA_CHECK(cudaMemset(d_fx, 0, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_fy, 0, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_fz, 0, (size_t)n * sizeof(double)));

    gpu_build_neighbour_list_3d(n, box_size);

    SimulationResult out;
    out.start_potential = gpu_compute_forces_3d(n, box_size);
    out.start_kinetic   = compute_ke(particles, n);
    out.start_total     = out.start_kinetic + out.start_potential;

    for (unsigned int step = 0; step < nsteps; ++step) {
        out.final_potential = gpu_leapfrog_step_3d(n, box_size);

        if (log_steps) {
            download_particles_3d(particles, n);
            out.final_kinetic = compute_ke(particles, n);
            out.final_total   = out.final_kinetic + out.final_potential;
            printf("step=%6u  KE=%12.6f  PE=%12.6f  E=%12.6f\n",
                   step, out.final_kinetic, out.final_potential, out.final_total);
        }
    }

    download_particles_3d(particles, n);
    out.final_kinetic = compute_ke(particles, n);
    out.final_total   = out.final_kinetic + out.final_potential;
    out.n         = n;
    out.particles = particles;
    return out;
}

SimulationResult run_simulation_gpu_v7_3d(Particle *particles, unsigned int n,
                                           unsigned int nsteps, double box_size,
                                           int log_steps) {
    return run_simulation_gpu_v6_3d(particles, n, nsteps, box_size, log_steps);
}