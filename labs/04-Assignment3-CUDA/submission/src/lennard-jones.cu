#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

// Include CUDA headers
#include <cuda_runtime.h>
#include <cuda.h>


#include "gifenc.h"
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


// =============================================================================
// CUDA KERNEL FORWARD DECLARATIONS
// =============================================================================

// Per-particle kernels (shared across GPU versions)
__global__ void half_kick_kernel(Particle *particles, unsigned int n);
__global__ void move_positions_kernel(Particle *particles, unsigned int n);
__global__ void wrap_positions_kernel(Particle *particles, unsigned int n, double box_size);
__global__ void zero_forces_kernel(Particle *particles, unsigned int n);
__global__ void compute_ke_kernel(Particle *particles, unsigned int n, double *ke);

// Force kernels
// v2: 2D grid launch — one thread per (i,j) pair, counts each pair twice
__global__ void compute_forces_kernel_v2(Particle *particles, unsigned int n, double box_size, double v_shift, double *pe);
// v3/v4: 1D launch — one thread per unique pair (i,j) with i<j, counts each pair once
__global__ void compute_forces_kernel_v3(Particle *particles, unsigned int n, double box_size, double v_shift, double *pe);

// v5: SoA layout, tiled shared memory, if/else wrap
__global__ void half_kick_kernel_v5      (ParticlesSoA p, unsigned int n);
__global__ void move_positions_kernel_v5 (ParticlesSoA p, unsigned int n);
__global__ void wrap_positions_kernel_v5 (ParticlesSoA p, unsigned int n, double box_size);
__global__ void zero_forces_kernel_v5    (ParticlesSoA p, unsigned int n);
__global__ void compute_ke_kernel_v5     (ParticlesSoA p, unsigned int n, double *ke);
__global__ void compute_forces_kernel_v5 (ParticlesSoA p, unsigned int n, double box_size, double v_shift, double *pe);


static const double SKIN_CANDIDATES[]   = { 0.1, 0.2, 0.3, 0.4, 0.5 };
static const int    N_SKIN_CANDIDATES   = 5;
static const int    WARMUP_STEPS        = 20;   // steps per candidate

#define NEIGHBOUR_SAFETY_FACTOR  2.5
#define NEIGHBOUR_MIN            32


static double        g_skin         = 0.3 * R_CUT;  // chosen skin thickness
static unsigned int  g_block_size   = 64;            // chosen block size
// R_LIST, R_LIST_SQ, HALF_SKIN_SQ derived at runtime from g_skin — see macros:
#define R_LIST          (R_CUT + g_skin)
#define R_LIST_SQ_HOST  ((R_CUT + g_skin) * (R_CUT + g_skin))
#define HALF_SKIN_SQ_HOST ((g_skin * 0.5) * (g_skin * 0.5))

// ─── GPU-side particle arrays (SoA for coalesced access) ─────────────────────
//
// Structure-of-Arrays layout: all x values together, all y values together,
// etc. This ensures adjacent threads in a warp read adjacent memory addresses
// → coalesced 128-byte cache-line loads instead of scattered AoS accesses.
//
static double       *d_x       = nullptr, *d_y       = nullptr;
static double       *d_vx      = nullptr, *d_vy      = nullptr;
static double       *d_fx      = nullptr, *d_fy      = nullptr;
static double       *d_pe_arr  = nullptr;

// ─── Verlet list on device ────────────────────────────────────────────────────
// d_nl[i * g_max_neighbours + k] = index of k-th neighbour of particle i.
// d_nl_count[i]                  = how many neighbours particle i has.
static int          *d_nl       = nullptr;
static unsigned int *d_nl_count = nullptr;

// Reference positions at last list build (to detect rebuild need).
static double       *d_x_ref   = nullptr, *d_y_ref = nullptr;

// Rebuild flag — set by kernel_check_rebuild, read back by host.
static unsigned int *d_flag     = nullptr;

// Overflow flag — set by kernel_build_neighbour_list when a particle's
// true neighbour count exceeds g_max_neighbours. Causes an abort on host.
static unsigned int *d_overflow = nullptr;

// Runtime neighbour capacity — computed once from system density in gpu_alloc.
static unsigned int  g_max_neighbours = 0;

static unsigned int  d_n = 0;  // current allocated size

// plotting functions
#if GENERATE_GIF
uint8_t palette[] = {0, 0, 0,
                     255, 255, 0};

void set_pixel(uint8_t *img, int w, int h, int x, int y, uint8_t index) {
    if (x < 0 || y < 0 || x >= w || y >= h) {
        return;
    }
    size_t idx = (size_t)y * (size_t)w + (size_t)x;
    img[idx] = index;
}

void render_frame_gif(ge_GIF *gif, const Particle *particles, unsigned int n, double box_size) {

    memset(gif->frame, 0, FRAME_WIDTH * FRAME_HEIGHT);

    for (unsigned int i = 0; i < n; ++i) {

        int px = (int)(particles[i].x / box_size * (double)(FRAME_WIDTH - 1));
        int py = (int)(particles[i].y / box_size * (double)(FRAME_HEIGHT - 1));
        py = (FRAME_HEIGHT - 1) - py;

        for (int dy = -FRAME_PARTICLE_RADIUS; dy <= FRAME_PARTICLE_RADIUS; ++dy) {
            for (int dx = -FRAME_PARTICLE_RADIUS; dx <= FRAME_PARTICLE_RADIUS; ++dx) {
                if (dx * dx + dy * dy <= FRAME_PARTICLE_RADIUS * FRAME_PARTICLE_RADIUS) {
                    set_pixel(gif->frame, FRAME_WIDTH, FRAME_HEIGHT, px + dx, py + dy, 1);
                }
            }
        }
    }
}
#endif
double random_double(void) {
    return (double)rand() / (double)RAND_MAX;
}

// compute kinetic energy of the system
double compute_ke(const Particle *particles, unsigned int n) {
    double ke = 0.0;
    for (unsigned int i = 0; i < n; ++i) {
        const Particle *p = &particles[i];
        ke += 0.5 * (p->vx * p->vx + p->vy * p->vy);
    }
    return ke;
}

int initialize_particles(Particle *particles, unsigned int n, double box_size, double placement_fraction, unsigned int seed, double temperature) {
    
    srand(seed);
    unsigned int n_side = (unsigned int)ceil(sqrt((double)n));
    double placement_size = placement_fraction * box_size;
    double offset = 0.5 * (box_size - placement_size);
    double delta = placement_size / (double)n_side;

    double mean_vx = 0.0;
    double mean_vy = 0.0;
    // place particles int he middle of the grid with some random jitter and assign random velocities
    for (unsigned int k = 0; k < n; k++) {
        double x0 = offset + (0.5 + (double)(k % n_side)) * delta;
        double y0 = offset + (0.5 + (double)(k / n_side)) * delta;

        particles[k].x = x0 + (2.0 * random_double() - 1.0) * JITTER * delta;
        particles[k].y = y0 + (2.0 * random_double() - 1.0) * JITTER * delta;

        particles[k].vx = 2.0 * random_double() - 1.0;
        particles[k].vy = 2.0 * random_double() - 1.0;
        
        mean_vx += particles[k].vx;
        mean_vy += particles[k].vy;
    }

    mean_vx /= (double)n;
    mean_vy /= (double)n;
    double ke = 0.0;
    // subtract mean velocity to ensure zero net momentum and compute initial kinetic energy
    for (unsigned int k = 0; k < n; k++) {
        particles[k].vx -= mean_vx;
        particles[k].vy -= mean_vy;
        ke += 0.5 * (
            particles[k].vx * particles[k].vx +
            particles[k].vy * particles[k].vy
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
    }

    return 1;
}

// apply periodic boundary conditions to ensure particles stay within the simulation box
void wrap_positions(Particle *particles, unsigned int n, double box_size) {
    for (unsigned int i = 0; i < n; ++i) {
        Particle *p = &particles[i];
        double wx = fmod(p->x, box_size);
        double wy = fmod(p->y, box_size);

        if (wx < 0.0) {
            wx += box_size;
        }
        if (wy < 0.0) {
            wy += box_size;
        }

        p->x = wx;
        p->y = wy;
    }
}

// shift potential to ensure it goes to zero at the cutoff distance, improving energy conservation
double compute_v_shift(void) {
    return 4.0 * EPSILON * (pow(SIGMA / R_CUT, 12.0) - pow(SIGMA / R_CUT, 6.0));
}

double compute_forces(Particle *particles, unsigned int n, double box_size) {

    for (unsigned int i = 0; i < n; ++i) {
        particles[i].fx = 0.0;
        particles[i].fy = 0.0;
    }
    double pe = 0.0;
    double v_shift = compute_v_shift();
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            if (j == i) {
                continue;
            }
            Particle *pi = &particles[i];
            Particle *pj = &particles[j];
            
            // compute distance between particles with periodic boundary conditions
            double dx = pi->x - pj->x;
            double dy = pi->y - pj->y;

            dx -= box_size * nearbyint(dx / box_size);
            dy -= box_size * nearbyint(dy / box_size);

            // compute Lennard-Jones force and potential energy contribution if particles are within the cutoff distance
            double r = sqrt(dx * dx + dy * dy);
            if (r >= R_CUT || r == 0.0) {
                continue;
            }
            double sr = SIGMA / r;

            double fij = 24.0 * EPSILON * (2.0 * pow(sr, 12.0) - pow(sr, 6.0)) / r;
            double fx = fij * dx / r;
            double fy = fij * dy / r;

            pi->fx += fx;
            pi->fy += fy;

            double vij = 4.0 * EPSILON * (pow(sr, 12.0) - pow(sr, 6.0)) - v_shift;
            pe += 0.5 * vij;
        }
    }

    return pe;
}

double leapfrog_step(Particle *particles, unsigned int n, double box_size) {
    // update velocities by half a time step, then update positions by a full time step, 
    //and finally update velocities by another half time step to complete the leapfrog integration step
    for (unsigned int i = 0; i < n; ++i) {
        Particle *p = &particles[i];
        p->vx += 0.5 * DT * p->fx;
        p->vy += 0.5 * DT * p->fy;

        p->x += DT * p->vx;
        p->y += DT * p->vy;
    }

    wrap_positions(particles, n, box_size);

    double pe = compute_forces(particles, n, box_size);

    for (unsigned int i = 0; i < n; ++i) {
        Particle *p = &particles[i];
        p->vx += 0.5 * DT * p->fx;
        p->vy += 0.5 * DT * p->fy;
    }

    return pe;
}

SimulationResult run_simulation(Particle *particles, unsigned int n, unsigned int nsteps, double box_size, int log_steps) {
    
    SimulationResult out;
    out.start_potential= compute_forces(particles, n, box_size);
    out.start_kinetic = compute_ke(particles, n);
    out.start_total = out.start_kinetic + out.start_potential;

    
#if GENERATE_GIF
    ge_GIF *gif = NULL;

    gif = ge_new_gif(GIF_FILE, (uint16_t)FRAME_WIDTH, (uint16_t)FRAME_HEIGHT, palette, 8, -1, 0);
    if (!gif) {
        fprintf(stderr, "Warning: failed to create GIF output %s\n", GIF_FILE);
    } else {
        render_frame_gif(gif, particles, n, box_size);
        ge_add_frame(gif, FRAME_DELAY);
    }
#endif

    for (unsigned int step = 0; step < nsteps; step++) {
        out.final_potential = leapfrog_step(particles, n, box_size);
        out.final_kinetic = compute_ke(particles, n);
        out.final_total = out.final_kinetic + out.final_potential;
        if (log_steps) {
            printf(
                "step=%6u  KE=%12.6f  PE=%12.6f  E=%12.6f\n",
                step,
                out.final_kinetic,
                out.final_potential,
                out.final_total
            );
        }

    
#if GENERATE_GIF
        if (gif && FRAME_EVERY > 0 && (step + 1) % FRAME_EVERY == 0) {
            render_frame_gif(gif, particles, n, box_size);
            ge_add_frame(gif, FRAME_DELAY);
        }
#endif
    }

#if GENERATE_GIF
    if (gif) {
        ge_close_gif(gif);
    }
#endif

    out.n = n;
    out.particles = particles;
    return out;
}


double compute_ke_seq_v2(const Particle *particles, unsigned int n) {
    double ke = 0.0;
 
    // Simple reduction; very fast compared with force computation.
    #pragma omp parallel for reduction(+:ke) schedule(static)
    for (unsigned int i = 0; i < n; ++i) {
        ke += 0.5 * (particles[i].vx * particles[i].vx +
                     particles[i].vy * particles[i].vy);
    }
    return ke;
}

int initialize_particles_seq_v2(Particle *particles, unsigned int n,
                         double box_size, double placement_fraction,
                         unsigned int seed, double temperature) {
    srand(seed);
 
    unsigned int n_side        = (unsigned int)ceil(sqrt((double)n));
    double       placement_size = placement_fraction * box_size;
    double       offset         = 0.5 * (box_size - placement_size);
    double       delta          = placement_size / (double)n_side;
 
    double mean_vx = 0.0;
    double mean_vy = 0.0;

    #pragma omp parallel reduction(+:mean_vx, mean_vy)
    {
        unsigned int thread_seed = seed + (unsigned int)omp_get_thread_num();

        #pragma omp for schedule(static)
        for (unsigned int k = 0; k < n; k++) {
            double x0 = offset + (0.5 + (double)(k % n_side)) * delta;
            double y0 = offset + (0.5 + (double)(k / n_side)) * delta;

            particles[k].x = x0 + (2.0 * ((double)rand_r(&thread_seed) / (double)RAND_MAX) - 1.0) * JITTER * delta;
            particles[k].y = y0 + (2.0 * ((double)rand_r(&thread_seed) / (double)RAND_MAX) - 1.0) * JITTER * delta;

            particles[k].vx = 2.0 * ((double)rand_r(&thread_seed) / (double)RAND_MAX) - 1.0;
            particles[k].vy = 2.0 * ((double)rand_r(&thread_seed) / (double)RAND_MAX) - 1.0;

            mean_vx += particles[k].vx;
            mean_vy += particles[k].vy;
        }
    }
 
    mean_vx /= (double)n;
    mean_vy /= (double)n;
 
    double ke = 0.0;
    // subtract mean velocity to ensure zero net momentum and compute initial kinetic energy
    // #pragma omp parallel for reduction(+:ke) schedule(static)
    for (unsigned int k = 0; k < n; k++) {
        particles[k].vx -= mean_vx;
        particles[k].vy -= mean_vy;
        ke += 0.5 * (
            particles[k].vx * particles[k].vx +
            particles[k].vy * particles[k].vy
        );
    }

    double current_temperature = ke / (double)n;
    if (current_temperature <= 0.0) {
        return 0;
    }

    // scale velocities to match the desired initial temperature of the system
    double scale = sqrt(temperature / current_temperature);
    // #pragma omp parallel for schedule(static)
    for (unsigned int k = 0; k < n; k++) {
        particles[k].vx *= scale;
        particles[k].vy *= scale;
    }
 
    return 1;
}

// apply periodic boundary conditions to ensure particles stay within the simulation box
void wrap_positions_seq_v2(Particle *particles, unsigned int n, double box_size) {
    #pragma omp parallel for schedule(static)
    for (unsigned int i = 0; i < n; ++i) {
        Particle *p = &particles[i];
        double wx = fmod(p->x, box_size);
        double wy = fmod(p->y, box_size);

        if (wx < 0.0) {
            wx += box_size;
        }
        if (wy < 0.0) {
            wy += box_size;
        }

        p->x = wx;
        p->y = wy;
    }
}

double compute_forces_seq_v2(Particle *particles, unsigned int n, double box_size) {
 
    // Zero forces.
    #pragma omp parallel for schedule(static)
    for (unsigned int i = 0; i < n; ++i) {
        particles[i].fx = 0.0;
        particles[i].fy = 0.0;
    }
 
    const double v_shift   = compute_v_shift();
    const double r_cut_sq  = R_CUT * R_CUT;
    double pe = 0.0;
 
    const int nthreads = omp_get_max_threads();
 
    // Per-thread scratch arrays (heap, so no stack overflow for large N).
    // 16 threads × 8000 particles × 8 bytes × 2 arrays ≈ 2 MB — negligible.
    double *fx_priv = (double *)calloc((size_t)nthreads * n, sizeof(double));
    double *fy_priv = (double *)calloc((size_t)nthreads * n, sizeof(double));
 
    #pragma omp parallel reduction(+:pe)
    {
        const int tid = omp_get_thread_num();
        double *fxi = fx_priv + (size_t)tid * n;
        double *fyi = fy_priv + (size_t)tid * n;
 
        // Dynamic scheduling balances the triangular workload across threads.
        #pragma omp for schedule(dynamic, 16) nowait
        for (unsigned int i = 0; i < n - 1; ++i) {
            const double xi = particles[i].x;
            const double yi = particles[i].y;
 
            for (unsigned int j = i + 1; j < n; ++j) {
                double dx = xi - particles[j].x;
                double dy = yi - particles[j].y;
 
                // Minimum-image convention (branchless via nearbyint).
                dx -= box_size * nearbyint(dx / box_size);
                dy -= box_size * nearbyint(dy / box_size);
 
                double r2 = dx * dx + dy * dy;
                if (r2 >= r_cut_sq || r2 == 0.0) continue;
 
                double inv_r2 = 1.0 / r2;
                double sr2    = (SIGMA * SIGMA) * inv_r2;  // (σ/r)²
                double sr6    = sr2 * sr2 * sr2;
                double sr12   = sr6 * sr6;
 
                // F = 24ε(2(σ/r)¹² − (σ/r)⁶) / r²  (scalar, divided by r²)
                double fij = 24.0 * EPSILON * (2.0 * sr12 - sr6) * inv_r2;
                double fx  = fij * dx;
                double fy  = fij * dy;
 
                // Newton's 3rd law: apply equal-and-opposite forces.
                fxi[i] += fx;  fyi[i] += fy;
                fxi[j] -= fx;  fyi[j] -= fy;
 
                double vij = 4.0 * EPSILON * (sr12 - sr6) - v_shift;
                pe += vij;   // full pair, not halved (each pair visited once)
            }
        }
    }
 
    // Reduce thread-local forces back into particles[].
    #pragma omp parallel for schedule(static)
    for (unsigned int i = 0; i < n; ++i) {
        double fx_sum = 0.0, fy_sum = 0.0;
        for (int t = 0; t < nthreads; ++t) {
            fx_sum += fx_priv[(size_t)t * n + i];
            fy_sum += fy_priv[(size_t)t * n + i];
        }
        particles[i].fx += fx_sum;
        particles[i].fy += fy_sum;
    }
 
    free(fx_priv);
    free(fy_priv);
    return pe;
}

double leapfrog_step_seq_v2(Particle *particles, unsigned int n, double box_size) {
    // update velocities by half a time step, then update positions by a full time step, 
    //and finally update velocities by another half time step to complete the leapfrog integration step
    #pragma omp parallel for schedule(static)
    for (unsigned int i = 0; i < n; ++i) {
        Particle *p = &particles[i];
        p->vx += 0.5 * DT * p->fx;
        p->vy += 0.5 * DT * p->fy;

        p->x += DT * p->vx;
        p->y += DT * p->vy;
    }

    wrap_positions_seq_v2(particles, n, box_size);

    double pe = compute_forces_seq_v2(particles, n, box_size);
    
    #pragma omp parallel for schedule(static)
    for (unsigned int i = 0; i < n; ++i) {
        Particle *p = &particles[i];
        p->vx += 0.5 * DT * p->fx;
        p->vy += 0.5 * DT * p->fy;
    }

    return pe;
}

SimulationResult run_simulation_seq_v2(Particle *particles, unsigned int n, unsigned int nsteps, double box_size, int log_steps) {
    
    SimulationResult out;
    out.start_potential= compute_forces_seq_v2(particles, n, box_size);
    out.start_kinetic = compute_ke_seq_v2(particles, n);
    out.start_total = out.start_kinetic + out.start_potential;

    
#if GENERATE_GIF
    ge_GIF *gif = NULL;

    gif = ge_new_gif(GIF_FILE, (uint16_t)FRAME_WIDTH, (uint16_t)FRAME_HEIGHT, palette, 8, -1, 0);
    if (!gif) {
        fprintf(stderr, "Warning: failed to create GIF output %s\n", GIF_FILE);
    } else {
        render_frame_gif(gif, particles, n, box_size);
        ge_add_frame(gif, FRAME_DELAY);
    }
#endif

    for (unsigned int step = 0; step < nsteps; step++) {
        out.final_potential = leapfrog_step_seq_v2(particles, n, box_size);
        out.final_kinetic = compute_ke_seq_v2(particles, n);
        out.final_total = out.final_kinetic + out.final_potential;
        if (log_steps) {
            printf(
                "step=%6u  KE=%12.6f  PE=%12.6f  E=%12.6f\n",
                step,
                out.final_kinetic,
                out.final_potential,
                out.final_total
            );
        }

    
#if GENERATE_GIF
        if (gif && FRAME_EVERY > 0 && (step + 1) % FRAME_EVERY == 0) {
            render_frame_gif(gif, particles, n, box_size);
            ge_add_frame(gif, FRAME_DELAY);
        }
#endif
    }

#if GENERATE_GIF
    if (gif) {
        ge_close_gif(gif);
    }
#endif

    out.n = n;
    out.particles = particles;
    return out;
}

// =============================================================================
// VERSION 2 — GPU FORCES (2D GRID), CPU INTEGRATION
// Forces computed on GPU with a 2D (i,j) grid kernel.
// Integration stays on CPU. GPU memory is allocated/freed every step.
// =============================================================================

// GPU kernel: 2D grid, thread (i,j) handles particle pair (i,j).
// Each pair is processed twice (i->j and j->i), so pe contribution is * 0.5.
__global__ void compute_forces_kernel_v2(Particle *particles, unsigned int n,
                                         double box_size, double v_shift, double *pe) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n || i == j) return;

    Particle *pi = &particles[i];
    Particle *pj = &particles[j];

    double dx = pi->x - pj->x;
    double dy = pi->y - pj->y;
    dx -= box_size * nearbyint(dx / box_size);
    dy -= box_size * nearbyint(dy / box_size);

    double r = sqrt(dx * dx + dy * dy);
    if (r >= R_CUT || r == 0.0) return;

    double sr  = SIGMA / r;
    double fij = 24.0 * EPSILON * (2.0 * pow(sr, 12.0) - pow(sr, 6.0)) / r;
    atomicAdd(&pi->fx, fij * dx / r);
    atomicAdd(&pi->fy, fij * dy / r);

    double vij = 4.0 * EPSILON * (pow(sr, 12.0) - pow(sr, 6.0)) - v_shift;
    atomicAdd(pe, 0.5 * vij);  // 0.5 because each pair is visited twice
}

// Offloads only force computation to GPU; integration remains on CPU.
// Allocates/frees GPU memory on every call — simple but slow for many steps.
double compute_forces_gpu_v2(Particle *particles, unsigned int n, double box_size) {
    // Zero forces on CPU before upload
    for (unsigned int i = 0; i < n; ++i) {
        particles[i].fx = 0.0;
        particles[i].fy = 0.0;
    }

    Particle *d_particles = NULL;
    double   *d_pe        = NULL;
    cudaMalloc(&d_particles, n * sizeof(Particle));
    cudaMalloc(&d_pe, sizeof(double));

    double pe_host = 0.0;
    cudaMemcpy(d_particles, particles, n * sizeof(Particle), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pe, &pe_host, sizeof(double), cudaMemcpyHostToDevice);

    double v_shift = compute_v_shift();
    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
                 (n + blockDim.y - 1) / blockDim.y);
    compute_forces_kernel_v2<<<gridDim, blockDim>>>(d_particles, n, box_size, v_shift, d_pe);
    cudaDeviceSynchronize();

    cudaMemcpy(particles, d_particles, n * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaMemcpy(&pe_host, d_pe, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_particles);
    cudaFree(d_pe);
    return pe_host;
}

double leapfrog_step_v2(Particle *particles, unsigned int n, double box_size) {
    for (unsigned int i = 0; i < n; ++i) {
        Particle *p = &particles[i];
        p->vx += 0.5 * DT * p->fx;
        p->vy += 0.5 * DT * p->fy;
        p->x  += DT * p->vx;
        p->y  += DT * p->vy;
    }

    wrap_positions(particles, n, box_size);
    double pe = compute_forces_gpu_v2(particles, n, box_size);

    for (unsigned int i = 0; i < n; ++i) {
        Particle *p = &particles[i];
        p->vx += 0.5 * DT * p->fx;
        p->vy += 0.5 * DT * p->fy;
    }
    return pe;
}

SimulationResult run_simulation_v2(Particle *particles, unsigned int n,
                                   unsigned int nsteps, double box_size, int log_steps) {
    SimulationResult out;
    out.start_potential = compute_forces_gpu_v2(particles, n, box_size);
    out.start_kinetic   = compute_ke(particles, n);
    out.start_total     = out.start_kinetic + out.start_potential;

    for (unsigned int step = 0; step < nsteps; step++) {
        out.final_potential = leapfrog_step_v2(particles, n, box_size);
        out.final_kinetic   = compute_ke(particles, n);
        out.final_total     = out.final_kinetic + out.final_potential;

        if (log_steps) {
            printf("step=%6u  KE=%12.6f  PE=%12.6f  E=%12.6f\n",
                   step, out.final_kinetic, out.final_potential, out.final_total);
        }
    }

    out.n = n;
    out.particles = particles;
    return out;
}


// =============================================================================
// VERSION 3 — GPU FORCES (1D PAIR INDEXING), CPU INTEGRATION
// Forces computed on GPU with a smarter 1D kernel over unique pairs.
// Integration stays on CPU. GPU memory is allocated/freed every step.
// =============================================================================

// GPU kernel: 1D launch, one thread per unique pair (i < j).
// Flat index k is mapped back to (i,j) using the triangular number inverse.
// Each pair is processed exactly once, so Newton's 3rd law is applied directly.
__global__ void compute_forces_kernel_v3(Particle *particles, unsigned int n,
                                         double box_size, double v_shift, double *pe) {
    unsigned int num_pairs = n * (n - 1) / 2;
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= num_pairs) return;

    // Recover (i, j) from flat pair index k
    double dk = (double)k;
    double dn = (double)n;
    unsigned int i = (unsigned int)floor(
        (2.0*dn - 1.0 - sqrt((2.0*dn - 1.0)*(2.0*dn - 1.0) - 8.0*dk)) / 2.0);
    unsigned int j = k - i * (2*n - i - 1) / 2 + i + 1;

    Particle *pi = &particles[i];
    Particle *pj = &particles[j];

    double dx = pi->x - pj->x;
    double dy = pi->y - pj->y;
    dx -= box_size * nearbyint(dx / box_size);
    dy -= box_size * nearbyint(dy / box_size);

    double r = sqrt(dx * dx + dy * dy);
    if (r >= R_CUT || r == 0.0) return;

    double sr  = SIGMA / r;
    double fij = 24.0 * EPSILON * (2.0 * pow(sr, 12.0) - pow(sr, 6.0)) / r;
    double fx  = fij * dx / r;
    double fy  = fij * dy / r;

    // Newton's 3rd law: apply equal and opposite forces
    atomicAdd(&pi->fx, +fx);
    atomicAdd(&pi->fy, +fy);
    atomicAdd(&pj->fx, -fx);
    atomicAdd(&pj->fy, -fy);

    // No 0.5 factor — each pair counted exactly once
    double vij = 4.0 * EPSILON * (pow(sr, 12.0) - pow(sr, 6.0)) - v_shift;
    atomicAdd(pe, vij);
}

// Offloads only force computation to GPU; integration remains on CPU.
// Uses the more efficient 1D pair kernel. Allocates/frees GPU memory every call.
double compute_forces_gpu_v3(Particle *particles, unsigned int n, double box_size) {
    for (unsigned int i = 0; i < n; ++i) {
        particles[i].fx = 0.0;
        particles[i].fy = 0.0;
    }

    Particle *d_particles = NULL;
    double   *d_pe        = NULL;
    cudaMalloc(&d_particles, n * sizeof(Particle));
    cudaMalloc(&d_pe, sizeof(double));

    double pe_host = 0.0;
    cudaMemcpy(d_particles, particles, n * sizeof(Particle), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pe, &pe_host, sizeof(double), cudaMemcpyHostToDevice);

    double       v_shift   = compute_v_shift();
    unsigned int num_pairs = n * (n - 1) / 2;
    unsigned int blockSize = 256;
    unsigned int gridSize  = (num_pairs + blockSize - 1) / blockSize;
    compute_forces_kernel_v3<<<gridSize, blockSize>>>(d_particles, n, box_size, v_shift, d_pe);
    cudaDeviceSynchronize();

    cudaMemcpy(particles, d_particles, n * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaMemcpy(&pe_host, d_pe, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_particles);
    cudaFree(d_pe);
    return pe_host;
}

double leapfrog_step_v3(Particle *particles, unsigned int n, double box_size) {
    for (unsigned int i = 0; i < n; ++i) {
        Particle *p = &particles[i];
        p->vx += 0.5 * DT * p->fx;
        p->vy += 0.5 * DT * p->fy;
        p->x  += DT * p->vx;
        p->y  += DT * p->vy;
    }

    wrap_positions(particles, n, box_size);
    double pe = compute_forces_gpu_v3(particles, n, box_size);

    for (unsigned int i = 0; i < n; ++i) {
        Particle *p = &particles[i];
        p->vx += 0.5 * DT * p->fx;
        p->vy += 0.5 * DT * p->fy;
    }
    return pe;
}

SimulationResult run_simulation_v3(Particle *particles, unsigned int n,
                                   unsigned int nsteps, double box_size, int log_steps) {
    SimulationResult out;
    out.start_potential = compute_forces_gpu_v3(particles, n, box_size);
    out.start_kinetic   = compute_ke(particles, n);
    out.start_total     = out.start_kinetic + out.start_potential;

    for (unsigned int step = 0; step < nsteps; step++) {
        out.final_potential = leapfrog_step_v3(particles, n, box_size);
        out.final_kinetic   = compute_ke(particles, n);
        out.final_total     = out.final_kinetic + out.final_potential;

        if (log_steps) {
            printf("step=%6u  KE=%12.6f  PE=%12.6f  E=%12.6f\n",
                   step, out.final_kinetic, out.final_potential, out.final_total);
        }
    }

    out.n = n;
    out.particles = particles;
    return out;
}


// =============================================================================
// VERSION 4 — FULLY GPU
// Everything stays on the GPU for the entire simulation.
// GPU memory is allocated once before the loop and freed once after.
// Uses the 1D pair kernel for forces and dedicated kernels for all other steps.
// =============================================================================

// Per-particle kernels

__global__ void half_kick_kernel(Particle *particles, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Particle *p = &particles[i];
    p->vx += 0.5 * DT * p->fx;
    p->vy += 0.5 * DT * p->fy;
}

__global__ void move_positions_kernel(Particle *particles, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Particle *p = &particles[i];
    p->x += DT * p->vx;
    p->y += DT * p->vy;
}

__global__ void wrap_positions_kernel(Particle *particles, unsigned int n, double box_size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Particle *p = &particles[i];
    double wx = fmod(p->x, box_size);
    double wy = fmod(p->y, box_size);
    if (wx < 0.0) wx += box_size;
    if (wy < 0.0) wy += box_size;
    p->x = wx;
    p->y = wy;
}

__global__ void zero_forces_kernel(Particle *particles, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    particles[i].fx = 0.0;
    particles[i].fy = 0.0;
}

__global__ void compute_ke_kernel(Particle *particles, unsigned int n, double *ke) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Particle *p = &particles[i];
    atomicAdd(ke, 0.5 * (p->vx * p->vx + p->vy * p->vy));
}

// Single leapfrog step entirely on GPU — d_particles and d_pe are persistent device pointers.
double leapfrog_step_v4(Particle *d_particles, unsigned int n, double box_size, double *d_pe) {
    unsigned int blockSize     = 256;
    unsigned int gridSize_n    = (n + blockSize - 1) / blockSize;
    unsigned int num_pairs     = n * (n - 1) / 2;
    unsigned int gridSize_pairs = (num_pairs + blockSize - 1) / blockSize;
    double       v_shift       = compute_v_shift();

    half_kick_kernel        <<<gridSize_n,     blockSize>>>(d_particles, n);
    cudaDeviceSynchronize();

    move_positions_kernel   <<<gridSize_n,     blockSize>>>(d_particles, n);
    cudaDeviceSynchronize();

    wrap_positions_kernel   <<<gridSize_n,     blockSize>>>(d_particles, n, box_size);
    cudaDeviceSynchronize();

    zero_forces_kernel      <<<gridSize_n,     blockSize>>>(d_particles, n);
    cudaDeviceSynchronize();

    cudaMemset(d_pe, 0, sizeof(double));
    compute_forces_kernel_v3<<<gridSize_pairs, blockSize>>>(d_particles, n, box_size, v_shift, d_pe);
    cudaDeviceSynchronize();

    half_kick_kernel        <<<gridSize_n,     blockSize>>>(d_particles, n);
    cudaDeviceSynchronize();

    double pe_host = 0.0;
    cudaMemcpy(&pe_host, d_pe, sizeof(double), cudaMemcpyDeviceToHost);
    return pe_host;
}

SimulationResult run_simulation_v4(Particle *particles, unsigned int n,
                                   unsigned int nsteps, double box_size, int log_steps) {
    SimulationResult out;

    // Allocate GPU memory once for the entire simulation
    Particle *d_particles = NULL;
    double   *d_pe        = NULL;
    double   *d_ke        = NULL;
    cudaMalloc(&d_particles, n * sizeof(Particle));
    cudaMalloc(&d_pe, sizeof(double));
    cudaMalloc(&d_ke, sizeof(double));

    // Upload initial state once
    cudaMemcpy(d_particles, particles, n * sizeof(Particle), cudaMemcpyHostToDevice);

    // Compute initial energies on GPU
    unsigned int blockSize     = 256;
    unsigned int gridSize_n    = (n + blockSize - 1) / blockSize;
    unsigned int num_pairs     = n * (n - 1) / 2;
    unsigned int gridSize_pairs = (num_pairs + blockSize - 1) / blockSize;
    double       v_shift       = compute_v_shift();

    zero_forces_kernel<<<gridSize_n, blockSize>>>(d_particles, n);
    cudaMemset(d_pe, 0, sizeof(double));
    compute_forces_kernel_v3<<<gridSize_pairs, blockSize>>>(d_particles, n, box_size, v_shift, d_pe);

    cudaMemset(d_ke, 0, sizeof(double));
    compute_ke_kernel<<<gridSize_n, blockSize>>>(d_particles, n, d_ke);
    cudaDeviceSynchronize();

    cudaMemcpy(&out.start_potential, d_pe, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out.start_kinetic,   d_ke, sizeof(double), cudaMemcpyDeviceToHost);
    out.start_total = out.start_kinetic + out.start_potential;

    // Main loop — data never leaves the GPU
    for (unsigned int step = 0; step < nsteps; step++) {
        out.final_potential = leapfrog_step_v4(d_particles, n, box_size, d_pe);

        cudaMemset(d_ke, 0, sizeof(double));
        compute_ke_kernel<<<gridSize_n, blockSize>>>(d_particles, n, d_ke);
        cudaDeviceSynchronize();
        cudaMemcpy(&out.final_kinetic, d_ke, sizeof(double), cudaMemcpyDeviceToHost);

        out.final_total = out.final_kinetic + out.final_potential;

        if (log_steps) {
            printf("step=%6u  KE=%12.6f  PE=%12.6f  E=%12.6f\n",
                   step, out.final_kinetic, out.final_potential, out.final_total);
        }
    }

    // Download final state once
    cudaMemcpy(particles, d_particles, n * sizeof(Particle), cudaMemcpyDeviceToHost);

    // Free GPU memory once
    cudaFree(d_particles);
    cudaFree(d_pe);
    cudaFree(d_ke);

    out.n = n;
    out.particles = particles;
    return out;
}


// =============================================================================
// VERSION 5 — FULLY GPU, SOA LAYOUT, SHARED MEMORY TILING, IF/ELSE WRAP
//
// Three improvements over v4:
//
//   1. SoA memory layout — x[], y[], vx[], vy[], fx[], fy[] stored as separate
//      contiguous arrays. Warp reads (e.g. all x[i..i+31]) are now a single
//      coalesced fetch instead of strided AoS accesses.
//
//   2. Tiled shared memory in the force kernel — positions are loaded into
//      on-chip shared memory in tiles of TILE_SIZE. Every thread in the block
//      reuses the same tile data, reducing global memory traffic by ~TILE_SIZE.
//      Forces for particle i are accumulated in registers across the tile loop
//      and flushed with a single atomicAdd per particle at the end.
//
//   3. if/else periodic wrap — replaces fmod (which internally divides) with a
//      single branch per axis. Safe because DT=0.002 makes crossing more than
//      one box per step physically impossible.
// =============================================================================

#define TILE_SIZE 32  // matches warp size for best occupancy

// ---------------------------------------------------------------------------
// Per-particle kernels (SoA)
// ---------------------------------------------------------------------------

__global__ void half_kick_kernel_v5(ParticlesSoA p, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    p.vx[i] += 0.5 * DT * p.fx[i];
    p.vy[i] += 0.5 * DT * p.fy[i];
}

__global__ void move_positions_kernel_v5(ParticlesSoA p, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    p.x[i] += DT * p.vx[i];
    p.y[i] += DT * p.vy[i];
}

// if/else wrap — no division, one branch per axis
__global__ void wrap_positions_kernel_v5(ParticlesSoA p, unsigned int n, double box_size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double xi = p.x[i];
    double yi = p.y[i];

    if      (xi >= box_size) xi -= box_size;
    else if (xi <  0.0)      xi += box_size;

    if      (yi >= box_size) yi -= box_size;
    else if (yi <  0.0)      yi += box_size;

    p.x[i] = xi;
    p.y[i] = yi;
}

__global__ void zero_forces_kernel_v5(ParticlesSoA p, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    p.fx[i] = 0.0;
    p.fy[i] = 0.0;
}

__global__ void compute_ke_kernel_v5(ParticlesSoA p, unsigned int n, double *ke) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double vx = p.vx[i];
    double vy = p.vy[i];
    atomicAdd(ke, 0.5 * (vx * vx + vy * vy));
}

// ---------------------------------------------------------------------------
// Force kernel — tiled shared memory, SoA, register accumulation
//
// Grid: one thread per particle i (same as per-particle kernels).
// Each block loops over tiles of TILE_SIZE j-particles:
//   - All threads cooperatively load x_j, y_j into shared memory (one
//     coalesced fetch per tile instead of n random global reads per thread).
//   - Each thread accumulates its force and pe into registers (fx_i, fy_i,
//     pe_i) — no atomicAdds inside the tile loop.
// After all tiles, one atomicAdd per particle flushes the accumulators.
// Uses r2 instead of r wherever possible to avoid an extra sqrt.
// ---------------------------------------------------------------------------
__global__ void compute_forces_kernel_v5(ParticlesSoA p, unsigned int n,
                                          double box_size, double v_shift, double *pe) {
    unsigned int num_pairs = n * (n - 1) / 2;
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= num_pairs) return;

    // Recover (i, j) from flat pair index — same as v4
    double dk = (double)k;
    double dn = (double)n;
    unsigned int i = (unsigned int)floor(
        (2.0*dn - 1.0 - sqrt((2.0*dn - 1.0)*(2.0*dn - 1.0) - 8.0*dk)) / 2.0);
    unsigned int j = k - i * (2*n - i - 1) / 2 + i + 1;

    // Coalesced reads from SoA — kept from v5
    double xi = p.x[i];
    double yi = p.y[i];
    double xj = p.x[j];
    double yj = p.y[j];

    double dx = xi - xj;
    double dy = yi - yj;
    dx -= box_size * nearbyint(dx / box_size);
    dy -= box_size * nearbyint(dy / box_size);

    double r2 = dx * dx + dy * dy;
    if (r2 == 0.0 || r2 >= R_CUT * R_CUT) return;

    // r2-based computation — kept from v5
    double r    = sqrt(r2);
    double sr   = SIGMA / r;
    double sr6  = sr * sr * sr * sr * sr * sr;
    double sr12 = sr6 * sr6;

    double fij = 24.0 * EPSILON * (2.0 * sr12 - sr6) / r2;
    double fx  = fij * dx;
    double fy  = fij * dy;

    // Newton's 3rd law — kept from v4
    atomicAdd(&p.fx[i], +fx);
    atomicAdd(&p.fy[i], +fy);
    atomicAdd(&p.fx[j], -fx);
    atomicAdd(&p.fy[j], -fy);

    // No 0.5 — each pair counted once
    atomicAdd(pe, 4.0 * EPSILON * (sr12 - sr6) - v_shift);
}


// ---------------------------------------------------------------------------
// SoA helpers: allocate / free device arrays, convert to/from AoS host layout
// ---------------------------------------------------------------------------

static ParticlesSoA alloc_soa_device(unsigned int n) {
    ParticlesSoA d;
    cudaMalloc(&d.x,  n * sizeof(double));
    cudaMalloc(&d.y,  n * sizeof(double));
    cudaMalloc(&d.vx, n * sizeof(double));
    cudaMalloc(&d.vy, n * sizeof(double));
    cudaMalloc(&d.fx, n * sizeof(double));
    cudaMalloc(&d.fy, n * sizeof(double));
    return d;
}

static void free_soa_device(ParticlesSoA d) {
    cudaFree(d.x);  cudaFree(d.y);
    cudaFree(d.vx); cudaFree(d.vy);
    cudaFree(d.fx); cudaFree(d.fy);
}

// AoS host → SoA device (called once before the simulation loop)
static void aos_to_soa_upload(const Particle *h, ParticlesSoA d, unsigned int n) {
    double *tmp = (double *)malloc(n * sizeof(double));

    #define UPLOAD(field, dst) \
        for (unsigned int k = 0; k < n; ++k) tmp[k] = h[k].field; \
        cudaMemcpy(dst, tmp, n * sizeof(double), cudaMemcpyHostToDevice);

    UPLOAD(x,  d.x)   UPLOAD(y,  d.y)
    UPLOAD(vx, d.vx)  UPLOAD(vy, d.vy)
    UPLOAD(fx, d.fx)  UPLOAD(fy, d.fy)
    #undef UPLOAD

    free(tmp);
}

// SoA device → AoS host (called once after the simulation loop)
static void soa_to_aos_download(ParticlesSoA d, Particle *h, unsigned int n) {
    double *tmp = (double *)malloc(n * sizeof(double));

    #define DOWNLOAD(src, field) \
        cudaMemcpy(tmp, src, n * sizeof(double), cudaMemcpyDeviceToHost); \
        for (unsigned int k = 0; k < n; ++k) h[k].field = tmp[k];

    DOWNLOAD(d.x,  x)   DOWNLOAD(d.y,  y)
    DOWNLOAD(d.vx, vx)  DOWNLOAD(d.vy, vy)
    DOWNLOAD(d.fx, fx)  DOWNLOAD(d.fy, fy)
    #undef DOWNLOAD

    free(tmp);
}

// ---------------------------------------------------------------------------
// Leapfrog step — all kernels use SoA, data stays on device the whole time
// ---------------------------------------------------------------------------
double leapfrog_step_v5(ParticlesSoA d, unsigned int n, double box_size, double *d_pe) {
    unsigned int blockSize      = 256;
    unsigned int gridSize_n     = (n + blockSize - 1) / blockSize;
    unsigned int num_pairs      = n * (n - 1) / 2;
    unsigned int gridSize_pairs = (num_pairs + blockSize - 1) / blockSize;
    double       v_shift        = compute_v_shift();

    half_kick_kernel_v5      <<<gridSize_n,     blockSize>>>(d, n);
    cudaDeviceSynchronize();

    move_positions_kernel_v5 <<<gridSize_n,     blockSize>>>(d, n);
    cudaDeviceSynchronize();

    wrap_positions_kernel_v5 <<<gridSize_n,     blockSize>>>(d, n, box_size);
    cudaDeviceSynchronize();

    zero_forces_kernel_v5    <<<gridSize_n,     blockSize>>>(d, n);
    cudaDeviceSynchronize();

    cudaMemset(d_pe, 0, sizeof(double));
    compute_forces_kernel_v5 <<<gridSize_pairs, blockSize>>>(d, n, box_size, v_shift, d_pe);
    cudaDeviceSynchronize();

    half_kick_kernel_v5      <<<gridSize_n,     blockSize>>>(d, n);
    cudaDeviceSynchronize();

    double pe_host = 0.0;
    cudaMemcpy(&pe_host, d_pe, sizeof(double), cudaMemcpyDeviceToHost);
    return pe_host;
}

SimulationResult run_simulation_v5(Particle *particles, unsigned int n,
                                   unsigned int nsteps, double box_size, int log_steps) {
    SimulationResult out;

    // Allocate SoA device arrays and scratch scalars once
    ParticlesSoA d    = alloc_soa_device(n);
    double      *d_pe = NULL;
    double      *d_ke = NULL;
    cudaMalloc(&d_pe, sizeof(double));
    cudaMalloc(&d_ke, sizeof(double));

    // Upload initial state once (AoS host → SoA device)
    aos_to_soa_upload(particles, d, n);

    unsigned int blockSize      = 256;
    unsigned int gridSize_n     = (n + blockSize - 1) / blockSize;
    unsigned int num_pairs      = n * (n - 1) / 2;
    unsigned int gridSize_pairs = (num_pairs + blockSize - 1) / blockSize;
    double       v_shift        = compute_v_shift();

    // Compute initial energies on GPU
    zero_forces_kernel_v5<<<gridSize_n, blockSize>>>(d, n);
    cudaMemset(d_pe, 0, sizeof(double));
    compute_forces_kernel_v5<<<gridSize_pairs, blockSize>>>(d, n, box_size, v_shift, d_pe);

    cudaMemset(d_ke, 0, sizeof(double));
    compute_ke_kernel_v5<<<gridSize_n, blockSize>>>(d, n, d_ke);
    cudaDeviceSynchronize();

    cudaMemcpy(&out.start_potential, d_pe, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out.start_kinetic,   d_ke, sizeof(double), cudaMemcpyDeviceToHost);
    out.start_total = out.start_kinetic + out.start_potential;

    // Main loop — data never leaves the GPU
    for (unsigned int step = 0; step < nsteps; step++) {
        out.final_potential = leapfrog_step_v5(d, n, box_size, d_pe);

        cudaMemset(d_ke, 0, sizeof(double));
        compute_ke_kernel_v5<<<gridSize_n, blockSize>>>(d, n, d_ke);
        cudaDeviceSynchronize();
        cudaMemcpy(&out.final_kinetic, d_ke, sizeof(double), cudaMemcpyDeviceToHost);

        out.final_total = out.final_kinetic + out.final_potential;

        if (log_steps) {
            printf("step=%6u  KE=%12.6f  PE=%12.6f  E=%12.6f\n",
                   step, out.final_kinetic, out.final_potential, out.final_total);
        }
    }

    // Download final state once (SoA device → AoS host)
    soa_to_aos_download(d, particles, n);

    // Free all device memory once
    free_soa_device(d);
    cudaFree(d_pe);
    cudaFree(d_ke);

    out.n = n;
    out.particles = particles;
    return out;
}

//------ Neighbor list + block of threads version (v3) ------------------------------
static void gpu_alloc(unsigned int n, double box_size) {
    if (d_n == n) return;
    if (d_n > 0) {
        CUDA_CHECK(cudaFree(d_x));       CUDA_CHECK(cudaFree(d_y));
        CUDA_CHECK(cudaFree(d_vx));      CUDA_CHECK(cudaFree(d_vy));
        CUDA_CHECK(cudaFree(d_fx));      CUDA_CHECK(cudaFree(d_fy));
        CUDA_CHECK(cudaFree(d_pe_arr));
        if (d_nl) { CUDA_CHECK(cudaFree(d_nl)); d_nl = nullptr; }
        CUDA_CHECK(cudaFree(d_nl_count));
        CUDA_CHECK(cudaFree(d_x_ref));   CUDA_CHECK(cudaFree(d_y_ref));
        CUDA_CHECK(cudaFree(d_flag));    CUDA_CHECK(cudaFree(d_overflow));
    }

    // ── Heuristic: compute max_neighbours from system density ────────────────
    //
    //   expected neighbours = (N / box_area) * π * R_LIST²
    //   max_neighbours      = NEIGHBOUR_SAFETY_FACTOR * expected
    //
    // The 2.5× safety factor covers local density fluctuations. A hard floor
    // of NEIGHBOUR_MIN guards very sparse systems.
    {
        const double density  = (double)n / (box_size * box_size);
        const double expected = density * M_PI * R_LIST * R_LIST;
        const unsigned int computed = (unsigned int)(NEIGHBOUR_SAFETY_FACTOR * expected) + 1;
        g_max_neighbours = (computed > NEIGHBOUR_MIN) ? computed : NEIGHBOUR_MIN;
        fprintf(stdout,
                "[NL] density=%.4f  expected neighbours=%.1f  "
                "max_neighbours=%u  (factor=%.1f, min=%u)\n",
                density, expected, g_max_neighbours,
                NEIGHBOUR_SAFETY_FACTOR, NEIGHBOUR_MIN);
    }

    const size_t sz = (size_t)n * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_x,      sz));
    CUDA_CHECK(cudaMalloc(&d_y,      sz));
    CUDA_CHECK(cudaMalloc(&d_vx,     sz));
    CUDA_CHECK(cudaMalloc(&d_vy,     sz));
    CUDA_CHECK(cudaMalloc(&d_fx,     sz));
    CUDA_CHECK(cudaMalloc(&d_fy,     sz));
    CUDA_CHECK(cudaMalloc(&d_pe_arr, sz));
    CUDA_CHECK(cudaMalloc(&d_x_ref,  sz));
    CUDA_CHECK(cudaMalloc(&d_y_ref,  sz));
    // d_nl is NOT allocated here — tune_skin allocates it after choosing
    // the optimal skin so the size matches the actual neighbour count.
    // d_nl_count is allocated here since its size is always n regardless of skin.
    CUDA_CHECK(cudaMalloc(&d_nl_count, (size_t)n * sizeof(unsigned int)));
    d_nl = nullptr;   // explicitly null — will be set by tune_skin
    CUDA_CHECK(cudaMalloc(&d_flag,     sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_overflow, sizeof(unsigned int)));
    d_n = n;
}

// ─── host↔device transfers ────────────────────────────────────────────────────
//
// Scatter AoS host layout → SoA device layout on upload.
// Gather SoA device layout → AoS host layout on download.
// Transfers only happen at init, at the final step, and when
// log/GIF output requires host-side data.
//
static void upload_particles(const Particle *p, unsigned int n) {
    const size_t sz = n * sizeof(double);
    double *hx  = (double *)malloc(sz), *hy  = (double *)malloc(sz);
    double *hvx = (double *)malloc(sz), *hvy = (double *)malloc(sz);
    for (unsigned int i = 0; i < n; ++i) {
        hx[i]=p[i].x;  hy[i]=p[i].y;
        hvx[i]=p[i].vx; hvy[i]=p[i].vy;
    }
    CUDA_CHECK(cudaMemcpy(d_x,  hx,  sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y,  hy,  sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx, hvx, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, hvy, sz, cudaMemcpyHostToDevice));
    free(hx); free(hy); free(hvx); free(hvy);
}

static void download_particles(Particle *p, unsigned int n) {
    const size_t sz = n * sizeof(double);
    double *hx  = (double *)malloc(sz), *hy  = (double *)malloc(sz);
    double *hvx = (double *)malloc(sz), *hvy = (double *)malloc(sz);
    double *hfx = (double *)malloc(sz), *hfy = (double *)malloc(sz);
    CUDA_CHECK(cudaMemcpy(hx,  d_x,  sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hy,  d_y,  sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hvx, d_vx, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hvy, d_vy, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hfx, d_fx, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hfy, d_fy, sz, cudaMemcpyDeviceToHost));
    for (unsigned int i = 0; i < n; ++i) {
        p[i].x=hx[i];  p[i].y=hy[i];
        p[i].vx=hvx[i]; p[i].vy=hvy[i];
        p[i].fx=hfx[i]; p[i].fy=hfy[i];
    }
    free(hx); free(hy); free(hvx); free(hvy); free(hfx); free(hfy);
}

// ─── KERNEL: build Verlet neighbour list ─────────────────────────────────────
//
// One thread per particle i. Iterates all j != i, records those within
// R_LIST into d_nl[i * g_max_neighbours + ...].
//
// Called infrequently (~every 20 steps) so the O(N²) cost is amortised.
// For very large N a cell-list pre-sort would further accelerate list
// building, but is unnecessary at N <= 8000.
//
__global__ void kernel_build_neighbour_list(
    const double * __restrict__ x,
    const double * __restrict__ y,
    int          * __restrict__ nl,
    unsigned int * __restrict__ nl_count,
    unsigned int * __restrict__ d_overflow,
    unsigned int n,
    unsigned int max_neighbours,
    double box_size,
    double r_list_sq)           // precomputed on host: (R_CUT + skin)²
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const double xi = x[i];
    const double yi = y[i];
    unsigned int count = 0;

    for (unsigned int j = 0; j < n; ++j) {
        if (j == i) continue;
        double dx = xi - x[j];
        double dy = yi - y[j];
        dx -= box_size * nearbyint(dx / box_size);
        dy -= box_size * nearbyint(dy / box_size);
        if (dx*dx + dy*dy < r_list_sq) {
            if (count < max_neighbours)
                nl[i * max_neighbours + count] = (int)j;
            count++;
        }
    }

    // Store clamped count. If the true count exceeds the capacity,
    // set the overflow flag so the host can detect and abort cleanly
    // rather than silently dropping neighbours and producing wrong physics.
    if (count > max_neighbours) {
        nl_count[i] = max_neighbours;
        atomicOr(d_overflow, i + 1u);  // store offending particle index + 1
    } else {
        nl_count[i] = count;
    }
}

// ─── KERNEL: compute forces using neighbour list ──────────────────────────────
//
// Grid:  one block per particle i  (gridDim.x = N).
// Block: BLOCK_SIZE threads cooperate over particle i's neighbour list.
//
// Each thread strides over the neighbour list with step BLOCK_SIZE,
// accumulates partial (fx, fy, pe) into registers, then a shared-memory
// tree reduction produces the final values for particle i.
//
// Why block-per-particle?
//   - Neighbour list lengths vary; a single thread per particle leaves
//     most of the warp idle while one thread grinds through a long list.
//   - With a full block, all BLOCK_SIZE threads stay busy on the neighbour
//     loop, and the reduction costs only log2(BLOCK_SIZE) steps.
//
__global__ void kernel_compute_forces(
    const double * __restrict__ x,
    const double * __restrict__ y,
    double       * __restrict__ fx,
    double       * __restrict__ fy,
    double       * __restrict__ pe_out,
    const int    * __restrict__ nl,
    const unsigned int * __restrict__ nl_count,
    unsigned int n,
    unsigned int max_neighbours,
    double box_size,
    double v_shift)
{
    // Dynamic shared memory: caller passes 3 * blockDim.x * sizeof(double).
    // Layout: [0..bs)=s_fx  [bs..2bs)=s_fy  [2bs..3bs)=s_pe
    extern __shared__ double s_shared[];
    const unsigned int bs  = blockDim.x;
    double *s_fx = s_shared;
    double *s_fy = s_shared + bs;
    double *s_pe = s_shared + 2 * bs;

    const unsigned int i   = blockIdx.x;   // one block per particle
    const unsigned int tid = threadIdx.x;

    if (i >= n) return;

    const double xi       = x[i];
    const double yi       = y[i];
    const double r_cut_sq = R_CUT * R_CUT;
    const double sig2     = SIGMA * SIGMA;

    double lfx = 0.0, lfy = 0.0, lpe = 0.0;
    const unsigned int nn = nl_count[i];

    // Each thread strides over the neighbour list with step blockDim.x.
    // Adjacent threads read consecutive nl[] entries → coalesced.
    for (unsigned int k = tid; k < nn; k += bs) {
        const unsigned int j = (unsigned int)nl[i * max_neighbours + k];

        double dx = xi - x[j];
        double dy = yi - y[j];
        dx -= box_size * nearbyint(dx / box_size);
        dy -= box_size * nearbyint(dy / box_size);

        double r2 = dx*dx + dy*dy;
        if (r2 >= r_cut_sq || r2 == 0.0) continue;

        double inv_r2 = 1.0 / r2;
        double sr2    = sig2 * inv_r2;
        double sr6    = sr2 * sr2 * sr2;
        double sr12   = sr6 * sr6;

        double fij = 24.0 * EPSILON * (2.0*sr12 - sr6) * inv_r2;
        lfx += fij * dx;
        lfy += fij * dy;
        // Factor of 0.5: each pair (i,j) appears in both i's and j's list.
        lpe += 0.5 * (4.0*EPSILON*(sr12 - sr6) - v_shift);
    }

    s_fx[tid] = lfx;
    s_fy[tid] = lfy;
    s_pe[tid] = lpe;
    __syncthreads();

    // Tree reduction: log2(blockDim.x) steps.
    for (unsigned int stride = bs / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_fx[tid] += s_fx[tid + stride];
            s_fy[tid] += s_fy[tid + stride];
            s_pe[tid] += s_pe[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes the final result for particle i.
    if (tid == 0) {
        fx[i]     = s_fx[0];
        fy[i]     = s_fy[0];
        pe_out[i] = s_pe[0];
    }
}

// ─── KERNEL: leapfrog half-kick + drift + wrap ────────────────────────────────
//
// Wrap is folded in here to avoid a separate kernel launch per step.
//
__global__ void kernel_leapfrog_kick_drift(
    double       * __restrict__ x,
    double       * __restrict__ y,
    double       * __restrict__ vx,
    double       * __restrict__ vy,
    const double * __restrict__ fx,
    const double * __restrict__ fy,
    unsigned int n, double box_size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    vx[i] += 0.5 * DT * fx[i];
    vy[i] += 0.5 * DT * fy[i];
    x[i]  += DT * vx[i];
    y[i]  += DT * vy[i];

    double wx = fmod(x[i], box_size); if (wx < 0.0) wx += box_size; x[i] = wx;
    double wy = fmod(y[i], box_size); if (wy < 0.0) wy += box_size; y[i] = wy;
}

// ─── KERNEL: leapfrog second half-kick ───────────────────────────────────────
__global__ void kernel_leapfrog_kick(
    double       * __restrict__ vx,
    double       * __restrict__ vy,
    const double * __restrict__ fx,
    const double * __restrict__ fy,
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    vx[i] += 0.5 * DT * fx[i];
    vy[i] += 0.5 * DT * fy[i];
}

// ─── KERNEL: displacement check for neighbour list rebuild ───────────────────
//
// Sets d_flag=1 if any particle has moved more than SKIN/2 from its
// reference position. The check covers all particles — not just the
// selected one — because two particles converging head-on can together
// close a gap of SKIN with each moving only SKIN/2.
//
__global__ void kernel_check_rebuild(
    const double * __restrict__ x,
    const double * __restrict__ y,
    const double * __restrict__ x_ref,
    const double * __restrict__ y_ref,
    unsigned int n, double box_size,
    double half_skin_sq,        // precomputed on host: (skin/2)²
    unsigned int *d_flag)
{
    extern __shared__ unsigned int s_flag[];
    const unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tid = threadIdx.x;
    s_flag[tid] = 0;

    if (i < n) {
        double dx = x[i] - x_ref[i];
        double dy = y[i] - y_ref[i];
        dx -= box_size * nearbyint(dx / box_size);
        dy -= box_size * nearbyint(dy / box_size);
        if (dx*dx + dy*dy > half_skin_sq)
            s_flag[tid] = 1;
    }
    __syncthreads();

    for (unsigned int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) s_flag[tid] |= s_flag[tid + stride];
        __syncthreads();
    }
    if (tid == 0 && s_flag[0])
        atomicOr(d_flag, 1u);
}

// ─── host: build neighbour list on GPU ───────────────────────────────────────
static void gpu_build_neighbour_list(unsigned int n, double box_size) {
    const unsigned int threads = 256;
    const unsigned int blocks  = (n + threads - 1) / threads;

    // Clear overflow flag before each build.
    CUDA_CHECK(cudaMemset(d_overflow, 0, sizeof(unsigned int)));

    const double r_list_sq = R_LIST_SQ_HOST;   // computed on host from g_skin
    kernel_build_neighbour_list<<<blocks, threads>>>(
        d_x, d_y, d_nl, d_nl_count, d_overflow, n, g_max_neighbours, box_size,
        r_list_sq);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check overflow: if any particle exceeded g_max_neighbours the flag
    // holds that particle's index + 1 so we can report it precisely.
    unsigned int overflow = 0;
    CUDA_CHECK(cudaMemcpy(&overflow, d_overflow, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    if (overflow > 0) {
        const unsigned int bad_particle = overflow - 1u;
        fprintf(stderr,
                "[NL] OVERFLOW: particle %u has more than %u neighbours "
                "(max_neighbours=%u, density=%.4f, R_LIST=%.4f).\n"
                "[NL] Increase NEIGHBOUR_SAFETY_FACTOR or reduce density.\n"
                "[NL] Aborting to avoid silent physics errors.\n",
                bad_particle, g_max_neighbours, g_max_neighbours,
                (double)n / (box_size * box_size), (double)R_LIST);
        exit(1);
    }

    // Snapshot reference positions for displacement tracking.
    CUDA_CHECK(cudaMemcpy(d_x_ref, d_x, n*sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_ref, d_y, n*sizeof(double), cudaMemcpyDeviceToDevice));
}

// ─── host: check whether list needs rebuilding ────────────────────────────────
static int gpu_needs_rebuild(unsigned int n, double box_size) {
    CUDA_CHECK(cudaMemset(d_flag, 0, sizeof(unsigned int)));
    const unsigned int threads = 128;  // fixed — independent of force block size
    const unsigned int blocks  = (n + threads - 1) / threads;
    const double half_skin_sq = HALF_SKIN_SQ_HOST; // computed on host from g_skin
    kernel_check_rebuild<<<blocks, threads, threads * sizeof(unsigned int)>>>(
        d_x, d_y, d_x_ref, d_y_ref, n, box_size, half_skin_sq, d_flag);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned int flag = 0;
    CUDA_CHECK(cudaMemcpy(&flag, d_flag, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    return (int)flag;
}

// ─── host: compute forces and return total PE ─────────────────────────────────
static double gpu_compute_forces(unsigned int n, double box_size) {
    const double v_shift = compute_v_shift();

    const size_t shm = 3 * g_block_size * sizeof(double);
    kernel_compute_forces<<<n, g_block_size, shm>>>(
        d_x, d_y, d_fx, d_fy, d_pe_arr,
        d_nl, d_nl_count, n, g_max_neighbours, box_size, v_shift);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Sum per-particle PE on host. Transfer is N doubles — negligible.
    double *h_pe = (double *)malloc(n * sizeof(double));
    CUDA_CHECK(cudaMemcpy(h_pe, d_pe_arr, n*sizeof(double),
                          cudaMemcpyDeviceToHost));
    double pe = 0.0;
    for (unsigned int i = 0; i < n; ++i) pe += h_pe[i];
    free(h_pe);
    return pe;
}

// ─── host: one full leapfrog timestep on GPU ─────────────────────────────────
static double gpu_leapfrog_step(unsigned int n, double box_size) {
    const unsigned int threads = 256;
    const unsigned int blocks  = (n + threads - 1) / threads;

    // Half-kick + drift + wrap (one kernel, no extra launch).
    kernel_leapfrog_kick_drift<<<blocks, threads>>>(
        d_x, d_y, d_vx, d_vy, d_fx, d_fy, n, box_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Rebuild neighbour list only when any particle has moved > SKIN/2.
    if (gpu_needs_rebuild(n, box_size))
        gpu_build_neighbour_list(n, box_size);

    const double pe = gpu_compute_forces(n, box_size);

    // Second half-kick with updated forces.
    kernel_leapfrog_kick<<<blocks, threads>>>(d_vx, d_vy, d_fx, d_fy, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return pe;
}

// ─── public API ───────────────────────────────────────────────────────────────
//
// tune_block_size()
// -----------------
// Inspects the current neighbour list (must have been built already via
// tune_skin or a prior simulation call) and returns the smallest power-of-2
// block size that keeps all threads busy for at least one full stride pass.
// Clamped to [32, 128].
//
// Also writes the chosen value into g_block_size so subsequent
// gpu_compute_forces calls use it automatically.
//
// Call after tune_skin (which builds the list) or after the first
// gpu_build_neighbour_list in your own setup code.
//
unsigned int tune_block_size(unsigned int n) {
    // Download nl_count to compute the average neighbour count.
    unsigned int *h_count = (unsigned int *)malloc(n * sizeof(unsigned int));
    CUDA_CHECK(cudaMemcpy(h_count, d_nl_count,
                          n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    unsigned long long total = 0;
    for (unsigned int i = 0; i < n; ++i) total += h_count[i];
    free(h_count);

    const double avg_nn = (double)total / (double)n;

    // Walk up powers of 2 until block size >= avg neighbour count.
    unsigned int bs = 32;
    while (bs < (unsigned int)avg_nn && bs < 128) bs *= 2;

    g_block_size = bs;

    fprintf(stdout,
            "[TUNE] tune_block_size: avg_neighbours=%.1f  block_size=%u\n",
            avg_nn, bs);
    return bs;
}

// tune_skin()
// -----------
// Sweeps SKIN_CANDIDATES skin fractions, runs WARMUP_STEPS leapfrog steps
// per candidate with cudaEvent GPU timing, and picks the skin that minimises
// wall time per step (force kernel cost + amortised rebuild overhead).
//
// Parameters:
//   particles  — initial particle state. If GPU memory has not been
//                allocated yet (first call), tune_skin allocates and uploads
//                automatically. No pre-setup required.
//   n          — particle count
//   box_size   — simulation box side length
//
// Returns a TuneResult containing:
//   skin           — chosen absolute skin thickness
//   max_neighbours — neighbour list capacity for the chosen skin
//   block_size     — chosen block size (also written to g_block_size)
//   ms_per_step    — measured GPU wall time for the winning configuration
//
// After this call the neighbour list is built with the chosen skin and
// g_skin, g_max_neighbours, g_block_size are all set. Call
// run_simulation immediately — no further setup needed.
//
// NOTE: tune_skin advances particles by N_SKIN_CANDIDATES * WARMUP_STEPS
// steps during the sweep. Re-upload (or re-initialise) particles before
// run_simulation if you need to start from exact t=0.
//
TuneResult tune_skin(Particle *particles, unsigned int n, double box_size) {

    // ── Lazy alloc + upload ───────────────────────────────────────────────────
    // If GPU memory has not been allocated yet (d_n == 0), do it now.
    // This makes tune_skin fully self-contained — the caller does not need
    // to run a dummy simulation or call internal setup functions first.
    if (d_n == 0 || d_n != n) {
        fprintf(stdout, "[TUNE] tune_skin: allocating GPU memory and uploading particles.\n");
        gpu_alloc(n, box_size);
        upload_particles(particles, n);
    }

    fprintf(stdout,
            "[TUNE] tune_skin: sweeping %d candidates x %d warmup steps\n",
            N_SKIN_CANDIDATES, WARMUP_STEPS);

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    const double       v_shift  = compute_v_shift();
    const unsigned int threads  = 256;
    const unsigned int blocks   = (n + threads - 1) / threads;

    double       best_time    = 1e30;
    double       best_skin    = SKIN_CANDIDATES[0] * R_CUT;
    unsigned int best_max_nn  = 0;
    unsigned int best_bs      = 64;   // tracked per-candidate, not fixed

    for (int c = 0; c < N_SKIN_CANDIDATES; ++c) {
        const double skin_abs = SKIN_CANDIDATES[c] * R_CUT;
        g_skin = skin_abs;

        // Recompute max_neighbours for this skin and reallocate list.
        {
            const double density  = (double)n / (box_size * box_size);
            const double expected = density * M_PI * R_LIST * R_LIST;
            const unsigned int computed =
                (unsigned int)(NEIGHBOUR_SAFETY_FACTOR * expected) + 1;
            g_max_neighbours = (computed > NEIGHBOUR_MIN) ? computed : NEIGHBOUR_MIN;
            CUDA_CHECK(cudaFree(d_nl));
            CUDA_CHECK(cudaMalloc(&d_nl,
                (size_t)n * g_max_neighbours * sizeof(int)));
        }

        gpu_build_neighbour_list(n, box_size);

        // Choose the block size that matches this skin's actual neighbour
        // count — skin, neighbour count, and block size are all connected.
        // Timing with a fixed block size would bias the measurement: a skin
        // that produces 44 neighbours timed with block=128 looks artificially
        // slow because 84 threads sit idle every pass.
        const unsigned int candidate_bs = tune_block_size(n);

        // Time WARMUP_STEPS full leapfrog steps with this candidate's
        // own block size so the measurement reflects the true combined cost.
        CUDA_CHECK(cudaEventRecord(ev_start));
        for (int s = 0; s < WARMUP_STEPS; ++s) {
            kernel_leapfrog_kick_drift<<<blocks, threads>>>(
                d_x, d_y, d_vx, d_vy, d_fx, d_fy, n, box_size);
            CUDA_CHECK(cudaDeviceSynchronize());

            if (gpu_needs_rebuild(n, box_size))
                gpu_build_neighbour_list(n, box_size);

            const size_t shm = 3 * candidate_bs * sizeof(double);
            kernel_compute_forces<<<n, candidate_bs, shm>>>(
                d_x, d_y, d_fx, d_fy, d_pe_arr,
                d_nl, d_nl_count, n, g_max_neighbours, box_size, v_shift);
            CUDA_CHECK(cudaDeviceSynchronize());

            kernel_leapfrog_kick<<<blocks, threads>>>(
                d_vx, d_vy, d_fx, d_fy, n);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        const double ms_per_step = (double)ms / WARMUP_STEPS;

        // Compute actual average neighbour count for logging.
        double avg_nn = 0.0;
        {
            unsigned int *h_count =
                (unsigned int *)malloc(n * sizeof(unsigned int));
            CUDA_CHECK(cudaMemcpy(h_count, d_nl_count,
                                  n * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost));
            unsigned long long tot = 0;
            for (unsigned int i = 0; i < n; ++i) tot += h_count[i];
            free(h_count);
            avg_nn = (double)tot / (double)n;
        }

        fprintf(stdout,
                "[TUNE]   skin=%.2f*R_CUT (abs=%.4f)  "
                "avg_nn=%.1f  max_nn=%u  block=%u  %.3f ms/step\n",
                SKIN_CANDIDATES[c], skin_abs,
                avg_nn, g_max_neighbours, candidate_bs, ms_per_step);

        if (ms_per_step < best_time) {
            best_time   = ms_per_step;
            best_skin   = skin_abs;
            best_max_nn = g_max_neighbours;
            best_bs     = candidate_bs;
        }
    }

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    // Apply the winning configuration — skin, max_neighbours, block size
    // are all set together because they were chosen as a unit.
    g_skin           = best_skin;
    g_max_neighbours = best_max_nn;
    g_block_size     = best_bs;

    CUDA_CHECK(cudaFree(d_nl));
    CUDA_CHECK(cudaMalloc(&d_nl,
        (size_t)n * g_max_neighbours * sizeof(int)));
    gpu_build_neighbour_list(n, box_size);

    fprintf(stdout,
            "[TUNE] tune_skin: selected skin=%.4f (%.2f*R_CUT)  "
            "max_neighbours=%u  block_size=%u  %.3f ms/step\n",
            best_skin, best_skin / R_CUT,
            best_max_nn, best_bs, best_time);

    TuneResult result;
    result.skin           = best_skin;
    result.max_neighbours = best_max_nn;
    result.block_size     = best_bs;
    result.ms_per_step    = best_time;
    return result;
}


SimulationResult run_simulation_gpu_v6(Particle *particles, unsigned int n,
                                unsigned int nsteps, double box_size,
                                int log_steps) {
    // Allocate all device memory up front.
    gpu_alloc(n, box_size);

    // Upload initial positions and velocities (one transfer, never repeated
    // unless log/GIF output requires a download mid-loop).
    upload_particles(particles, n);

    // Build initial neighbour list with current g_skin and g_block_size.
    // Call tune_skin() before run_simulation() to auto-select these values.
    gpu_build_neighbour_list(n, box_size);

    SimulationResult out;

    // Compute initial forces and PE with current parameters.
    out.start_potential = gpu_compute_forces(n, box_size);

    // KE uses velocities already on the host from initialize_particles —
    // no download needed here.
    out.start_kinetic = compute_ke(particles, n);
    out.start_total   = out.start_kinetic + out.start_potential;

#if GENERATE_GIF
    ge_GIF *gif = ge_new_gif(GIF_FILE, (uint16_t)FRAME_WIDTH,
                             (uint16_t)FRAME_HEIGHT, palette, 8, -1, 0);
    if (!gif)
        fprintf(stderr, "Warning: failed to create GIF output %s\n", GIF_FILE);
    else {
        render_frame_gif(gif, particles, n, box_size);
        ge_add_frame(gif, FRAME_DELAY);
    }
#endif

    for (unsigned int step = 0; step < nsteps; step++) {
        out.final_potential = gpu_leapfrog_step(n, box_size);

        // Download only when the host actually needs the data.
        // In a pure compute run (no logging, no GIF) this never executes
        // and the simulation runs with exactly two host-device transfers
        // total: the initial upload and the final download below.
        if (log_steps || (GENERATE_GIF && FRAME_EVERY > 0 &&
                          (step+1) % FRAME_EVERY == 0)) {
            download_particles(particles, n);
            out.final_kinetic = compute_ke(particles, n);
            out.final_total   = out.final_kinetic + out.final_potential;

            if (log_steps)
                printf("step=%6u  KE=%12.6f  PE=%12.6f  E=%12.6f\n",
                       step, out.final_kinetic,
                       out.final_potential, out.final_total);

#if GENERATE_GIF
            if (gif && FRAME_EVERY > 0 && (step+1) % FRAME_EVERY == 0) {
                render_frame_gif(gif, particles, n, box_size);
                ge_add_frame(gif, FRAME_DELAY);
            }
#endif
        }
    }

    // Final download: bring last state back to host for the caller.
    download_particles(particles, n);
    out.final_kinetic = compute_ke(particles, n);
    out.final_total   = out.final_kinetic + out.final_potential;

#if GENERATE_GIF
    if (gif) ge_close_gif(gif);
#endif

    out.n         = n;
    out.particles = particles;
    return out;
}

// 2 GPU version (naive)
// ─── KERNEL: O(N²) force computation ─────────────────────────────────────────
__global__ void kernel_forces_v4(
    const double * __restrict__ x_own,
    const double * __restrict__ y_own,
    const double * __restrict__ x_all,
    const double * __restrict__ y_all,
    double       * __restrict__ fx,
    double       * __restrict__ fy,
    double       * __restrict__ pe_out,
    unsigned int n_own,
    unsigned int n_total,
    unsigned int offset,
    double box_size,
    double v_shift)
{
    const unsigned int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= n_own) return;

    const unsigned int gi = offset + li;
    const double xi       = x_own[li];
    const double yi       = y_own[li];
    const double r_cut_sq = R_CUT * R_CUT;
    const double sig2     = SIGMA * SIGMA;
    double lfx = 0.0, lfy = 0.0, lpe = 0.0;

    for (unsigned int j = 0; j < n_total; ++j) {
        if (j == gi) continue;
        double dx = xi - x_all[j];
        double dy = yi - y_all[j];
        dx -= box_size * nearbyint(dx / box_size);
        dy -= box_size * nearbyint(dy / box_size);
        const double r2 = dx*dx + dy*dy;
        if (r2 >= r_cut_sq || r2 == 0.0) continue;
        const double inv_r2 = 1.0 / r2;
        const double sr2    = sig2 * inv_r2;
        const double sr6    = sr2 * sr2 * sr2;
        const double sr12   = sr6 * sr6;
        const double fij    = 24.0 * EPSILON * (2.0*sr12 - sr6) * inv_r2;
        lfx += fij * dx;
        lfy += fij * dy;
        lpe += 0.5 * (4.0*EPSILON*(sr12 - sr6) - v_shift);
    }
    fx[li]     = lfx;
    fy[li]     = lfy;
    pe_out[li] = lpe;
}

// ─── KERNEL: half-kick + drift + wrap ────────────────────────────────────────
__global__ void kernel_kick_drift_v4(
    double       * __restrict__ x,
    double       * __restrict__ y,
    double       * __restrict__ vx,
    double       * __restrict__ vy,
    const double * __restrict__ fx,
    const double * __restrict__ fy,
    unsigned int n, double box_size)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    vx[i] += 0.5 * DT * fx[i];
    vy[i] += 0.5 * DT * fy[i];
    x[i]  += DT * vx[i];
    y[i]  += DT * vy[i];
    double wx = fmod(x[i], box_size); if (wx < 0.0) wx += box_size; x[i] = wx;
    double wy = fmod(y[i], box_size); if (wy < 0.0) wy += box_size; y[i] = wy;
}

// ─── KERNEL: second half-kick ─────────────────────────────────────────────────
__global__ void kernel_kick_v4(
    double       * __restrict__ vx,
    double       * __restrict__ vy,
    const double * __restrict__ fx,
    const double * __restrict__ fy,
    unsigned int n)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    vx[i] += 0.5 * DT * fx[i];
    vy[i] += 0.5 * DT * fy[i];
}

// ─── Module: upload ───────────────────────────────────────────────────────────
//
// Scatter AoS host particles → SoA pinned buffers → device.
// Uploads positions and velocities for one GPU's slice.
//
static void v4_upload(
    int device_id,
    const Particle *particles,
    unsigned int offset, unsigned int n_own,
    double *h_x,  double *h_y,
    double *h_vx, double *h_vy,
    double *d_x,  double *d_y,
    double *d_vx, double *d_vy,
    double *d_x_all, double *d_y_all,
    const double *h_x_all, const double *h_y_all,
    size_t sz_own, size_t sz_all)
{
    for (unsigned int i = 0; i < n_own; ++i) {
        h_x[i]  = particles[offset+i].x;
        h_y[i]  = particles[offset+i].y;
        h_vx[i] = particles[offset+i].vx;
        h_vy[i] = particles[offset+i].vy;
    }
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaMemcpy(d_x,     h_x,     sz_own,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y,     h_y,     sz_own,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx,    h_vx,    sz_own,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy,    h_vy,    sz_own,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_all, h_x_all, sz_all,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_all, h_y_all, sz_all,  cudaMemcpyHostToDevice));
}

// ─── Module: download ─────────────────────────────────────────────────────────
//
// Download SoA device arrays → pinned host buffers → gather into AoS particles.
//
static void v4_download(
    int device_id,
    Particle *particles,
    unsigned int offset, unsigned int n_own,
    double *h_x,  double *h_y,
    double *h_vx, double *h_vy,
    double *h_fx, double *h_fy,
    double *d_x,  double *d_y,
    double *d_vx, double *d_vy,
    double *d_fx, double *d_fy,
    size_t sz_own)
{
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaMemcpy(h_x,  d_x,  sz_own, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_y,  d_y,  sz_own, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vx, d_vx, sz_own, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vy, d_vy, sz_own, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_fx, d_fx, sz_own, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_fy, d_fy, sz_own, cudaMemcpyDeviceToHost));
    for (unsigned int i = 0; i < n_own; ++i) {
        particles[offset+i].x  = h_x[i];  particles[offset+i].y  = h_y[i];
        particles[offset+i].vx = h_vx[i]; particles[offset+i].vy = h_vy[i];
        particles[offset+i].fx = h_fx[i]; particles[offset+i].fy = h_fy[i];
    }
}

// ─── Module: force computation + PE sum ──────────────────────────────────────
//
// Launch force kernel on one GPU, download pe_out, return summed PE.
//
static double v4_compute_forces(
    int device_id,
    double *d_x,  double *d_y,
    double *d_x_all, double *d_y_all,
    double *d_fx, double *d_fy,
    double *d_pe,
    unsigned int n_own, unsigned int n_total,
    unsigned int offset,
    double box_size, double v_shift,
    unsigned int threads)
{
    const unsigned int blocks = (n_own + threads - 1) / threads;
    CUDA_CHECK(cudaSetDevice(device_id));
    kernel_forces_v4<<<blocks, threads>>>(
        d_x, d_y, d_x_all, d_y_all,
        d_fx, d_fy, d_pe,
        n_own, n_total, offset,
        box_size, v_shift);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    double *h_pe = (double *)malloc(n_own * sizeof(double));
    CUDA_CHECK(cudaMemcpy(h_pe, d_pe,
                          n_own * sizeof(double), cudaMemcpyDeviceToHost));
    double pe = 0.0;
    for (unsigned int i = 0; i < n_own; ++i) pe += h_pe[i];
    free(h_pe);
    return pe;
}

// ─── Module: kick + drift ─────────────────────────────────────────────────────
static void v4_kick_drift(
    int device_id,
    double *d_x, double *d_y,
    double *d_vx, double *d_vy,
    double *d_fx, double *d_fy,
    unsigned int n_own, double box_size,
    unsigned int threads)
{
    const unsigned int blocks = (n_own + threads - 1) / threads;
    CUDA_CHECK(cudaSetDevice(device_id));
    kernel_kick_drift_v4<<<blocks, threads>>>(
        d_x, d_y, d_vx, d_vy, d_fx, d_fy, n_own, box_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ─── Module: second half-kick ─────────────────────────────────────────────────
static void v4_second_kick(
    int device_id,
    double *d_vx, double *d_vy,
    double *d_fx, double *d_fy,
    unsigned int n_own,
    unsigned int threads)
{
    const unsigned int blocks = (n_own + threads - 1) / threads;
    CUDA_CHECK(cudaSetDevice(device_id));
    kernel_kick_v4<<<blocks, threads>>>(d_vx, d_vy, d_fx, d_fy, n_own);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ─── Module: position exchange ───────────────────────────────────────────────
//
// After the drift each GPU has updated d_x/d_y for its own half.
// Synchronise d_x_all/d_y_all on both GPUs so all N positions are
// visible before force computation.
//
// P2P:    direct GPU-GPU cudaMemcpyPeer — no host involvement.
// Staged: download both halves to pinned host buffer, upload full array.
//
static void v4_exchange(
    int p2p,
    unsigned int n0, unsigned int n1, unsigned int off1,
    double *d0_x, double *d0_y, double *d0_x_all, double *d0_y_all,
    double *d1_x, double *d1_y, double *d1_x_all, double *d1_y_all,
    double *h_x_all, double *h_y_all,
    size_t sz0, size_t sz1, size_t sz_all)
{
    if (p2p) {
        // Refresh each GPU's own region in its x_all.
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMemcpy(d0_x_all,        d0_x, sz0, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d0_y_all,        d0_y, sz0, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaMemcpy(d1_x_all + off1, d1_x, sz1, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d1_y_all + off1, d1_y, sz1, cudaMemcpyDeviceToDevice));
        // Cross-copy.
        CUDA_CHECK(cudaMemcpyPeer(d1_x_all,         1, d0_x, 0, sz0));
        CUDA_CHECK(cudaMemcpyPeer(d1_y_all,         1, d0_y, 0, sz0));
        CUDA_CHECK(cudaMemcpyPeer(d0_x_all + off1,  0, d1_x, 1, sz1));
        CUDA_CHECK(cudaMemcpyPeer(d0_y_all + off1,  0, d1_y, 1, sz1));
    } else {
        // Stage through pinned host buffer.
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMemcpy(h_x_all,        d0_x, sz0, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_y_all,        d0_y, sz0, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaMemcpy(h_x_all + off1, d1_x, sz1, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_y_all + off1, d1_y, sz1, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMemcpy(d0_x_all, h_x_all, sz_all, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d0_y_all, h_y_all, sz_all, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaMemcpy(d1_x_all, h_x_all, sz_all, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d1_y_all, h_y_all, sz_all, cudaMemcpyHostToDevice));
    }
}

// ─── public API ───────────────────────────────────────────────────────────────
SimulationResult run_simulation_gpu_v7(Particle *particles, unsigned int n,
                                       unsigned int nsteps, double box_size,
                                       int log_steps) {

    // ── 1. GPU availability ───────────────────────────────────────────────────
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count < 2) {
        fprintf(stderr, "[v4] ERROR: need at least 2 CUDA devices, found %d.\n",
                device_count);
        exit(1);
    }
    fprintf(stdout, "[v4] Using devices 0 and 1 of %d available.\n",
            device_count);

    // ── 2. Peer-to-peer check ─────────────────────────────────────────────────
    int can_01 = 0, can_10 = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can_01, 0, 1));
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can_10, 1, 0));
    const int p2p = can_01 && can_10;
    if (p2p) {
        fprintf(stdout, "[v4] P2P available — direct GPU-GPU transfers.\n");
        CUDA_CHECK(cudaSetDevice(0)); CUDA_CHECK(cudaDeviceEnablePeerAccess(1, 0));
        CUDA_CHECK(cudaSetDevice(1)); CUDA_CHECK(cudaDeviceEnablePeerAccess(0, 0));
    } else {
        fprintf(stdout, "[v4] No P2P — host-staged transfers.\n");
    }

    // ── 3. Partition ──────────────────────────────────────────────────────────
    const unsigned int n0   = (n + 1) / 2;
    const unsigned int n1   = n - n0;
    const unsigned int off1 = n0;
    fprintf(stdout, "[v4] N=%u  GPU0=%u  GPU1=%u particles.\n", n, n0, n1);

    const size_t sz0    = n0 * sizeof(double);
    const size_t sz1    = n1 * sizeof(double);
    const size_t sz_all = n  * sizeof(double);
    const unsigned int threads = 256;

    // ── 4. Allocate device arrays (SoA) ───────────────────────────────────────
    double *d0_x, *d0_y, *d0_vx, *d0_vy, *d0_fx, *d0_fy, *d0_pe;
    double *d0_x_all, *d0_y_all;
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc(&d0_x,     sz0));   CUDA_CHECK(cudaMalloc(&d0_y,     sz0));
    CUDA_CHECK(cudaMalloc(&d0_vx,    sz0));   CUDA_CHECK(cudaMalloc(&d0_vy,    sz0));
    CUDA_CHECK(cudaMalloc(&d0_fx,    sz0));   CUDA_CHECK(cudaMalloc(&d0_fy,    sz0));
    CUDA_CHECK(cudaMalloc(&d0_pe,    sz0));
    CUDA_CHECK(cudaMalloc(&d0_x_all, sz_all)); CUDA_CHECK(cudaMalloc(&d0_y_all, sz_all));

    double *d1_x, *d1_y, *d1_vx, *d1_vy, *d1_fx, *d1_fy, *d1_pe;
    double *d1_x_all, *d1_y_all;
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaMalloc(&d1_x,     sz1));   CUDA_CHECK(cudaMalloc(&d1_y,     sz1));
    CUDA_CHECK(cudaMalloc(&d1_vx,    sz1));   CUDA_CHECK(cudaMalloc(&d1_vy,    sz1));
    CUDA_CHECK(cudaMalloc(&d1_fx,    sz1));   CUDA_CHECK(cudaMalloc(&d1_fy,    sz1));
    CUDA_CHECK(cudaMalloc(&d1_pe,    sz1));
    CUDA_CHECK(cudaMalloc(&d1_x_all, sz_all)); CUDA_CHECK(cudaMalloc(&d1_y_all, sz_all));

    // ── 5. Allocate pinned host buffers (SoA) ─────────────────────────────────
    double *h0_x, *h0_y, *h0_vx, *h0_vy, *h0_fx, *h0_fy;
    double *h1_x, *h1_y, *h1_vx, *h1_vy, *h1_fx, *h1_fy;
    double *h_x_all, *h_y_all;
    CUDA_CHECK(cudaMallocHost(&h0_x,    sz0));  CUDA_CHECK(cudaMallocHost(&h0_y,    sz0));
    CUDA_CHECK(cudaMallocHost(&h0_vx,   sz0));  CUDA_CHECK(cudaMallocHost(&h0_vy,   sz0));
    CUDA_CHECK(cudaMallocHost(&h0_fx,   sz0));  CUDA_CHECK(cudaMallocHost(&h0_fy,   sz0));
    CUDA_CHECK(cudaMallocHost(&h1_x,    sz1));  CUDA_CHECK(cudaMallocHost(&h1_y,    sz1));
    CUDA_CHECK(cudaMallocHost(&h1_vx,   sz1));  CUDA_CHECK(cudaMallocHost(&h1_vy,   sz1));
    CUDA_CHECK(cudaMallocHost(&h1_fx,   sz1));  CUDA_CHECK(cudaMallocHost(&h1_fy,   sz1));
    CUDA_CHECK(cudaMallocHost(&h_x_all, sz_all)); CUDA_CHECK(cudaMallocHost(&h_y_all, sz_all));

    // ── 6. Initial upload ─────────────────────────────────────────────────────
    // Build the full position staging buffer once — used by both GPUs.
    for (unsigned int i = 0; i < n; ++i) {
        h_x_all[i] = particles[i].x;
        h_y_all[i] = particles[i].y;
    }
    v4_upload(0, particles, 0,    n0, h0_x, h0_y, h0_vx, h0_vy,
              d0_x, d0_y, d0_vx, d0_vy, d0_x_all, d0_y_all,
              h_x_all, h_y_all, sz0, sz_all);
    v4_upload(1, particles, off1, n1, h1_x, h1_y, h1_vx, h1_vy,
              d1_x, d1_y, d1_vx, d1_vy, d1_x_all, d1_y_all,
              h_x_all, h_y_all, sz1, sz_all);

    // ── 7. Initial forces ─────────────────────────────────────────────────────
    const double v_shift = compute_v_shift();
    double pe_gpu[2] = {0.0, 0.0};

    #pragma omp parallel num_threads(2)
    {
        const int tid = omp_get_thread_num();
        if (tid == 0)
            pe_gpu[0] = v4_compute_forces(0, d0_x, d0_y, d0_x_all, d0_y_all,
                            d0_fx, d0_fy, d0_pe, n0, n, 0, box_size, v_shift, threads);
        else
            pe_gpu[1] = v4_compute_forces(1, d1_x, d1_y, d1_x_all, d1_y_all,
                            d1_fx, d1_fy, d1_pe, n1, n, off1, box_size, v_shift, threads);
    }

    SimulationResult out;
    out.start_potential = pe_gpu[0] + pe_gpu[1];
    out.start_kinetic   = compute_ke(particles, n);
    out.start_total     = out.start_kinetic + out.start_potential;

#if GENERATE_GIF
    ge_GIF *gif = ge_new_gif(GIF_FILE, (uint16_t)FRAME_WIDTH,
                             (uint16_t)FRAME_HEIGHT, palette, 8, -1, 0);
    if (!gif) fprintf(stderr, "Warning: failed to create GIF %s\n", GIF_FILE);
    else { render_frame_gif(gif, particles, n, box_size); ge_add_frame(gif, FRAME_DELAY); }
#endif

    // ── 8. Main loop ──────────────────────────────────────────────────────────
    for (unsigned int step = 0; step < nsteps; ++step) {

        // (a) Half-kick + drift — both GPUs simultaneously.
        #pragma omp parallel num_threads(2)
        {
            if (omp_get_thread_num() == 0)
                v4_kick_drift(0, d0_x, d0_y, d0_vx, d0_vy,
                              d0_fx, d0_fy, n0, box_size, threads);
            else
                v4_kick_drift(1, d1_x, d1_y, d1_vx, d1_vy,
                              d1_fx, d1_fy, n1, box_size, threads);
        }

        // (b) Position exchange — serial (both drifts complete after join).
        v4_exchange(p2p, n0, n1, off1,
                    d0_x, d0_y, d0_x_all, d0_y_all,
                    d1_x, d1_y, d1_x_all, d1_y_all,
                    h_x_all, h_y_all, sz0, sz1, sz_all);

        // (c) Force computation — both GPUs simultaneously.
        pe_gpu[0] = pe_gpu[1] = 0.0;
        #pragma omp parallel num_threads(2)
        {
            if (omp_get_thread_num() == 0)
                pe_gpu[0] = v4_compute_forces(0, d0_x, d0_y, d0_x_all, d0_y_all,
                                d0_fx, d0_fy, d0_pe, n0, n, 0,
                                box_size, v_shift, threads);
            else
                pe_gpu[1] = v4_compute_forces(1, d1_x, d1_y, d1_x_all, d1_y_all,
                                d1_fx, d1_fy, d1_pe, n1, n, off1,
                                box_size, v_shift, threads);
        }

        // (d) Second half-kick — both GPUs simultaneously.
        #pragma omp parallel num_threads(2)
        {
            if (omp_get_thread_num() == 0)
                v4_second_kick(0, d0_vx, d0_vy, d0_fx, d0_fy, n0, threads);
            else
                v4_second_kick(1, d1_vx, d1_vy, d1_fx, d1_fy, n1, threads);
        }

        out.final_potential = pe_gpu[0] + pe_gpu[1];

        // Download only when host needs the data.
        if (log_steps || (GENERATE_GIF && FRAME_EVERY > 0 &&
                          (step+1) % FRAME_EVERY == 0)) {
            v4_download(0, particles, 0,    n0,
                        h0_x, h0_y, h0_vx, h0_vy, h0_fx, h0_fy,
                        d0_x, d0_y, d0_vx, d0_vy, d0_fx, d0_fy, sz0);
            v4_download(1, particles, off1, n1,
                        h1_x, h1_y, h1_vx, h1_vy, h1_fx, h1_fy,
                        d1_x, d1_y, d1_vx, d1_vy, d1_fx, d1_fy, sz1);

            out.final_kinetic = compute_ke(particles, n);
            out.final_total   = out.final_kinetic + out.final_potential;

            if (log_steps)
                printf("step=%6u  KE=%12.6f  PE=%12.6f  E=%12.6f\n",
                       step, out.final_kinetic,
                       out.final_potential, out.final_total);
#if GENERATE_GIF
            if (gif && FRAME_EVERY > 0 && (step+1) % FRAME_EVERY == 0) {
                render_frame_gif(gif, particles, n, box_size);
                ge_add_frame(gif, FRAME_DELAY);
            }
#endif
        }
    }

    // ── 9. Final download ─────────────────────────────────────────────────────
    v4_download(0, particles, 0,    n0,
                h0_x, h0_y, h0_vx, h0_vy, h0_fx, h0_fy,
                d0_x, d0_y, d0_vx, d0_vy, d0_fx, d0_fy, sz0);
    v4_download(1, particles, off1, n1,
                h1_x, h1_y, h1_vx, h1_vy, h1_fx, h1_fy,
                d1_x, d1_y, d1_vx, d1_vy, d1_fx, d1_fy, sz1);
    out.final_kinetic = compute_ke(particles, n);
    out.final_total   = out.final_kinetic + out.final_potential;

#if GENERATE_GIF
    if (gif) ge_close_gif(gif);
#endif

    // ── 10. Cleanup ───────────────────────────────────────────────────────────
    if (p2p) {
        CUDA_CHECK(cudaSetDevice(0)); CUDA_CHECK(cudaDeviceDisablePeerAccess(1));
        CUDA_CHECK(cudaSetDevice(1)); CUDA_CHECK(cudaDeviceDisablePeerAccess(0));
    }
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(d0_x));     CUDA_CHECK(cudaFree(d0_y));
    CUDA_CHECK(cudaFree(d0_vx));    CUDA_CHECK(cudaFree(d0_vy));
    CUDA_CHECK(cudaFree(d0_fx));    CUDA_CHECK(cudaFree(d0_fy));
    CUDA_CHECK(cudaFree(d0_pe));
    CUDA_CHECK(cudaFree(d0_x_all)); CUDA_CHECK(cudaFree(d0_y_all));
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaFree(d1_x));     CUDA_CHECK(cudaFree(d1_y));
    CUDA_CHECK(cudaFree(d1_vx));    CUDA_CHECK(cudaFree(d1_vy));
    CUDA_CHECK(cudaFree(d1_fx));    CUDA_CHECK(cudaFree(d1_fy));
    CUDA_CHECK(cudaFree(d1_pe));
    CUDA_CHECK(cudaFree(d1_x_all)); CUDA_CHECK(cudaFree(d1_y_all));

    CUDA_CHECK(cudaFreeHost(h0_x));  CUDA_CHECK(cudaFreeHost(h0_y));
    CUDA_CHECK(cudaFreeHost(h0_vx)); CUDA_CHECK(cudaFreeHost(h0_vy));
    CUDA_CHECK(cudaFreeHost(h0_fx)); CUDA_CHECK(cudaFreeHost(h0_fy));
    CUDA_CHECK(cudaFreeHost(h1_x));  CUDA_CHECK(cudaFreeHost(h1_y));
    CUDA_CHECK(cudaFreeHost(h1_vx)); CUDA_CHECK(cudaFreeHost(h1_vy));
    CUDA_CHECK(cudaFreeHost(h1_fx)); CUDA_CHECK(cudaFreeHost(h1_fy));
    CUDA_CHECK(cudaFreeHost(h_x_all)); CUDA_CHECK(cudaFreeHost(h_y_all));

    out.n         = n;
    out.particles = particles;
    return out;
}