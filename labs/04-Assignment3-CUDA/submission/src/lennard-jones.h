#ifndef LJ_H
#define LJ_H

#ifdef __cplusplus
extern "C" {
#endif

#define DT 0.002
#define SIGMA 1.0
#define EPSILON 1.0
#define R_CUT 2.5
#define JITTER 0.05

#define GENERATE_GIF 0
#define FRAME_WIDTH 800
#define FRAME_HEIGHT 800
#define FRAME_EVERY 5
#define FRAME_PARTICLE_RADIUS 2
#define FRAME_DELAY 3
#define GIF_FILE "simulation.gif"


typedef struct {
    double       skin;           // chosen absolute skin thickness
    unsigned int max_neighbours; // neighbour list capacity for this skin
    unsigned int block_size;     // chosen CUDA block size for force kernel
    double       ms_per_step;    // measured GPU wall time for this config
} TuneResult;

typedef struct {
    double x;
    double y;
    double vx;
    double vy;
    double fx;
    double fy;
} Particle;

// Structure of Arrays layout used by v5 — enables coalesced GPU memory access
typedef struct {
    double *x,  *y;
    double *vx, *vy;
    double *fx, *fy;
} ParticlesSoA;

typedef struct {
    unsigned int n;
    const Particle *particles;
    double start_kinetic;
    double start_potential;
    double start_total;
    double final_kinetic;
    double final_potential;
    double final_total;
} SimulationResult;

int initialize_particles(
    Particle *particles,
    unsigned int n,
    double box_size,
    double placement_fraction,
    unsigned int seed,
    double temperature
);
void wrap_positions(Particle *particles, unsigned int n, double box_size);

double compute_v_shift(void);
double compute_forces(
    Particle *particles,
    unsigned int n,
    double box_size
);
double leapfrog_step(
    Particle *particles,
    unsigned int n,
    double box_size
);
SimulationResult run_simulation(Particle *particles, unsigned int n, unsigned int nsteps, double box_size, int log_steps);

// Sequential versions of the above functions, which will be used for comparison with the parallelized versions in the CUDA implementation

int initialize_particles_seq_v2(
    Particle *particles,
    unsigned int n,
    double box_size,
    double placement_fraction,
    unsigned int seed,
    double temperature
);
void wrap_positions_seq_v2(Particle *particles, unsigned int n, double box_size);

double compute_forces_seq_v2(
    Particle *particles,
    unsigned int n,
    double box_size
);
double leapfrog_step_seq_v2(
    Particle *particles,
    unsigned int n,
    double box_size
);

SimulationResult run_simulation_seq_v2(Particle *particles, unsigned int n, unsigned int nsteps, double box_size, int log_steps);


// =============================================================================
// VERSION 2 — GPU forces (2D grid kernel), CPU integration
// Forces on GPU via a 2D (i,j) grid — each pair computed twice.
// Integration on CPU. GPU memory allocated/freed every step.
// =============================================================================

double           compute_forces_gpu_v2 (Particle *particles, unsigned int n, double box_size);
double           leapfrog_step_v2      (Particle *particles, unsigned int n, double box_size);
SimulationResult run_simulation_v2     (Particle *particles, unsigned int n, unsigned int nsteps,
                                        double box_size, int log_steps);

// =============================================================================
// VERSION 3 — GPU forces (1D pair kernel), CPU integration
// Forces on GPU via a 1D kernel over unique pairs — Newton's 3rd law applied.
// Integration on CPU. GPU memory allocated/freed every step.
// =============================================================================

double           compute_forces_gpu_v3 (Particle *particles, unsigned int n, double box_size);
double           leapfrog_step_v3      (Particle *particles, unsigned int n, double box_size);
SimulationResult run_simulation_v3     (Particle *particles, unsigned int n, unsigned int nsteps,
                                        double box_size, int log_steps);

// =============================================================================
// VERSION 4 — Fully GPU
// All steps (forces, integration, KE) run on GPU.
// GPU memory allocated once before the loop, freed once after.
// =============================================================================

double           leapfrog_step_v4   (Particle *d_particles, unsigned int n,
                                     double box_size, double *d_pe);
SimulationResult run_simulation_v4  (Particle *particles, unsigned int n, unsigned int nsteps,
                                     double box_size, int log_steps);

// =============================================================================
// VERSION 5 — Fully GPU, SoA layout, shared memory tiling, if/else wrap
// SoA device arrays for coalesced reads, tiled shared memory in force kernel
// to reduce global memory traffic ~TILE_SIZE, if/else PBC wrap (no division).
// =============================================================================

double           leapfrog_step_v5   (ParticlesSoA d, unsigned int n,
                                     double box_size, double *d_pe);
SimulationResult run_simulation_v5  (Particle *particles, unsigned int n, unsigned int nsteps,
                                     double box_size, int log_steps);


// Neighbor list + block of threads (version 6)

TuneResult tune_skin(Particle *particles, unsigned int n, double box_size);

unsigned int tune_block_size(unsigned int n);

SimulationResult run_simulation_gpu_v6(Particle *particles, unsigned int n, unsigned int nsteps, double box_size, int log_steps);

// Version 7: 2 GPUs

SimulationResult run_simulation_gpu_v7(Particle *particles, unsigned int n,
                                       unsigned int nsteps, double box_size,
                                       int log_steps);

#ifdef __cplusplus
}
#endif

#endif
