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
    unsigned int id;
    double x;
    double y;
    double z;
    double vx;
    double vy;
    double vz;
    double fx;
    double fy;
    double fz;
} Particle;

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

typedef struct {
    double       skin;
    unsigned int max_neighbours;
    unsigned int block_size;
    double       ms_per_step;
} TuneResult;


TuneResult tune_skin_3d(Particle *particles, unsigned int n, double box_size);
 
// Re-tune block size only (without re-sweeping skin).
unsigned int tune_block_size_3d(unsigned int n);

// Offline sweep: tunes skin for N=n_min..n_max step n_step and prints a C table to stdout.
// Paste the output into lennard-jones.cu to eliminate runtime tuning.
void tune_sweep_and_print(unsigned int n_min, unsigned int n_max, unsigned int n_step,
                           double density, double temperature, unsigned int seed);

// GPU simulation: neighbour list + block-per-particle (3D).
SimulationResult run_simulation_gpu_v6_3d(
    Particle  *particles,
    unsigned int n,
    unsigned int nsteps,
    double       box_size,
    int          log_steps);

SimulationResult run_simulation_gpu_v7_3d(Particle *particles, unsigned int n,
                                           unsigned int nsteps, double box_size,
                                           int log_steps);

SimulationResult run_simulation_gpu_v8_3d(Particle *particles, unsigned int n,
                                           unsigned int nsteps, double box_size,
                                           int log_steps);

SimulationResult run_simulation_gpu_v9_3d(Particle *particles, unsigned int n,
                                           unsigned int nsteps, double box_size,
                                           int log_steps);

#ifdef __cplusplus
}
#endif

#endif
