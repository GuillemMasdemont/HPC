#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "lennard-jones.h"

// ─── Benchmark configuration ──────────────────────────────────────────────────
static const unsigned int BENCH_N[]         = { 1000, 2000, 5000, 8000 };
static const int          N_BENCH           = 4;
static const int          MAX_RUNS          = 5;
static const int          MAX_RUNS_LARGE    = 1;
static const unsigned int LARGE_N_THRESHOLD = 5000;

static const unsigned int NSTEPS      = 5000;
static const double       DENSITY     = 0.95;
static const double       TEMPERATURE = 0.5;
static const unsigned int SEED        = 42;

// CSV output filename.
static const char *CSV_FILE = "resultscpu.csv";

// ─── Version table ────────────────────────────────────────────────────────────
typedef SimulationResult (*SimFn)(Particle *, unsigned int,
                                   unsigned int, double, int);
typedef struct {
    SimFn       fn;
    const char *label;           // human-readable label for stdout table
    const char *csv_label;       // short identifier for CSV header (no spaces/commas)
    int         tune_needed;     // 1 = call tune_skin before timing
    int         two_gpu;         // 1 = requires 2 GPUs, skipped in 1-GPU mode
    int         skip_in_two_gpu; // 1 = 1-GPU only, skipped when --two-gpu passed
} Version;

static Version VERSIONS[] = {
    //                                                         tune  two   skip_in
    //                                                         needed gpu   two_gpu
    { (SimFn)run_simulation_seq_v2, "seq_v2 (OpenMP CPU)",          "seq_v2", 0, 0, 0 },
    { (SimFn)run_simulation_v2,     "v2 (GPU 2D grid, CPU integ)",  "gpu_v2", 0, 0, 1 },
    { (SimFn)run_simulation_v3,     "v3 (GPU 1D pairs, CPU integ)", "gpu_v3", 0, 0, 1 },
    { (SimFn)run_simulation_v4,     "v4 (fully GPU)",               "gpu_v4", 0, 0, 1 },
    { (SimFn)run_simulation_v5,     "v5 (fully GPU, SoA + tiling)", "gpu_v5", 0, 0, 1 },
    { (SimFn)run_simulation_gpu_v6, "gpu_v6 (1 GPU + NL + block)",  "gpu_v6", 1, 0, 1 },
    { (SimFn)run_simulation_gpu_v7, "gpu_v7 (2 GPU naive)",         "gpu_v7", 0, 1, 0 },
};
static const int N_VERSIONS = 7;

// ─── helpers ──────────────────────────────────────────────────────────────────
static void print_help(const char *exe) {
    printf("Usage: %s [--two-gpu] [nsteps]\n", exe);
    printf("  --two-gpu  include gpu_v7 (2-GPU); requires 2 CUDA devices\n");
    printf("  nsteps     steps per run (default %u)\n", NSTEPS);
    printf("\nOutput: stdout table + %s\n", CSV_FILE);
}

static int reset_particles(Particle *particles, unsigned int n,
                            double box_size, double box_fraction) {
    return initialize_particles_seq_v2(
        particles, n, box_size, box_fraction, SEED, TEMPERATURE);
}

static double bench(const Version *ver,
                    Particle *particles, unsigned int n,
                    unsigned int nsteps, double box_size, double box_fraction,
                    int runs, SimulationResult *last_result) {
    double total = 0.0;
    for (int r = 0; r < runs; ++r) {
        if (!reset_particles(particles, n, box_size, box_fraction)) {
            fprintf(stderr, "Failed to reset particles (N=%u run %d).\n", n, r);
            return -1.0;
        }
        double t0 = omp_get_wtime();
        SimulationResult res = ver->fn(particles, n, nsteps, box_size, 0);
        double t1 = omp_get_wtime();
        total += (t1 - t0);
        if (r == runs - 1) *last_result = res;
        fprintf(stdout, "  [%s] run %d/%d  %.4f s\n",
                ver->label, r+1, runs, t1 - t0);
        fflush(stdout);
    }
    return total / (double)runs;
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char **argv) {
    unsigned int nsteps  = NSTEPS;
    int          two_gpu = 0;

    for (int a = 1; a < argc; ++a) {
        if (strcmp(argv[a], "-h") == 0 || strcmp(argv[a], "--help") == 0) {
            print_help(argv[0]);
            return 0;
        } else if (strcmp(argv[a], "--two-gpu") == 0) {
            two_gpu = 1;
        } else {
            nsteps = (unsigned int)strtoul(argv[a], NULL, 10);
        }
    }

    // ── Open CSV file ─────────────────────────────────────────────────────────
    FILE *csv = fopen(CSV_FILE, "w");
    if (!csv) {
        fprintf(stderr, "Warning: could not open %s for writing.\n", CSV_FILE);
    }

    // CSV header:
    //   N, runs, nsteps, density, temperature,
    //   <version>_time_s, <version>_speedup   (one pair per version)
    //   tune_skin, tune_max_nn, tune_block     (for gpu_v6)
    if (csv) {
        fprintf(csv, "N,runs,nsteps,density,temperature");
        for (int v = 0; v < N_VERSIONS; ++v)
            fprintf(csv, ",%s_time_s,%s_speedup", VERSIONS[v].csv_label,
                    VERSIONS[v].csv_label);
        fprintf(csv, ",tune_skin,tune_max_nn,tune_block\n");
    }

    printf("==========================================================\n");
    printf("  Lennard-Jones Speedup Benchmark\n");
    printf("  nsteps      : %u\n", nsteps);
    printf("  density     : %.2f\n", DENSITY);
    printf("  temperature : %.2f\n", TEMPERATURE);
    printf("  baseline    : %s\n", VERSIONS[0].label);
    printf("  mode        : %s\n", two_gpu ? "2-GPU (gpu_v7 included)"
                                           : "1-GPU (gpu_v7 skipped)");
    printf("  csv output  : %s\n", CSV_FILE);
    printf("==========================================================\n\n");

    // Stdout table header.
    printf("%-8s  %-6s  %-14s", "N", "runs", VERSIONS[0].label);
    for (int v = 1; v < N_VERSIONS; ++v)
        printf("  %-14s  %-10s", VERSIONS[v].label, "speedup");
    printf("\n");
    printf("%-8s  %-6s  %-14s", "--------", "------", "--------------");
    for (int v = 1; v < N_VERSIONS; ++v)
        printf("  %-14s  %-10s", "--------------", "----------");
    printf("\n");

    for (int bi = 0; bi < N_BENCH; ++bi) {
        const unsigned int n    = BENCH_N[bi];
        const int          runs = (n >= LARGE_N_THRESHOLD)
                                  ? MAX_RUNS_LARGE : MAX_RUNS;

        double particle_box_size = ceil(sqrt((double)n / DENSITY));
        double box_size          = (4.0 / 3.0) * particle_box_size;
        double box_fraction      = particle_box_size / box_size;

        Particle *particles = (Particle *)calloc(n, sizeof(Particle));
        if (!particles) {
            fprintf(stderr, "Failed to allocate particles for N=%u.\n", n);
            if (csv) fclose(csv);
            return 1;
        }

        printf("\n── N = %u  (%d run%s each) ──────────────────────────\n",
               n, runs, runs > 1 ? "s" : "");

        // ── Pre-tune ─────────────────────────────────────────────────────────
        // Only needed in 1-GPU mode where gpu_v6 uses the NL tuning API.
        // In 2-GPU mode only seq_v2 and gpu_v7 run — neither needs tuning.
        if (!two_gpu) {
            printf("  [pre-tune] allocating GPU and setting NL params for N=%u...\n", n);
            if (!reset_particles(particles, n, box_size, box_fraction)) {
                fprintf(stderr, "Failed to reset for pre-tune N=%u.\n", n);
                free(particles); if (csv) fclose(csv); return 1;
            }
            tune_skin(particles, n, box_size);
            printf("\n");
        }

        // ── Run all versions ──────────────────────────────────────────────────
        double times[N_VERSIONS];
        SimulationResult results[N_VERSIONS];
        TuneResult tune = {0};
        for (int v = 0; v < N_VERSIONS; ++v) times[v] = -1.0;

        for (int v = 0; v < N_VERSIONS-1; ++v) {
            // Skip 2-GPU version in 1-GPU mode.
            if (VERSIONS[v].two_gpu && !two_gpu) {
                printf("  [%s] skipped (run with --two-gpu to include)\n\n",
                       VERSIONS[v].label);
                continue;
            }
            // Skip 1-GPU versions in 2-GPU mode — only seq_v2 and gpu_v7 run.
            if (VERSIONS[v].skip_in_two_gpu && two_gpu) {
                printf("  [%s] skipped (1-GPU only, not needed in 2-GPU mode)\n\n",
                       VERSIONS[v].label);
                continue;
            }

            if (VERSIONS[v].tune_needed) {
                printf("  [tune] re-tuning for %s...\n", VERSIONS[v].label);
                if (!reset_particles(particles, n, box_size, box_fraction)) {
                    fprintf(stderr, "Failed to reset for tuning N=%u.\n", n);
                    free(particles); if (csv) fclose(csv); return 1;
                }
                tune = tune_skin(particles, n, box_size);
                printf("  [tune] skin=%.4f  max_nn=%u  block=%u  %.3f ms/step\n\n",
                       tune.skin, tune.max_neighbours,
                       tune.block_size, tune.ms_per_step);
            }

            printf("  Benchmarking %s...\n", VERSIONS[v].label);
            times[v] = bench(&VERSIONS[v], particles, n, nsteps,
                              box_size, box_fraction, runs, &results[v]);
            if (times[v] < 0.0) {
                free(particles); if (csv) fclose(csv); return 1;
            }

            const double drift =
                fabs(results[v].final_total - results[v].start_total)
                / fabs(results[v].start_total);
            if (drift > 0.05)
                printf("  WARNING: %s energy drift %.2f%%\n",
                       VERSIONS[v].label, drift * 100.0);
            printf("\n");
        }

        // ── Compute baseline and speedups ─────────────────────────────────────
        double t_baseline = -1.0;
        for (int v = 0; v < N_VERSIONS; ++v) {
            if (times[v] >= 0.0) { t_baseline = times[v]; break; }
        }

        // ── Stdout result row ─────────────────────────────────────────────────
        printf("  %-8u  %-6d", n, runs);
        for (int v = 0; v < N_VERSIONS; ++v) {
            if (times[v] < 0.0) {
                printf("  %-14s  %-10s", "skipped", "-");
            } else if (times[v] == t_baseline) {
                printf("  %-14.4f  %-10s", times[v], "(baseline)");
            } else {
                printf("  %-14.4f  %-10.2f", times[v], t_baseline / times[v]);
            }
        }
        printf("\n");

        // ── CSV result row ────────────────────────────────────────────────────
        // Columns: N, runs, nsteps, density, temperature,
        //          then for each version: time_s, speedup (empty if skipped),
        //          then tune_skin, tune_max_nn, tune_block (gpu_v6 params).
        if (csv) {
            fprintf(csv, "%u,%d,%u,%.4f,%.4f",
                    n, runs, nsteps, DENSITY, TEMPERATURE);
            for (int v = 0; v < N_VERSIONS; ++v) {
                if (times[v] < 0.0) {
                    // skipped — empty cells so column count stays consistent
                    fprintf(csv, ",,");
                } else if (times[v] == t_baseline) {
                    // baseline — speedup is 1.0 by definition
                    fprintf(csv, ",%.6f,1.000000", times[v]);
                } else {
                    fprintf(csv, ",%.6f,%.6f",
                            times[v], t_baseline / times[v]);
                }
            }
            // gpu_v6 tuning params (empty if tune was never called for this N)
            if (tune.skin > 0.0)
                fprintf(csv, ",%.4f,%u,%u",
                        tune.skin, tune.max_neighbours, tune.block_size);
            else
                fprintf(csv, ",,,");
            fprintf(csv, "\n");
            fflush(csv);   // flush after each N so partial results are saved
        }

        free(particles);
    }

    printf("\n==========================================================\n");
    printf("  Legend:\n");
    printf("    [0] %s — baseline\n", VERSIONS[0].label);
    for (int v = 1; v < N_VERSIONS; ++v)
        printf("    [%d] %s\n", v, VERSIONS[v].label);
    printf("  CSV saved to: %s\n", CSV_FILE);
    printf("==========================================================\n");

    if (csv) fclose(csv);
    return 0;
}