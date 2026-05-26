#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lenia.h"

#define DEFAULT_N     128
#define DEFAULT_STEPS 100
#define DEFAULT_RUNS  3
#define DT            0.1
#define KERNEL_SIZE   26

int main(int argc, char *argv[])
{
    int myid, procs;
    char node_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    MPI_Get_processor_name(node_name, &name_len);

    /* --- Parse command-line arguments ------------------------------------ */
    int    N            = DEFAULT_N;
    int    num_steps    = DEFAULT_STEPS;
    int    num_runs     = DEFAULT_RUNS;
    int    generate_gif = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--gif") == 0) {
            generate_gif = 1;
        } else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            num_runs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            num_steps = atoi(argv[++i]);
        } else {
            int val = atoi(argv[i]);
            if (val > 0) N = val;
        }
    }

    /* GIF generation is a one-shot operation; no point averaging */
    if (generate_gif) num_runs = 1;

    /* Orbium placement (relative to grid size) */
    struct orbium_coo orbiums[] = {
        {0,     N / 3, 0  },
        {N / 3, 0,     180}
    };
    int num_orbiums = 2;

    if (myid == 0) {
        printf("Grid: %dx%d | Steps: %d | Processes: %d | Runs: %d%s\n",
               N, N, num_steps, procs, num_runs,
               generate_gif ? " | GIF: yes" : "");
    }

    /* --- Benchmark loop -------------------------------------------------- */
    double total_time = 0.0;

    for (int run = 0; run < num_runs; run++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        double *world = evolve_lenia_mpi(
            (unsigned int)N, (unsigned int)N,
            (unsigned int)num_steps, DT,
            KERNEL_SIZE,
            orbiums, (unsigned int)num_orbiums,
            generate_gif);

        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        if (myid == 0) {
            total_time += t1 - t0;
            free(world);  /* NULL on non-root is fine to free() */
        }
    }

    if (myid == 0) {
        double avg = total_time / num_runs;
        printf("Average time (%d run%s): %.4f s\n",
               num_runs, num_runs > 1 ? "s" : "", avg);
        /* CSV line for easy post-processing: N,procs,avg_time */
        printf("CSV,%d,%d,%.4f\n", N, procs, avg);
    }

    MPI_Finalize();
    return 0;
}
