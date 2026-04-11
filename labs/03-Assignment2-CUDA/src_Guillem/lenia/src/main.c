#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "lenia.h"

#define NUM_STEPS 100
#define DT 0.1
#define KERNEL_SIZE 26 // MAKE SURE MAX_KERNEL_SIZE in .cu is >= 26!
#define NUM_ORBIUMS 2
#define NUM_RUNS 5

int main()
{
    // Open a .txt file for writing
    FILE *fp = fopen("benchmark_results.txt", "w");
    if (fp == NULL) {
        return EXIT_FAILURE; 
    }

    // Write the CSV header line to the file
    fprintf(fp, "N,T_seq,T_naive,T_v1,T_v2,T_v3(f),Speedup(v3)\n");

    int grid_sizes[] = {128, 256, 512, 1024, 2048, 4096};
    int num_sizes = sizeof(grid_sizes) / sizeof(grid_sizes[0]);

    for (int i = 0; i < num_sizes; i++) {
        int N = grid_sizes[i];

        // Place two orbiums in the world with different angles. (y, x, angle)
        // Orbiums size is 20x20, supproted angles are 0, 90, 180 and 270 degrees.
        struct orbium_coo orbiums[NUM_ORBIUMS] = {{0, N / 3, 0}, {N / 3, 0, 180}};
        
        double total_ts = 0, total_tnaive = 0, total_tv1 = 0, total_tv2 = 0, total_tv3 = 0;
        int current_runs = (N >= 4096) ? 1 : NUM_RUNS;

        for (int r = 0; r < current_runs; r++) {

            // 1. CPU Sequential
            double start_s = omp_get_wtime();
            double *world_s = evolve_lenia_seq(N, N, NUM_STEPS, DT, KERNEL_SIZE, orbiums, NUM_ORBIUMS);
            total_ts += (omp_get_wtime() - start_s);
            
            // 2. GPU - Naive CUDA
            double start_naive = omp_get_wtime();
            double *world_naive = evolve_lenia_cuda(N, N, NUM_STEPS, DT, KERNEL_SIZE, orbiums, NUM_ORBIUMS);
            total_tnaive += (omp_get_wtime() - start_naive);

            // 3. GPU - V1 (Shared Memory)
            double start_v1 = omp_get_wtime();
            double *world_v1 = evolve_lenia_v1(N, N, NUM_STEPS, DT, KERNEL_SIZE, orbiums, NUM_ORBIUMS);
            total_tv1 += (omp_get_wtime() - start_v1);

            // 4. GPU - V2 (Optimized Double)
            double start_v2 = omp_get_wtime();
            double *world_v2 = evolve_lenia_v2(N, N, NUM_STEPS, DT, KERNEL_SIZE, orbiums, NUM_ORBIUMS);
            total_tv2 += (omp_get_wtime() - start_v2);

            // 5. GPU - V3 (Optimized Float)
            // Note: DT is cast to float, and it returns a float*
            double start_v3 = omp_get_wtime();
            float *world_v3 = evolve_lenia_v3(N, N, NUM_STEPS, (float)DT, KERNEL_SIZE, orbiums, NUM_ORBIUMS);
            total_tv3 += (omp_get_wtime() - start_v3);

            // Cleanup
            free(world_s);
            free(world_naive);
            free(world_v1);
            free(world_v2);
            free(world_v3);
        }

        double avg_ts = total_ts / current_runs;
        double avg_tnaive = total_tnaive / current_runs;
        double avg_tv1 = total_tv1 / current_runs;
        double avg_tv2 = total_tv2 / current_runs;
        double avg_tv3 = total_tv3 / current_runs;
        
        // Let's calculate the final speedup comparing the CPU to the fastest GPU version
        double speedup = avg_ts / avg_tv3;

        // Write the comma-separated values to the .txt file
        fprintf(fp, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.2f\n", 
               N, avg_ts, avg_tnaive, avg_tv1, avg_tv2, avg_tv3, speedup);
    }

    // Close the file 
    fclose(fp);

    return 0;
}