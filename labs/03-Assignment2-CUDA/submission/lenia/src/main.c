#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <cuda_runtime.h>
#include "lenia.h"

#define NUM_STEPS 100
#define DT 0.1
#define KERNEL_SIZE 26 
#define NUM_ORBIUMS 2
#define NUM_RUNS 5

int main()
{
    FILE *fp = fopen("benchmark_results.txt", "w");
    if (fp == NULL) {
        return EXIT_FAILURE; 
    }

    fprintf(fp, "N,T_seq_v2,T_naive,T_v1,T_v2,T_v3(f),T_v5(f),T_v6(2gpu),Speedup(v3_vs_seq_v2),Speedup(v5_vs_seq_v2),Speedup(v6_vs_seq_v2)\n");

    int num_visible_gpus = 0;
    cudaError_t dev_count_err = cudaGetDeviceCount(&num_visible_gpus);
    if (dev_count_err != cudaSuccess) {
        num_visible_gpus = 0;
    }

    int run_single_versions = 1;
    int run_v6 = (num_visible_gpus >= 2);
    const char *bench_mode = getenv("LENIA_BENCH_MODE");

    if (bench_mode != NULL)
    {
        if (strcmp(bench_mode, "single") == 0)
        {
            run_single_versions = 1;
            run_v6 = 0;
        }
        else if (strcmp(bench_mode, "v6") == 0)
        {
            run_single_versions = 0;
            run_v6 = (num_visible_gpus >= 2);
            if (!run_v6)
            {
                fprintf(stderr, "[main] LENIA_BENCH_MODE=v6 but <2 visible GPUs.\n");
                fclose(fp);
                return EXIT_FAILURE;
            }
        }
        else
        {
            fprintf(stderr, "[main] Unknown LENIA_BENCH_MODE=%s, using full benchmark.\n", bench_mode);
        }
    }

    if (!run_v6)
    {
        fprintf(stderr, "[main] v6 (dual-GPU) benchmark will be skipped.\n");
    }

    int cpu_baseline_threads = 32;
    const char *cpu_threads_env = getenv("LENIA_CPU_THREADS");
    if (cpu_threads_env != NULL)
    {
        int parsed_threads = atoi(cpu_threads_env);
        if (parsed_threads > 0)
        {
            cpu_baseline_threads = parsed_threads;
        }
    }

    omp_set_dynamic(0);
    omp_set_num_threads(cpu_baseline_threads);
    int effective_threads = omp_get_max_threads();
    fprintf(stderr, "[main] CPU baseline: evolve_lenia_seq_v2 with %d OpenMP threads.\n", effective_threads);
    if (effective_threads != cpu_baseline_threads)
    {
        fprintf(stderr, "[main] Warning: requested %d CPU threads, runtime reports %d.\n",
                cpu_baseline_threads, effective_threads);
    }

    int grid_sizes[] = {128, 256, 512, 1024, 2048, 4096};
    int num_sizes = sizeof(grid_sizes) / sizeof(grid_sizes[0]);

    for (int i = 0; i < num_sizes; i++) {
        int N = grid_sizes[i];

        // Place two orbiums in the world with different angles. (y, x, angle)
        // Orbiums size is 20x20, supproted angles are 0, 90, 180 and 270 degrees.
        struct orbium_coo orbiums[NUM_ORBIUMS] = {{0, N / 3, 0}, {N / 3, 0, 180}};
        
        double total_ts_v2 = 0, total_tnaive = 0, total_tv1 = 0, total_tv2 = 0, total_tv3 = 0, total_tv5 = 0, total_tv6 = 0;
        int v6_runs = 0;
        int current_runs = (N >= 4096) ? 1 : NUM_RUNS;

        for (int r = 0; r < current_runs; r++) {

            double *world_s_v2 = NULL;
            double *world_naive = NULL;
            double *world_v1 = NULL;
            double *world_v2 = NULL;
            float *world_v3 = NULL;
            float *world_v5 = NULL;
            double *world_v6 = NULL;

            if (run_single_versions)
            {
                // 1. CPU Sequential V2 (baseline)
                double start_s_v2 = omp_get_wtime();
                world_s_v2 = evolve_lenia_seq_v2(N, N, NUM_STEPS, DT, KERNEL_SIZE, orbiums, NUM_ORBIUMS);
                total_ts_v2 += (omp_get_wtime() - start_s_v2);
                
                // 2. GPU - Naive CUDA
                double start_naive = omp_get_wtime();
                world_naive = evolve_lenia_cuda(N, N, NUM_STEPS, DT, KERNEL_SIZE, orbiums, NUM_ORBIUMS);
                total_tnaive += (omp_get_wtime() - start_naive);

                // 3. GPU - V1 (Shared Memory)
                double start_v1 = omp_get_wtime();
                world_v1 = evolve_lenia_v1(N, N, NUM_STEPS, DT, KERNEL_SIZE, orbiums, NUM_ORBIUMS);
                total_tv1 += (omp_get_wtime() - start_v1);

                // 4. GPU - V2 (Optimized Double)
                double start_v2 = omp_get_wtime();
                world_v2 = evolve_lenia_v2(N, N, NUM_STEPS, DT, KERNEL_SIZE, orbiums, NUM_ORBIUMS);
                total_tv2 += (omp_get_wtime() - start_v2);

                // 5. GPU - V3 (Optimized Float)
                // Note: DT is cast to float, and it returns a float*
                double start_v3 = omp_get_wtime();
                world_v3 = evolve_lenia_v3(N, N, NUM_STEPS, (float)DT, KERNEL_SIZE, orbiums, NUM_ORBIUMS);
                total_tv3 += (omp_get_wtime() - start_v3);

                // 6. GPU - V5 (Block-shape experiment, Optimized Float)
                double start_v5 = omp_get_wtime();
                world_v5 = evolve_lenia_v5(N, N, NUM_STEPS, (float)DT, KERNEL_SIZE, orbiums, NUM_ORBIUMS);
                double elapsed_v5 = (omp_get_wtime() - start_v5);
                double autotune_v5 = lenia_v5_get_last_autotune_seconds();
                double net_v5 = elapsed_v5 - autotune_v5;
                if (net_v5 < 0.0)
                {
                    net_v5 = 0.0;
                }
                total_tv5 += net_v5;
            }

            // 7. GPU - V6 (Dual-GPU), only when two GPUs are visible
            if (run_v6) {
                double start_v6 = omp_get_wtime();
                world_v6 = evolve_lenia_v6(N, N, NUM_STEPS, DT, KERNEL_SIZE, orbiums, NUM_ORBIUMS);
                if (world_v6 != NULL) {
                    total_tv6 += (omp_get_wtime() - start_v6);
                    v6_runs++;
                }
            }

            // Cleanup
            free(world_s_v2);
            free(world_naive);
            free(world_v1);
            free(world_v2);
            free(world_v3);
            free(world_v5);
            free(world_v6);
        }

        double avg_ts_v2 = run_single_versions ? (total_ts_v2 / current_runs) : -1.0;
        double avg_tnaive = run_single_versions ? (total_tnaive / current_runs) : -1.0;
        double avg_tv1 = run_single_versions ? (total_tv1 / current_runs) : -1.0;
        double avg_tv2 = run_single_versions ? (total_tv2 / current_runs) : -1.0;
        double avg_tv3 = run_single_versions ? (total_tv3 / current_runs) : -1.0;
        double avg_tv5 = run_single_versions ? (total_tv5 / current_runs) : -1.0;
        double avg_tv6 = (v6_runs > 0) ? (total_tv6 / v6_runs) : -1.0;
        
        // Speedups use the improved CPU baseline (seq_v2, OpenMP).
        double speedup_naive = (avg_ts_v2 > 0.0 && avg_tnaive > 0.0) ? (avg_ts_v2 / avg_tnaive) : -1.0;
        double speedup_v2 = (avg_ts_v2 > 0.0 && avg_tv2 > 0.0) ? (avg_ts_v2 / avg_tv2) : -1.0;
        double speedup_v3 = (avg_ts_v2 > 0.0 && avg_tv3 > 0.0) ? (avg_ts_v2 / avg_tv3) : -1.0;
        double speedup_v5 = (avg_ts_v2 > 0.0 && avg_tv5 > 0.0) ? (avg_ts_v2 / avg_tv5) : -1.0;
        double speedup_v6 = (avg_ts_v2 > 0.0 && avg_tv6 > 0.0) ? (avg_ts_v2 / avg_tv6) : -1.0;

        // Write the comma-separated values to the .txt file
        fprintf(fp, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.2f,%.2f,%.2f,%.2f,%.2f\n", 
            N, avg_ts_v2, avg_tnaive, avg_tv1, avg_tv2, avg_tv3, avg_tv5, avg_tv6, speedup_naive, speedup_v2,
            speedup_v3, speedup_v5, speedup_v6);
    }

    // Close the file 
    fclose(fp);

    return 0;
}