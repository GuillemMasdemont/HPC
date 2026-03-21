#!/bin/bash

#SBATCH --account=fri-users
#SBATCH --reservation=fri
#SBATCH --job-name=parallel_seam_carving
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=sample_out.log
#SBATCH --hint=nomultithread

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Compile
gcc -O3 -fopenmp parallel_seam_carving.c -lm -o parallel_seam_carving

# Run
srun ./parallel_seam_carving --benchmark ../test_images 128 5 benchmark_results.csv

# Try these combinations on the supercomputer:
#
# 1) Benchmark with 1 CPU per task
# srun --ntasks=1 --cpus-per-task=1  --hint=nomultithread ./parallel_seam_carving --benchmark ../test_images 128 5 benchmark_1thread.csv
#
# 2) Benchmark with 4 CPUs per task
# srun --ntasks=1 --cpus-per-task=4  --hint=nomultithread ./parallel_seam_carving --benchmark ../test_images 128 5 benchmark_4threads.csv
#
# 3) Benchmark with 8 CPUs per task
# srun --ntasks=1 --cpus-per-task=8  --hint=nomultithread ./parallel_seam_carving --benchmark ../test_images 128 5 benchmark_8threads.csv
#
# 4) Single image, triangular parallel
# srun --ntasks=1 --cpus-per-task=8  --hint=nomultithread ./parallel_seam_carving ../test_images/720x480.png output_triangular.png 128 8 triangular_parallel
#
# 5) Single image, greedy parallel
# srun --ntasks=1 --cpus-per-task=8  --hint=nomultithread ./parallel_seam_carving ../test_images/720x480.png output_greedy.png 128 8 greedy_parallel
