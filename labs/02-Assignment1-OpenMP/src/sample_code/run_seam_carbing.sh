#!/bin/bash

#SBATCH --account=fri-users
#SBATCH --reservation=fri
#SBATCH --job-name=code_sample
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=sample_out.log
#SBATCH --hint=nomultithread

# Set OpenMP environment variables for thread placement and binding    
export OMP_PLACES=cores
export OMP_PROC_BIND=close #threads spawn to each other on cores that are close to each other with regards to cash.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK #number of threads inside your program to the number of cores requestes. In that case we set the number of threads to 8

# Load the numactl module to enable numa library linking, in order to use the numa_node_cpu function in the sample.c script. 
module load numactl

# Compile
gcc -O3 -lm -lnuma --openmp seam_carving.c -o seam_carving

# Run
srun  seam_carving
