#!/bin/bash

#SBATCH --job-name=pmult_par
#SBATCH --output=pmult_par.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00
#SBATCH --mem-per-cpu=2000
#SBATCH --reservation=fri

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

#./pmult_par 1048576 32
./pmult_par 131072 16
./pmult_par 131072 32
./pmult_par 131072 64

