#!/bin/bash

#SBATCH --job-name=pmult_ser
#SBATCH --output=pmult_ser.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --time=20:00
#SBATCH --mem-per-cpu=2000
#SBATCH --reservation=fri

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

#./pmult_ser 1048576 32
./pmult_ser 131072 32
