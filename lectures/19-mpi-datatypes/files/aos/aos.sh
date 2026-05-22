#!/bin/bash

#SBATCH --job-name=aos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=10:00
#SBATCH --output=aos.log
#SBATCH --reservation=fri

module load OpenMPI

mpirun --display-allocation --n 4 $SLURM_JOB_NAME

