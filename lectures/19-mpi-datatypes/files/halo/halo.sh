#!/bin/bash

#SBATCH --job-name=halo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=10:00
#SBATCH --output=halo.log
#SBATCH --reservation=fri

module load OpenMPI

mpirun --display-allocation --n 2 $SLURM_JOB_NAME

