#!/bin/bash

#SBATCH --job-name=conway
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=10:00
#SBATCH --output=conway.log
#SBATCH --reservation=fri

module load OpenMPI

mpirun --display-allocation --n 4 $SLURM_JOB_NAME
