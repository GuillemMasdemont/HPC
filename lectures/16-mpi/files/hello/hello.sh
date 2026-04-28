#!/bin/bash

#SBATCH --job-name=hello
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=10:00
#SBATCH --output=hello.log
#SBATCH --reservation=fri

module load OpenMPI

mpirun --display-allocation -n 8 $SLURM_JOB_NAME

