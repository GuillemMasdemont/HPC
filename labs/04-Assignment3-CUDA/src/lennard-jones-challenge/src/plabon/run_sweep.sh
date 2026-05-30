#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lj_sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --output=sweep_out.log
#SBATCH --time=08:00:00

# LOAD MODULES
module load CUDA

# BUILD
make clean
make sweep

# RUN — pretune table to stdout, tuner progress to stderr (both go to sweep_out.log)
srun ./lj_sweep.out > pretune_table.txt