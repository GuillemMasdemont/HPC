#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lj-tuner-v3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --output=lj_tuner_v3_out.log

# LOAD MODULES
module load CUDA

# BUILD
make clean
make WITH_TUNER_V3=1

# RUN
srun ./lj.out
