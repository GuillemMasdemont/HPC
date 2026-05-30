#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lj-tuner-v2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --output=lj_tuner_v2_out.log

# LOAD MODULES
module load CUDA

# BUILD
make clean
make WITH_TUNER_V2=1

# RUN
srun ./lj.out
