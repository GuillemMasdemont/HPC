#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=lennard-jones
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=lj_out_cpu.log

#LOAD MODULES 
module load CUDA

#BUILD
make

#RUN
srun ./lj.out --two-gpu