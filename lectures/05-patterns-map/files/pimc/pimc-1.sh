#!/bin/bash

#SBATCH --job-name=pimc-1
#SBATCH --output=pimc-1.log
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --ntasks-per-core=1
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

srun ./pimc 1024 0
