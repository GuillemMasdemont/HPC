#!/bin/bash

#SBATCH --job-name=falsesharing
#SBATCH --output=falsesharing.log
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --threads-per-core=1
#SBATCH --time=5:00
#SBATCH --mem-per-cpu=20000
#SBATCH --reservation=fri

srun ./falsesharing 1
srun ./falsesharing 8