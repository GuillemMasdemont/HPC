#!/bin/bash

#SBATCH --job-name=fs
#SBATCH --output=fs.log
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --threads-per-core=1
#SBATCH --time=5:00
#SBATCH --mem-per-cpu=20000
#SBATCH --reservation=fri

srun ./fs 1
srun ./fs 8