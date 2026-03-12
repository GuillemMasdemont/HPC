#!/bin/bash

#SBATCH --job-name=pisum-4
#SBATCH --output=pisum-4.log
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --ntasks-per-core=1
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

srun ./pisum 4
