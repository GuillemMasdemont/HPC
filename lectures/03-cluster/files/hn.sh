#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --partition=all
#SBATCH --ntasks=4 
#SBATCH --nodes=1 
#SBATCH --mem-per-cpu=100MB 
#SBATCH --output=myjob.out 
#SBATCH --time=00:01:00

srun sleep 10
srun hostname
