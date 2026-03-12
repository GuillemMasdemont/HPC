#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=all
#SBATCH --job-name=my_first_job
#SBATCH --ntasks=2
#SBATCH --time=00:01:00
#SBATCH --mem-per-cpu=100MB
#SBATCH --output=job-output.log

srun hostname