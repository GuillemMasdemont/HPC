#!/bin/bash

#SBATCH --job-name=pimc-4
#SBATCH --output=pimc-4-%a.log
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --ntasks-per-core=1
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100
#SBATCH --array=0-3

srun ./pimc 256 $SLURM_ARRAY_TASK_ID
