#!/bin/bash

#SBATCH --job-name=adaptquad_ser
#SBATCH --output=adaptquad_ser.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --time=20:00
#SBATCH --mem-per-cpu=20000
#SBATCH --reservation=fri

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

./adaptquad_ser
