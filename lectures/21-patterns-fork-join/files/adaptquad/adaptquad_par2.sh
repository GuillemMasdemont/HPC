#!/bin/bash

#SBATCH --job-name=adaptquad_par2
#SBATCH --output=adaptquad_par2.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --threads-per-core=1
#SBATCH --time=20:00
#SBATCH --mem-per-cpu=2000
#SBATCH --reservation=fri

export OMP_PLACES=cores
export OMP_PROC_BIND=close

export OMP_NUM_THREADS=1
./adaptquad_par2

export OMP_NUM_THREADS=2
./adaptquad_par2

export OMP_NUM_THREADS=4
./adaptquad_par2

export OMP_NUM_THREADS=8
./adaptquad_par2
