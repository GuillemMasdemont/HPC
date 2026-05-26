#!/bin/bash
#
# Benchmark script for the MPI Lenia simulation.
# Measures average execution time for all required grid sizes and core counts.
# Submit with:  sbatch run_lenia.sh
#

#SBATCH --reservation=fri
#SBATCH --job-name=lenia_bench
#SBATCH --ntasks=32
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --output=lenia_bench_%j.log
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00

# Load MPI module
module load OpenMPI

# Build
make clean
make

# GRID_SIZES="128 512 1024 2048 4096"
GRID_SIZES="4096"

CORE_COUNTS="1 2 4 16 32"
# CORE_COUNTS="32"

STEPS=100
RUNS=3   # number of runs to average per configuration

echo "========================================================"
echo "Lenia MPI Benchmark"
echo "Steps: ${STEPS}  |  Averaged over ${RUNS} runs"
echo "========================================================"
echo ""
echo "# Raw output — lines starting with CSV can be extracted for speedup:"
echo "# Format: CSV,grid_size,procs,avg_time_seconds"
echo ""

for N in $GRID_SIZES; do
    echo "--- Grid ${N}x${N} ---"
    for NP in $CORE_COUNTS; do
        mpirun --mca pml ob1 -np ${NP} ./lenia.out ${N} --steps ${STEPS} --runs ${RUNS}
    done
    echo ""
done

echo "========================================================"
echo "GIF generation test (128x128, 4 processes)"
echo "========================================================"
mpirun --mca pml ob1 -np 4 ./lenia.out 128 --steps 1000 --gif
echo "GIF written to lenia.gif"
