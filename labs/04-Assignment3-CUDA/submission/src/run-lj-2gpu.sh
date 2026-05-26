#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lennard-jones-2gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --output=lj_2gpu_out.log

# ── Load modules ──────────────────────────────────────────────────────────────
module load CUDA

# ── Build ─────────────────────────────────────────────────────────────────────
make

# ── Run ───────────────────────────────────────────────────────────────────────
# OMP_NUM_THREADS=2 gives each GPU its own dedicated CPU thread.
# Binding is set to none so OpenMP threads are free to call cudaSetDevice()
# on whichever GPU they need — SLURM's default CPU binding can otherwise
# pin both threads to the same core and prevent them running in parallel.
export OMP_NUM_THREADS=2
export OMP_PROC_BIND=false

srun ./lj.out --two-gpu