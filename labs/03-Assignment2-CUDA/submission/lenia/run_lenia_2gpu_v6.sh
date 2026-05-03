#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lenia-2gpu-v6
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --output=lenia_2gpu_v6_out.log


# Prefer SLURM_SUBMIT_DIR so outputs stay in the submitted project folder.
BASE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
LOG_DIR="$BASE_DIR/logs"
OUT_DIR="$BASE_DIR/outputs"

mkdir -p "$LOG_DIR" "$OUT_DIR"

RUN_ID="${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/lenia_openmp_v2_${RUN_ID}.log"
OUT_FILE="$OUT_DIR/lenia_openmp_v2_${RUN_ID}.out"

exec > >(tee -a "$LOG_FILE") 2>&1

# LOAD MODULES
module load CUDA

# BUILD
make

# RUN ONLY DUAL-GPU V6
srun --export=ALL,LENIA_BENCH_MODE=v6 ./lenia.out

# Preserve output with a distinct name
cp benchmark_results.txt benchmark_results_v6_2gpu.txt
