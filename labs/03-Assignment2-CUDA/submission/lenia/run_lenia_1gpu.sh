#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lenia-1gpu
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --output=lenia_1gpu_out.log

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

# Use the allocated CPU count as OpenMP thread count by default.
# If a 32-core node is available, submit with:
# sbatch --cpus-per-task=32 run_lenia_1gpu.sh
CPU_THREADS="${LENIA_CPU_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
echo "[run] Using ${CPU_THREADS} CPU threads for seq_v2 baseline"

# RUN ONLY SINGLE-GPU VERSIONS (seq, seq_v2, naive, v1, v2, v3, v5)
# Enforce no SMT and pin OpenMP threads to cores.
srun --hint=nomultithread --cpu-bind=cores \
	--export=ALL,LENIA_BENCH_MODE=single,LENIA_CPU_THREADS=${CPU_THREADS},OMP_NUM_THREADS=${CPU_THREADS},OMP_PLACES=cores,OMP_PROC_BIND=close \
	./lenia.out

# Preserve output with a distinct name
cp benchmark_results.txt benchmark_results_1gpu_tile8.txt
