#!/bin/bash

set -euo pipefail

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lenia_openmp_v2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --nodes=1

# In SLURM batch jobs, $0 may point to a temporary spool path.
# Prefer SLURM_SUBMIT_DIR so outputs stay in the submitted project folder.
BASE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
LOG_DIR="$BASE_DIR/logs"
OUT_DIR="$BASE_DIR/outputs"

mkdir -p "$LOG_DIR" "$OUT_DIR"

RUN_ID="${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/lenia_openmp_v2_${RUN_ID}.log"
OUT_FILE="$OUT_DIR/lenia_openmp_v2_${RUN_ID}.out"

exec > >(tee -a "$LOG_FILE") 2>&1

#LOAD MODULES
module load CUDA

#BUILD
make -f Makefile.openmp_v2

#RUN
THREADS="${1:-${SLURM_CPUS_PER_TASK:-1}}"
echo "Running with $THREADS OpenMP thread(s)"
srun ./lenia_openmp_v2.out "$THREADS" > "$OUT_FILE"

echo "Program output saved to: $OUT_FILE"
echo "Build/runtime log saved to: $LOG_FILE"