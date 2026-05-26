#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=lenia_nccl_bench
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --output=lenia_nccl_bench_%j.log
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00



module load CUDA
module load OpenMPI          # still needed for the MPI bootstrap in NCCL init
module load NCCL        # exact name may differ — run: module avail nccl
 
 
# Let each MPI rank select its own GPU via cudaSetDevice(rank % num_gpus)
unset CUDA_VISIBLE_DEVICES

# Force NCCL to use PCIe P2P / NVLink instead of shared memory.
# NCCL SHM is blocked by the kernel's Yama ptrace_scope, producing Read -1 / errno=1 warnings.
export NCCL_SHM_DISABLE=1
 
# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
# replace with:
make -f Makefile_nccl clean
make -f Makefile_nccl 
# ---------------------------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------------------------
GRID_SIZES="128 512 1024 2048 4096"
# GRID_SIZES="4096"
 
# GPU counts to sweep — must be <= total GPUs allocated above
# GPU_COUNTS="1 2 4 8"
GPU_COUNTS="1"
 
STEPS=100
RUNS=3        # number of runs to average per configuration
 
echo "========================================================"
echo "Lenia NCCL+CUDA Benchmark"
echo "Steps: ${STEPS}  |  Averaged over ${RUNS} runs"
echo "Node count : ${SLURM_NNODES}"
echo "GPUs/node  : ${SLURM_NTASKS_PER_NODE}"
echo "========================================================"
echo ""
echo "# Raw output — lines starting with CSV can be extracted for speedup:"
echo "# Format: CSV,grid_size,gpus,avg_time_seconds"
echo ""
 
for N in $GRID_SIZES; do
    echo "--- Grid ${N}x${N} ---"
    for NG in $GPU_COUNTS; do
 
        # srun launches NG MPI ranks, each pinned to one GPU.
        # --gpus-per-task=1 guarantees CUDA_VISIBLE_DEVICES=0 per rank via SLURM.
        mpirun -np ${NG} \
             --mca pml ob1 --mca btl vader,self \
             --mca btl_vader_single_copy_mechanism none \
             ./lenia_nccl.out ${N} --steps ${STEPS} --runs ${RUNS}
    done
    echo ""
done
 
echo "========================================================"
echo "GIF generation test (128x128, 4 GPUs)"
echo "========================================================"
mpirun -np ${SLURM_NTASKS} \
     --mca pml ob1 --mca btl vader,self \
     --mca btl_vader_single_copy_mechanism none \
     ./lenia_nccl.out 128 --steps 1000 --gif
echo "GIF written to lenia.gif"
 