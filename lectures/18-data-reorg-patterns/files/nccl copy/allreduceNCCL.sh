#!/bin/bash
#SBATCH --job-name=allreduceNCCL
#SBATCH --output=allreduceNCCL.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=2
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --partition=gpu
#SBATCH --reservation=fri

# compile
module purge
module load NCCL
nvcc -o allreduceNCCL allreduceNCCL.cu -lnccl -O2

# print topology
nvidia-smi topo -m
echo ""

# execute code
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0  # ensure peer-to-peer is enabled

./allreduceNCCL
