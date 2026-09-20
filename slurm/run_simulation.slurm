#!/bin/bash
# Simulation study (paper Section "Experiments" + Appendix): one array task per DGP.
#   0: linear 3cov   1: linear 5cov   2: linear 10cov   3: polynomial   4: sinusoidal
#
# Submit from the repository root:
#   sbatch slurm/run_simulation.slurm
# Edit the account / partition / environment lines for your own cluster.
#SBATCH --job-name=fna_sim
#SBATCH --array=0-4
##SBATCH --account=<your-account>
##SBATCH --partition=<gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

# --- Environment (adapt to your cluster) ---
# module load Anaconda3
# source activate fna-bounds

DGPS=("--dgp-type linear --cov 3cov"
      "--dgp-type linear --cov 5cov"
      "--dgp-type linear --cov 10cov"
      "--dgp-type polynomial"
      "--dgp-type sinusoidal")
DGP_ARGS=${DGPS[$SLURM_ARRAY_TASK_ID]}

mkdir -p logs results/simulation
echo "Task ${SLURM_ARRAY_TASK_ID}: ${DGP_ARGS}  ($(date))"

python experiments/simulation/fna_simulation.py \
    --output-dir results/simulation \
    --base-seed 2026 \
    --n-sims 300 \
    --n-sample 5000 \
    --n-mc 5 \
    --epochs1 30 \
    --epochs2 50 \
    --hidden 64 \
    --n-layers 2 \
    --lr 1e-3 \
    --batch-size 256 \
    --weight-decay 1e-4 \
    --print-every 400 \
    ${DGP_ARGS}

echo "Finished $(date)"
