#!/bin/bash
# Moving to Opportunity (MTO) application.
# Requires the MTO data file at data/mto.dta (not distributed; see data/README.md).
#
# Submit from the repository root:
#   sbatch slurm/run_mto.slurm
# Edit the account / partition / environment lines for your own cluster.
#SBATCH --job-name=mto_fna_dr
##SBATCH --account=<your-account>
##SBATCH --partition=<gpu-partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# --- Environment (adapt to your cluster) ---
# module load Anaconda3
# source activate fna-bounds

mkdir -p logs results/mto
nvidia-smi || echo "no GPU visible (will fall back to CPU)"

python experiments/mto/run_mto_fna.py \
    --data data/mto.dta \
    --out  results/mto \
    --K 5 \
    --n-mc 10000 \
    --epochs-stage1 50 \
    --epochs-stage2 150 \
    --hidden 128 \
    --n-grid 900 \
    --seed 0

echo "Done $(date). Results in results/mto/mto_fna_dr_results.json"
