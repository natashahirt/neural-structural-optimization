#!/bin/bash
#SBATCH -p mit_normal
#SBATCH --job-name=neo-run
#SBATCH -c 8                      # CPUs per task
#SBATCH --mem=32G                 # adjust if needed
#SBATCH --time=00:15:00
#SBATCH --output=logs/neo-%j.out
#SBATCH --error=logs/neo-%j.err

set -euo pipefail
cd /home/nhirt/neural-structural-optimization

# Environment setup
module load miniforge            # remove if you prefer a pure venv
source /home/nhirt/neural-structural-optimization/.venv/bin/activate

# Ensure package is importable without installation
export PYTHONPATH=/home/nhirt/neural-structural-optimization:${PYTHONPATH:-}

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run without command-line args (per your preference)
python script/run.py