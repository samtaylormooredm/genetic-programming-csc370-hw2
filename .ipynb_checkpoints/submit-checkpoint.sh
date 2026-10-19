#!/bin/bash
#SBATCH --job-name=gp
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out

set -e
cd "$SLURM_SUBMIT_DIR"

./.venv/bin/python -u gp.py "$1"