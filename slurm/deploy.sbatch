#!/bin/bash
#SBATCH --job-name=spr-train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=30:00

load_conda11

echo "Activating environment…"
conda activate eb-denoising

cd "$HOME/eb-denoising"
export PYTHONPATH="$HOME/eb-denoising/src:$PYTHONPATH"

echo "Python version:"
python

echo "Installed packages:"
python -m pip list | head

echo "Deploying simulations..."
python ~/eb-denoising/src/run_simulations.py --config="~/eb-denoising/configs/base.json" --output_dir="metrics" --use_slurm --max_nodes=500
