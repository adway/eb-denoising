#!/bin/bash
#SBATCH --job-name=combine_eb_denoising
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=30:00

module load python3.11-anaconda/2024.02

echo "Activating environment…"
source activate eb-denoising

cd "$HOME/eb-denoising"
export PYTHONPATH="$HOME/eb-denoising/src:$PYTHONPATH"

echo "Python version:"
python

echo "Installed packages:"
python -m pip list | head

echo "Starting visualizations"
python ~/eb_denoising/src/combine_and_analyze.py --input_dir=metrics --output_dir=summary_metrics