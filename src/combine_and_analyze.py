import scipy.stats as stats
import numpy as np
import pandas as pd
import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

def combine_and_analyze(input_dir, output_csv):
    all_files = list(Path(input_dir).glob('sim_*.npz'))
    records = []
    
    for file in all_files:
        data = np.load(file, allow_pickle=True)
        prior_type = data['prior_type'].item()
        n = data['n'].item()
        sigma2 = data['sigma2'].item()
        metrics = data['metrics'].item()
        record = {
            'prior_type': prior_type,
            'n': n,
            'sigma2': sigma2,
            **metrics
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    
    summary_df = df.groupby(['prior_type', 'n', 'sigma2']).agg(['mean', 'std']).reset_index()
    
    summary_df.to_csv(output_csv, index=False)

def make_bivariate_density_plots(input_dir, output_dir):
    
    all_files = list(Path(input_dir).glob('sim_*.npz'))
    records = []
    
    for file in all_files:
        data = np.load(file, allow_pickle=True)
        prior_type = data['prior_type'].item()
        n = data['n'].item()
        sigma2 = data['sigma2'].item()
        metrics = data['metrics'].item()
        record = {
            'prior_type': prior_type,
            'n': n,
            'sigma2': sigma2,
            'prior_dist': metrics['prior_dist'],
            'post_dist': metrics['post_dist']
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for (prior_type, n, sigma2), group in df.groupby(['prior_type', 'n', 'sigma2']):
      plt.figure(figsize=(8, 6))
      # can you make the plot discretized instead of a smooth kde?
      sns.kdeplot(
          x=group['prior_dist'],
          y=group['post_dist'],
          fill=True,
          cmap="Blues",
          thresh=0,
          levels=100,
          alpha=0.7
      )
      # Add reference lines
      max_val = max(group['prior_dist'].max(), group['post_dist'].max())
      plt.plot([0, max_val], [0, max_val], color='red', linestyle='--')
      # add labels and title
      plt.title(f'Bivariate Density Plot\nPrior: {prior_type}, n: {n}, sigma2: {sigma2}')
      plt.xlabel('Prior Distance')
      plt.ylabel('Posterior Distance')
      # add legend for colors
      # RuntimeError: No mappable was found to use for colorbar creation. First define a mappable such as an image (with imshow) or a contour set (with contourf).
      plt.colorbar(label='Density')
      
      plt.savefig(Path(output_dir) / f'bivariate_density_{prior_type}_n{n}_sigma2{sigma2}.png')
      plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Combine and analyze simulation results.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing simulation npz files.')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to output CSV file for summary statistics.')
    args = parser.parse_args()
    
    # combine_and_analyze(args.input_dir, args.output_csv)
    make_bivariate_density_plots(args.input_dir, "plots")