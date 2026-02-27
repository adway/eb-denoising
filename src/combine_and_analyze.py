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

        # Compute 2D histogram
        H, xedges, yedges = np.histogram2d(
            group['prior_dist'],
            group['post_dist'],
            bins=100,
            range=[[0, 0.35], [0, 0.35]]
        )

        # Plot as pixel grid
        mesh = plt.pcolormesh(
            xedges,
            yedges,
            H.T,                     # transpose is critical
            cmap="viridis",          # close to your example
            shading="flat",
            edgecolors="k",          # draw grid lines
            linewidth=0.05           # thin grid lines
        )

        # Colorbar
        plt.colorbar(mesh, label="Count")

        # Reference line y = x
        plt.plot([0, 0.35], [0, 0.35], color="red", linestyle="--", linewidth=1)

        # Labels and title
        plt.xlabel("Prior Distance")
        plt.ylabel("Posterior Distance")
        plt.title(f"Bivariate Density Plot\nPrior: {prior_type}, n: {n}, sigma2: {sigma2}")

        # Fix axes
        plt.xlim(0, 0.35)
        plt.ylim(0, 0.35)

        plt.savefig(
            Path(output_dir) / f"bivariate_density_{prior_type}_n{n}_sigma2{sigma2}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

def make_log_log_plots(input_dir, output_dir):
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
            # 'prior_dist': metrics['prior_dist'],
            # 'post_dist': metrics['post_dist'],
            # 'evcb_distance': metrics['evcb_distance'],
            # 'denoise_regret': metrics['denoise_regret'],
            # 'denoise_diff': metrics['denoise_diff']
            **metrics
        }
        records.append(record)
    df = pd.DataFrame(records)
    os.makedirs(output_dir, exist_ok=True)

    for prior_type, prior_group in df.groupby('prior_type'):
        for distance_type in metrics.keys(): # ['prior_dist', 'post_dist', 'evcb_distance', 'denoise_regret', 'denoise_diff']
            plt.figure(figsize=(8, 6))
            for sigma2, sigma_group in prior_group.groupby('sigma2'):
                mean_distances = sigma_group.groupby('n')[distance_type].mean().reset_index()
                plt.plot(
                    mean_distances['n'],
                    mean_distances[distance_type],
                    marker='o',
                    label=f'sigma2={sigma2}'
                )
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('n (log scale)')
            plt.ylabel(f'{distance_type} (log scale)')
            plt.title(f'Log-Log Plot of {distance_type} vs n\nPrior: {prior_type}')
            plt.legend()
            plt.grid(True, which="both", ls="--")
            plt.savefig(
                Path(output_dir) / f"log_log_{distance_type}_{prior_type}.png",
                dpi=300,
                bbox_inches="tight"
            )
            plt.close()

    # get slope of lines for each distance as a function of n, for each prior type and sigma2
    slope_records = []
    for prior_type, prior_group in df.groupby('prior_type'):
        for sigma2, sigma_group in prior_group.groupby('sigma2'):
            for distance_type in ['prior_dist', 'post_dist', 'evcb_distance', 'denoise_regret', 'denoise_diff']:
                mean_distances = sigma_group.groupby('n')[distance_type].mean().reset_index()
                log_n = np.log(mean_distances['n'])
                log_distance = np.log(mean_distances[distance_type])
                slope, intercept = np.polyfit(log_n, log_distance, 1)
                slope_records.append({
                    'prior_type': prior_type,
                    'sigma2': sigma2,
                    'distance_type': distance_type,
                    'slope': slope,
                    'intercept': intercept
                })
    slope_df = pd.DataFrame(slope_records)
    slope_df.to_csv(Path(output_dir) / "log_log_slopes.csv", index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Combine and analyze simulation results.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing simulation npz files.')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to output CSV file for summary statistics.')
    args = parser.parse_args()
    
    combine_and_analyze(args.input_dir, args.output_csv)
    # make_bivariate_density_plots(args.input_dir, "plots")
    make_log_log_plots(args.input_dir, "plots")