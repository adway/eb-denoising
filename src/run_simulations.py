# entry point script to run simulations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import itertools
import numpy as np
from one_sim import OneSimulation
import argparse
import json
import os
import submitit

def run_simulation(prior_type, prior_params, n, sigma2, n_supp, output_path):
    sim = OneSimulation(prior_type, prior_params, n, sigma2, n_supp)
    estimates, metrics = sim.run_simulation()
    # save estimates and simulation parameters to output_path in parquet format
    results = {
        'prior_type': prior_type,
        'prior_params': prior_params,
        'n': n,
        'sigma2': sigma2,
        'n_supp': n_supp,
        'metrics': metrics
    }
    np.savez_compressed(output_path, **results) # output path is the actual file that needs saving.

def run_batch(batch, prior_params, n_supp, output_dir): # batch is a list of (prior_type, n, sigma2), need this in order to submit onto the cluster more easily
    # os.makedirs(output_dir, exist_ok=True)
    fail_log = Path(output_dir) / 'failed_jobs.log'
    n_ok = 0
    n_fail = 0
    for idx, prior_type, n, sigma2 in batch:
        output_path = os.path.join(output_dir, f'sim_{idx}.npz')
        try:
            run_simulation(prior_type, prior_params, n, sigma2, n_supp, output_path)
            n_ok += 1
        except Exception as e:
            with open(fail_log, 'a') as f:
                f.write(f"Failed job {idx}, {prior_type}, n={n}, sigma2={sigma2}. Error: {e}\n")
            n_fail += 1
            
    return {"ok": n_ok, "fail": n_fail}
    

def get_grid(prior_types, n_list, sigma2_list, n_per_sim):
    grid = []
    for prior_type, n, sigma2 in itertools.product(prior_types, n_list, sigma2_list):
        # append n_per_sim copies of each setting
        for _ in range(n_per_sim):
            grid.append((prior_type, n, sigma2))
    return grid

def split_list(lst, n):
    """
    Split lst into n approximately equal chunks.
    Identical behavior to submitit.helpers.split_list.
    """
    n = min(n, len(lst))
    k, m = divmod(len(lst), n)
    return [
        lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
        for i in range(n)
    ]

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Run simulations for empirical Bayes estimation.')
    parser.add_argument('--config', type=str, required=True, help='Path to JSON config file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save simulation results.')
    parser.add_argument('--use_slurm', action='store_true', help='Use SLURM for job submission.')
    parser.add_argument('--max_nodes', type=int, default=1, help='Maximum number of SLURM nodes to use.')
    args = parser.parse_args()

    # load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    prior_types = config['prior_types']
    prior_params = config['prior_params']
    n_list = config['n_list']
    sigma2_list = config['sigma2_list']
    n_per_sim = config['n_per_sim']
    n_supp = config.get('n_supp', 150)

    grid = get_grid(prior_types, n_list, sigma2_list, n_per_sim)
    os.makedirs(args.output_dir, exist_ok=True)

    indexed_grid = [(i, prior_type, n, sigma2) for i, (prior_type, n, sigma2)    in enumerate(grid)]
    np.random.shuffle(indexed_grid) # shuffle the grid to better distribute load across jobs

    if args.use_slurm:
        executor = submitit.AutoExecutor(folder="run_logs")
        executor.update_parameters(nodes = 1, partition="standard", mem_gb=16, cpus_per_task=1, time=65, account="stats_dept1", slurm_array_parallelism=args.max_nodes)
        num_jobs = min(args.max_nodes, len(grid)) # number of parallel jobs to run
        batches = list(split_list(indexed_grid, num_jobs)) # split the grid into num_jobs batches
        jobs = executor.map_array(
            run_batch,
            batches,
            itertools.repeat(prior_params), # one for each batch, but same, because needs iteration.
            itertools.repeat(n_supp),
            itertools.repeat(args.output_dir)
          )
    else:
        for i, (prior_type, n, sigma2) in enumerate(grid):
            output_path = os.path.join(args.output_dir, f'sim_{i}.npz')
            run_simulation(prior_type, prior_params, n, sigma2, n_supp, output_path)

if __name__ == '__main__':
    main()