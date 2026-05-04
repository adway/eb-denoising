# run_calibration_submitit.py

import os
import sys
import random
import warnings
import pandas as pd
import submitit

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from calibration_helpers import make_pvals, calibrate_and_assess

M = 2000
pi0s = [0.5, 0.75, 0.9]
alphas = [0.5, 0.7, 0.95]
beta = 2.3

n_grid = [500, 1_000, 5_000, 10_000, 100_000, 1_000_000]
calibrators = ["lfdr", "q-value", "p-value"]

N_JOBS = 500
OUTDIR = "calibration_results"


def split_into_chunks(tasks, n_chunks):
    chunks = [[] for _ in range(n_chunks)]
    for idx, task in enumerate(tasks):
        chunks[idx % n_chunks].append(task)
    return chunks


def run_task_chunk(chunk_id, tasks):
    os.makedirs(OUTDIR, exist_ok=True)
    rows = []

    for task in tasks:
        pi0 = task["pi0"]
        alpha = task["alpha"]
        rep = task["rep"]
        n = task["n"]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                labels, pvals = make_pvals(
                    n=n,
                    pi0=pi0,
                    alpha=alpha,
                    beta=beta,
                )

                lam = 1 - len(pvals) ** (-1 / 5)

                for calibrator in calibrators:
                    try:
                        calibration_error, _ = calibrate_and_assess(
                            pvals,
                            labels,
                            lam=lam,
                            n_bins=10,
                            calibrator=calibrator,
                            pi0=pi0,
                            alpha=alpha,
                            beta=beta,
                        )

                        rows.append({
                            "chunk_id": chunk_id,
                            "rep": rep,
                            "n": n,
                            "pi0": pi0,
                            "alpha": alpha,
                            "beta": beta,
                            "calibrator": calibrator,
                            "calibration_error": calibration_error,
                            "error": None,
                        })

                    except Exception as e:
                        rows.append({
                            "chunk_id": chunk_id,
                            "rep": rep,
                            "n": n,
                            "pi0": pi0,
                            "alpha": alpha,
                            "beta": beta,
                            "calibrator": calibrator,
                            "calibration_error": None,
                            "error": repr(e),
                        })

        except Exception as e:
            rows.append({
                "chunk_id": chunk_id,
                "rep": rep,
                "n": n,
                "pi0": pi0,
                "alpha": alpha,
                "beta": beta,
                "calibrator": None,
                "calibration_error": None,
                "error": repr(e),
            })

    path = f"{OUTDIR}/results_chunk_{chunk_id:04d}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs("run_logs", exist_ok=True)

    tasks = [
        {"pi0": pi0, "alpha": alpha, "rep": rep, "n": n}
        for pi0 in pi0s
        for alpha in alphas
        for rep in range(M)
        for n in n_grid
    ]

    random.seed(123)
    random.shuffle(tasks)

    chunks = split_into_chunks(tasks, N_JOBS)

    print(f"Total datasets: {len(tasks)}")
    print(f"Total jobs: {len(chunks)}")
    print(f"Tasks per job: about {len(tasks) / len(chunks):.1f}")

    executor = submitit.AutoExecutor(folder="run_logs")
    executor.update_parameters(
        slurm_job_name="calib",
        slurm_partition="standard", 
        account="jonth1",  # change this
        slurm_time=240,               # minutes
        slurm_mem="16G",
        cpus_per_task=1,
        tasks_per_node=1
    )

    jobs = []
    # with executor.batch():
    for chunk_id, chunk in enumerate(chunks):
        jobs.append(
            executor.submit(run_task_chunk, chunk_id, chunk)
        )

    print(f"Submitted {len(jobs)} jobs.")
    print([job.job_id for job in jobs[:10]])