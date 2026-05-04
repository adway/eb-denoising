import glob
import pandas as pd

files = glob.glob("calibration_results/results_chunk_*.csv")

results_df = pd.concat(
    [pd.read_csv(f) for f in files],
    ignore_index=True,
)

summary_df = (
    results_df
    .dropna(subset=["calibration_error"])
    .groupby(["pi0", "alpha", "beta", "n", "calibrator"], as_index=False)
    .agg(
        mean_calibration_error=("calibration_error", "mean"),
        sd_calibration_error=("calibration_error", "std"),
        n_success=("calibration_error", "size"),
    )
)

summary_df.to_csv("calibration_results/summary.csv", index=False)