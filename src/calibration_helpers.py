import numpy as np
import pandas as pd
from npeb import Grenander, local_fdr
import scipy.stats as stats
from scipy.io import loadmat
from scipy.integrate import quad

def make_pvals(n, pi0, alpha = 0.5, beta = 1.0):
    n0 = int(pi0 * n)
    n1 = n - n0
    pvals = np.concatenate([np.random.uniform(size=n0), np.random.beta(a=alpha, b=beta, size=n1)])
    labels = np.concatenate([np.zeros(n0), np.ones(n1)])
    perm = np.random.permutation(n)
    pvals = pvals[perm]
    labels = labels[perm]
    return labels, pvals

def qvalues(pvals, lam=0.5):
    pvals = np.asarray(pvals)
    m = len(pvals)

    # sort p-values
    order = np.argsort(pvals)
    p_sorted = pvals[order]

    # estimate pi0
    pi0 = np.mean(pvals > lam) / (1 - lam)
    pi0 = min(pi0, 1.0)

    # compute initial q-values (i.e. pFDRs for each p-value)
    qvals = m * p_sorted * pi0 / np.arange(1, m + 1)

    # enforce monotonicity by taking minimum at each step
    qvals = np.minimum.accumulate(qvals[::-1])[::-1]
    qvals = np.minimum(qvals, 1.0)

    # return in original order
    # q_original = np.empty_like(qvals)
    # q_original[order] = qvals

    q_original = qvals[np.argsort(order)]

    return q_original


def mixture_density(x, pi0, alpha, beta):
    return pi0 + (1 - pi0) * stats.beta.pdf(x, a=alpha, b=beta)

def conditional_mean(x, pi0, alpha, beta):
    return pi0 / mixture_density(x, pi0, alpha, beta)

def integral(i, pi0, alpha, beta, knots, heights):
    def integrand(x):
        return mixture_density(x, pi0, alpha, beta) * (conditional_mean(x, pi0, alpha, beta) - heights[i])**2
    return quad(integrand, knots[i], knots[i+1])[0]

def total_integral(pi0, alpha, beta, knots, heights):
    return sum(integral(i, pi0, alpha, beta, knots, heights) for i in range(len(knots) - 1))

def calibrate_and_assess(pvals, labels, lam = 0.5, n_bins = 10, calibrator = "p-value", pi0 = 0.90, alpha = 0.5, beta = 2.3):
  pi_0_hat = np.mean(pvals > lam) / (1 - lam)
  if calibrator == "p-value":
    stat_df = pd.DataFrame({"p_val": pvals, "label": labels})
    bins = np.linspace(0, 1, n_bins + 1)
    stat_df["bin"] = pd.cut(stat_df["p_val"], bins=bins, include_lowest=True, right = True)
    bin_stats = stat_df.groupby("bin").agg(
        p_min = ("p_val", "min"),
        p_max = ("p_val", "max"),
        p_mean = ("p_val", "mean"),
        p_bin_length = ("p_val", lambda x: x.max() - x.min()),
        stat_mean = ("p_val", "mean"),
        count = ("p_val", "count"),
        proportion_true_nulls = ("label", lambda x: 1 - np.mean(x)),
    ).reset_index()
    bin_stats["proportion_true_nulls_hat"] = np.minimum(pi_0_hat * bin_stats["p_bin_length"] * len(stat_df["p_val"])/ bin_stats["count"], 1)
  elif calibrator == "q-value":
    qvals = qvalues(pvals, lam = lam)
    stat_df = pd.DataFrame({"p_val": pvals, "q_val": qvals, "label": labels})
    bins = np.linspace(0, 1, n_bins + 1)
    stat_df["bin"] = pd.cut(stat_df["q_val"], bins=bins, include_lowest=True, right = True)
    bin_stats = stat_df.groupby("bin").agg(
        p_min = ("p_val", "min"),
        p_max = ("p_val", "max"),
        p_mean = ("p_val", "mean"),
        p_bin_length = ("p_val", lambda x: x.max() - x.min()),
        stat_mean = ("q_val", "mean"),
        count = ("p_val", "count"),
        proportion_true_nulls = ("label", lambda x: 1 - np.mean(x)),
    ).reset_index()
    bin_stats["proportion_true_nulls_hat"] = np.minimum(pi_0_hat * bin_stats["p_bin_length"] * len(stat_df["p_val"])/ bin_stats["count"], 1)

  elif calibrator == "lfdr":
    train_idx = np.random.choice(len(pvals), size=int(0.5 * len(pvals)), replace=False)
    train_pvals = pvals[train_idx]
    train_labels = labels[train_idx]
    test_pvals = pvals[~np.isin(pvals, train_pvals)]
    test_labels = labels[~np.isin(pvals, train_pvals)]

    gren = Grenander(x_min=0, x_max=1.0)
    fhat = gren.fit(train_pvals)
    lam = 1 - len(test_pvals)**(-1/5)

    pi_hat_0_gren = np.mean(test_pvals > lam) / (1 - lam)
    gren.pdf(test_pvals)
    test_lfdr = pi_hat_0_gren / gren.pdf(test_pvals)

    stat_df = pd.DataFrame({"p_val": test_pvals, "lfdr": test_lfdr, "label": test_labels})
    bins = np.linspace(0, 1, n_bins + 1)
    stat_df["bin"] = pd.cut(stat_df["lfdr"], bins=bins, include_lowest=True, right = True)
    bin_stats = stat_df.groupby("bin").agg(
        p_min = ("p_val", "min"),
        p_max = ("p_val", "max"),
        p_mean = ("p_val", "mean"),
        p_bin_length = ("p_val", lambda x: x.max() - x.min()),
        stat_mean = ("lfdr", "mean"),
        count = ("p_val", "count"),
        proportion_true_nulls = ("label", lambda x: 1 - np.mean(x)),
    ).reset_index()
    bin_stats["proportion_true_nulls_hat"] = np.minimum(pi_hat_0_gren * bin_stats["p_bin_length"] * len(stat_df["p_val"])/ bin_stats["count"], 1)
  
  calibration_error = None
  if calibrator == "p-value":
    def integrand(x):
        return mixture_density(x, pi0, alpha, beta) * (conditional_mean(x, pi0, alpha, beta) - x)**2
    calibration_error = quad(integrand, 0, 1)[0]
  elif calibrator == "q-value":
    # append zero and one to pvals and save as knots
    order = np.argsort(pvals)
    knots = np.concatenate([[0], pvals[order], [1]])
    heights = np.concatenate([[qvals[order][0]], qvals[order], [1]])
    calibration_error = total_integral(pi0, alpha, beta, knots, heights)
  elif calibrator == "lfdr":
    knots = gren.knots
    heights = np.minimum(pi_hat_0_gren / gren.slopes, 1)
    calibration_error = total_integral(pi0, alpha, beta, knots, heights)
  return calibration_error, bin_stats