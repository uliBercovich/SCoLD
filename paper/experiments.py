from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ld_estimates.calibration import generate_genotypes


R2Estimator = Callable[[np.ndarray], float]  # expects pair-genotypes (n,2)


def variance_filter(G: np.ndarray) -> np.ndarray:
    """
    Returns boolean mask of variants with non-zero variance across samples.
    G shape: (n_samples, n_variants)
    """
    return np.var(G, axis=0) > 0


def compute_true_r2(g1: np.ndarray, g2: np.ndarray) -> float:
    """
    Fast r^2 for two genotype vectors (no corrcoef overhead).
    Returns nan if variance is zero.
    """
    g1c = g1 - g1.mean()
    g2c = g2 - g2.mean()
    num = float(np.sum(g1c * g2c) ** 2)
    den = float(np.sum(g1c ** 2) * np.sum(g2c ** 2))
    return num / den if den > 0 else np.nan


def _sample_pairs_by_distance_bins(
    pos: np.ndarray,
    idx_variants: np.ndarray,
    distance_bins: np.ndarray,
) -> Dict[int, list[Tuple[int, int]]]:
    """
    Pre-compute candidate pairs per bin for a set of variant indices.
    Uses a forward scan + searchsorted window (like in your notebook).
    """
    pos = np.asarray(pos)
    idx_variants = np.asarray(idx_variants, dtype=int)

    # Ensure variants are in increasing genomic position order for window logic
    order = np.argsort(pos[idx_variants])
    idx_sorted = idx_variants[order]
    pos_sorted = pos[idx_sorted]

    num_bins = len(distance_bins) - 1
    candidates_per_bin: Dict[int, list[Tuple[int, int]]] = {b: [] for b in range(num_bins)}
    maxdist = distance_bins[-1]

    for ii, i in enumerate(idx_sorted):
        pi = pos[i]
        # only look forward within maxdist
        end = np.searchsorted(pos_sorted, pi + maxdist, side="right")
        for jj in range(ii + 1, end):
            j = idx_sorted[jj]
            d = pos[j] - pi
            b = int(np.digitize(d, distance_bins) - 1)
            if 0 <= b < num_bins:
                candidates_per_bin[b].append((i, j))

    return candidates_per_bin


def run_replication(
    rep_id: int,
    n_values: Sequence[int],
    G: np.ndarray,
    pos: np.ndarray,
    r2_estimators: Dict[str, R2Estimator],
    distance_bins: np.ndarray,
    max_pairs_per_bin: int,
    base_seed: int,
) -> Dict[int, Dict[int, Dict[str, np.ndarray]]]:
    """
    One bootstrap replication:
      - subsample individuals for each n in n_values
      - variance filter variants in the subsample
      - for each distance bin, sample up to max_pairs_per_bin pairs
      - compute true r^2 on full G, estimated r^2 on subsample G_sub

    Returns a nested dict:
      results[rep_id][n][bin_id]["true_r2"] = array
      results[rep_id][n][bin_id][method]    = array
    """
    rng = np.random.default_rng(base_seed + rep_id)
    n_samples, _ = G.shape

    rep_results: Dict[int, Dict[int, Dict[str, np.ndarray]]] = {rep_id: {}}

    for n in n_values:
        # Subsample individuals
        idx_sub = rng.choice(n_samples, size=int(n), replace=False)
        G_sub = G[idx_sub, :]

        # Filter polymorphic variants in subsample
        keep = variance_filter(G_sub)
        idx_variants = np.where(keep)[0]
        if idx_variants.size == 0:
            continue

        candidates_per_bin = _sample_pairs_by_distance_bins(pos, idx_variants, distance_bins)

        n_bins = len(distance_bins) - 1
        bin_results: Dict[int, Dict[str, Any]] = {
            b: {name: [] for name in r2_estimators.keys()} | {"true_r2": []}
            for b in range(n_bins)
        }

        for b in range(n_bins):
            candidates = candidates_per_bin[b]
            if len(candidates) == 0:
                continue

            k = min(max_pairs_per_bin, len(candidates))
            chosen_idx = rng.choice(len(candidates), size=k, replace=False)
            sampled_pairs = [candidates[t] for t in chosen_idx]

            for i, j in sampled_pairs:
                # True r2 on FULL data
                tr2 = compute_true_r2(G[:, i], G[:, j])
                bin_results[b]["true_r2"].append(tr2)

                # Estimated r2 on SUBSAMPLE (pair matrix n x 2)
                pair = np.column_stack([G_sub[:, i], G_sub[:, j]])
                for name, fn in r2_estimators.items():
                    bin_results[b][name].append(fn(pair))

        # Convert lists to arrays
        for b in range(n_bins):
            for key, vals in bin_results[b].items():
                bin_results[b][key] = np.asarray(vals, dtype=float)

        rep_results[rep_id][int(n)] = bin_results

    return rep_results


def run_experiment(
    G: np.ndarray,
    pos: np.ndarray,
    n_values: Sequence[int],
    r2_estimators: Dict[str, R2Estimator],
    distance_bins: np.ndarray,
    max_pairs_per_bin: int = 1000,
    n_rep: int = 10,
    seed: int = 42,
    n_jobs: int = -1,
) -> Dict[int, Dict[int, Dict[int, Dict[str, np.ndarray]]]]:
    """
    Parallel wrapper over replications. Merges dict outputs into one dict.
    """
    results_list = Parallel(n_jobs=n_jobs)(
        delayed(run_replication)(
            rep_id=rep_id,
            n_values=n_values,
            G=G,
            pos=pos,
            r2_estimators=r2_estimators,
            distance_bins=distance_bins,
            max_pairs_per_bin=max_pairs_per_bin,
            base_seed=seed,
        )
        for rep_id in range(n_rep)
    )

    all_results: Dict[int, Any] = {}
    for d in results_list:
        all_results.update(d)
    return all_results


def collect_bias_variance_results(
    results: Dict[int, Dict[int, Dict[int, Dict[str, np.ndarray]]]],
    n_values: Sequence[int],
    distance_bins: np.ndarray,
    r2_estimators: Dict[str, R2Estimator],
) -> pd.DataFrame:
    """
    Converts raw results dict into a tidy DataFrame with bias/variance/rmse per:
      (rep, n, bin, method)

    bias    = mean(est - true)
    variance= var(est - true)
    rmse    = sqrt(bias^2 + variance)
    """
    records = []
    n_bins = len(distance_bins) - 1

    for rep_id, rep_data in results.items():
        for n in n_values:
            if n not in rep_data:
                continue

            for b in range(n_bins):
                bin_data = rep_data[n].get(b, None)
                if not bin_data:
                    continue

                true_vals = bin_data.get("true_r2", np.array([]))
                if true_vals.size == 0:
                    continue

                for method in r2_estimators.keys():
                    est_vals = bin_data.get(method, np.array([]))
                    if est_vals.size == 0:
                        continue

                    errors = est_vals - true_vals
                    bias = float(np.mean(errors))
                    var = float(np.var(errors))
                    rmse = float(np.sqrt(bias * bias + var))

                    records.append(
                        {
                            "rep": rep_id,
                            "n": int(n),
                            "bin": int(b),
                            "method": method,
                            "bias": bias,
                            "variance": var,
                            "rmse": rmse,
                        }
                    )

    return pd.DataFrame.from_records(records)


def simulate_metrics_curve(
    n: int,
    p_s: float,
    p_t: float,
    r2_grid: np.ndarray,
    Nrep: int,
    r2_estimator: R2Estimator,
    pseudohaploid: bool = False,
    seed: int | None = None,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Simulates bias, variance, and RMSE of r2_estimator relative to true ρ².

    Returns (x_coords, bias, variance, rmse) for each reachable point in
    r2_grid. bias = E[estimator(G)] - true_ρ².
    """
    rng = np.random.default_rng(seed)
    x_coords: List[float] = []
    bias_list: List[float] = []
    var_list:  List[float] = []
    rmse_list: List[float] = []

    for true_r2 in r2_grid:
        var_prod = p_s * (1 - p_s) * p_t * (1 - p_t)
        if var_prod <= 0:
            continue
        D_required = float(np.sqrt(true_r2 * var_prod))
        if D_required < max(-p_s * p_t, -(1 - p_s) * (1 - p_t)) - 1e-8:
            continue
        if D_required > min(p_s * (1 - p_t), (1 - p_s) * p_t) + 1e-8:
            continue

        obs: List[float] = []
        for _ in range(Nrep):
            G = generate_genotypes(n, p_s, p_t, D_required, pseudohaploid=pseudohaploid, rng=rng)
            if G is None:
                continue
            val = r2_estimator(G)
            if not np.isnan(val):
                obs.append(float(val))

        if obs:
            arr = np.asarray(obs)
            b = float(arr.mean()) - float(true_r2)
            v = float(arr.var())
            x_coords.append(float(true_r2))
            bias_list.append(b)
            var_list.append(v)
            rmse_list.append(float(np.sqrt(b ** 2 + v)))

    return x_coords, bias_list, var_list, rmse_list


def simulate_bias_curve_data(
    n: int,
    p_s: float,
    p_t: float,
    r2_grid: np.ndarray,
    Nrep: int,
    r2_estimator: R2Estimator,
    pseudohaploid: bool = False,
    quantiles: Tuple[float, float] = (0.25, 0.75),
    seed: int | None = None,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Simulates the bias curve for one (n, p_s, p_t) scenario.

    Returns (x_coords, y_mean, y_low, y_high) where x is the true ρ² grid,
    y_mean is the mean observed r² across Nrep replicates, and y_low/y_high
    are the lower and upper quantile bounds (default 25th–75th percentile).
    Quantile bounds are naturally asymmetric and stay within [0, 1].
    Points where the required D is out of range or all replicates fail are skipped.

    Set pseudohaploid=True to use the aDNA pseudohaploid data model.
    """
    rng = np.random.default_rng(seed)
    x_coords: List[float] = []
    y_mean: List[float] = []
    y_low: List[float] = []
    y_high: List[float] = []

    for true_r2 in r2_grid:
        var_prod = p_s * (1 - p_s) * p_t * (1 - p_t)
        if var_prod <= 0:
            continue
        D_required = float(np.sqrt(true_r2 * var_prod))
        if D_required < max(-p_s * p_t, -(1 - p_s) * (1 - p_t)) - 1e-8:
            continue
        if D_required > min(p_s * (1 - p_t), (1 - p_s) * p_t) + 1e-8:
            continue

        obs: List[float] = []
        for _ in range(Nrep):
            G = generate_genotypes(n, p_s, p_t, D_required, pseudohaploid=pseudohaploid, rng=rng)
            if G is None:
                continue
            val = r2_estimator(G)
            if not np.isnan(val):
                obs.append(float(val))

        if obs:
            arr = np.asarray(obs)
            x_coords.append(float(true_r2))
            y_mean.append(float(arr.mean()))
            y_low.append(float(np.percentile(arr, quantiles[0] * 100)))
            y_high.append(float(np.percentile(arr, quantiles[1] * 100)))

    return x_coords, y_mean, y_low, y_high


def _ld_scores_one_rep(
    rep_id: int,
    n_values: Sequence[int],
    G: np.ndarray,
    pos: np.ndarray,
    r2_estimators: Dict[str, R2Estimator],
    n_variants: int,
    window_bp: float,
    base_seed: int,
) -> list[dict]:
    """
    One LD-score replication. For each n, subsample individuals, pick target
    variants, and compute each variant's LD score = sum of r^2 with neighbours
    within +/- window_bp. Ground truth uses the FULL sample; estimates use the
    subsample. Returns per-(n, method) RMSE across the sampled target variants.
    """
    rng = np.random.default_rng(base_seed + rep_id)
    pos = np.asarray(pos)
    n_samples, _ = G.shape
    records = []

    for n in n_values:
        idx_sub = rng.choice(n_samples, size=int(n), replace=False)
        G_sub = G[idx_sub, :]
        idx_v = np.where(variance_filter(G_sub))[0]
        if idx_v.size == 0:
            continue

        n_take = min(n_variants, idx_v.size)
        targets = rng.choice(idx_v, size=n_take, replace=False)

        true_scores = np.zeros(n_take)
        est_scores = {m: np.zeros(n_take) for m in r2_estimators}

        for t, j in enumerate(targets):
            in_win = idx_v[(np.abs(pos[idx_v] - pos[j]) <= window_bp) & (idx_v != j)]
            for k in in_win:
                tr = compute_true_r2(G[:, j], G[:, k])
                if not np.isnan(tr):
                    true_scores[t] += tr
                pair = np.column_stack([G_sub[:, j], G_sub[:, k]])
                for m, fn in r2_estimators.items():
                    v = fn(pair)
                    if not np.isnan(v):
                        est_scores[m][t] += v

        for m in r2_estimators:
            rmse = float(np.sqrt(np.mean((est_scores[m] - true_scores) ** 2)))
            records.append({"rep": rep_id, "n": int(n), "method": m, "rmse": rmse})

    return records


def compute_f1_by_threshold(
    results: Dict[int, Dict[int, Dict[int, Dict[str, np.ndarray]]]],
    n_values: Sequence[int],
    distance_bins: np.ndarray,
    r2_estimators: Dict[str, R2Estimator],
    thresholds: Sequence[float] = (0.2, 0.5, 0.8),
) -> pd.DataFrame:
    """
    F1 score for LD-based classification, reusing the pairs already sampled by
    run_experiment. For each (rep, n, threshold, method) we pool the pairs across
    distance bins and treat the task as a binary classification: a pair is a true
    positive of "high LD" when its population value exceeds the threshold
    (true_r2 >= thr) and it is predicted positive when the estimated value does
    (est >= thr). Precision = TP/(TP+FP) tracks over-pruning and recall =
    TP/(TP+FN) tracks under-pruning; F1 is their harmonic mean. Returns a tidy
    DataFrame with columns (rep, n, thr, method, f1).
    """
    records = []
    n_bins = len(distance_bins) - 1

    for rep_id, rep_data in results.items():
        for n in n_values:
            if n not in rep_data:
                continue

            # pool pairs across all distance bins for this (rep, n)
            true_all = np.concatenate(
                [rep_data[n][b]["true_r2"] for b in range(n_bins)
                 if b in rep_data[n] and rep_data[n][b].get("true_r2", np.array([])).size]
            ) if any(b in rep_data[n] for b in range(n_bins)) else np.array([])
            if true_all.size == 0:
                continue

            for method in r2_estimators.keys():
                est_all = np.concatenate(
                    [rep_data[n][b][method] for b in range(n_bins)
                     if b in rep_data[n] and rep_data[n][b].get(method, np.array([])).size]
                )
                ok = np.isfinite(true_all) & np.isfinite(est_all)
                t, e = true_all[ok], est_all[ok]
                for thr in thresholds:
                    truth = t >= thr
                    pred = e >= thr
                    tp = int(np.sum(truth & pred))
                    fp = int(np.sum(~truth & pred))
                    fn = int(np.sum(truth & ~pred))
                    denom = 2 * tp + fp + fn
                    f1 = (2 * tp / denom) if denom > 0 else np.nan
                    records.append({"rep": rep_id, "n": int(n), "thr": float(thr),
                                    "method": method, "f1": f1})

    return pd.DataFrame.from_records(records)


def compute_ld_scores(
    G: np.ndarray,
    pos: np.ndarray,
    r2_estimators: Dict[str, R2Estimator],
    n_values: Sequence[int],
    n_variants: int = 1000,
    n_reps: int = 50,
    window_bp: float = 500_000,
    seed: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    LD-score RMSE experiment (mirrors Supplementary Section S3). Returns a tidy
    DataFrame with columns (rep, n, method, rmse), one row per replicate/sample
    size/estimator, ready for plot_metric_distribution.
    """
    out = Parallel(n_jobs=n_jobs)(
        delayed(_ld_scores_one_rep)(
            rep_id, n_values, G, pos, r2_estimators, n_variants, window_bp, seed,
        )
        for rep_id in range(n_reps)
    )
    records = [r for rep in out for r in rep]
    return pd.DataFrame.from_records(records)
