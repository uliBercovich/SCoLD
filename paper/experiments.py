from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple, Any
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


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
