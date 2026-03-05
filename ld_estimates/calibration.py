# calibration.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Literal, Tuple, DefaultDict, Any
from collections import defaultdict

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


CalibrationType = Literal["general", "independence", "meancorrection"]
EstimatorFn = Callable[[np.ndarray], float]  # expects G with shape (n, 2) for a SNP pair


def generate_genotypes(n: int, pA: float, pB: float, D: float, rng: np.random.Generator | None = None):
    """
    Simulates one valid UNPHASED diploid genotype matrix G for n individuals at 2 loci.
    Returns array of shape (n, 2) with entries in {0,1,2}, or None if invalid.
    """
    rng = np.random.default_rng() if rng is None else rng

    minD = max(-pA * pB, -(1 - pA) * (1 - pB))
    maxD = min(pA * (1 - pB), (1 - pA) * pB)

    if D < minD - 1e-8 or D > maxD + 1e-8:
        return None
    if np.isclose(D, maxD):
        D = maxD
    elif np.isclose(D, minD):
        D = minD

    pAB = D + pA * pB
    pAb = pA - pAB
    paB = pB - pAB
    pab = 1 - pAB - pAb - paB

    hap_freqs = np.array([pab, paB, pAb, pAB], dtype=float)
    hap_freqs = np.maximum(0.0, hap_freqs)
    hap_freqs = hap_freqs / hap_freqs.sum()

    # sample 2n haplotypes, pair into n diploid individuals
    # hap index: 0=pab, 1=paB, 2=pAb, 3=pAB
    for _ in range(50):  # avoid pathological infinite loops
        hap_idx = rng.choice(4, size=2 * n, p=hap_freqs)
        allelesA = (hap_idx >= 2).astype(int)  # hap types 2,3 carry A
        allelesB = (hap_idx % 2 == 1).astype(int)  # hap types 1,3 carry B

        G = np.column_stack(
            [
                allelesA[0::2] + allelesA[1::2],
                allelesB[0::2] + allelesB[1::2],
            ]
        )

        if np.var(G[:, 0], ddof=1) > 1e-8 and np.var(G[:, 1], ddof=1) > 1e-8:
            return G

    return None


def _mac_key_from_ps_pt(n: int, ps: float, pt: float) -> Tuple[int, int]:
    macs = int(round(ps * 2 * n))
    mact = int(round(pt * 2 * n))
    return tuple(sorted((macs, mact)))


def _simulate_one_scenario_all_estimators(
    n: int,
    Nrep: int,
    ps: float,
    pt: float,
    D: float,
    true_r2: float,
    estimators: Dict[str, EstimatorFn],
    seed: int | None = None,
):
    rng = np.random.default_rng(seed)

    results_raw: Dict[str, list[float]] = {name: [] for name in estimators.keys()}

    for _ in range(Nrep):
        G = generate_genotypes(n, ps, pt, D, rng=rng)
        if G is None:
            continue
        for name, fn in estimators.items():
            r2obs = fn(G)
            if not np.isnan(r2obs):
                results_raw[name].append(float(r2obs))

    results_mean = {name: float(np.mean(vals)) for name, vals in results_raw.items() if len(vals) > 0}

    mac_key = _mac_key_from_ps_pt(n, ps, pt)
    true_r2_bin = round(float(true_r2), 3)
    return mac_key, true_r2_bin, results_mean


def build_calibration_models(
    n: int,
    N_replicates: int,
    r2_grid_to_model: np.ndarray,
    estimators_to_calibrate: Dict[str, EstimatorFn],
    n_jobs: int = -1,
    seeds: Iterable[int] | None = None,
):
    """
    Builds calibration models for ALL provided estimators for a single sample size n.

    Returns:
      dict[estimator_name][mac_key][true_r2_bin] = mean_observed_r2
      where mac_key = (min(MAC_s, MAC_t), max(MAC_s, MAC_t)) with MAC = round(p*2n).
    """
    mac_values = np.arange(1, n + 1)  # exclude 0 and 2n; mirrors your notebook choice [file:25]

    valid_scenarios = []
    for i, macs in enumerate(mac_values):
        for mact in mac_values[i:]:
            ps = macs / (2 * n)
            pt = mact / (2 * n)
            for true_r2 in r2_grid_to_model:
                var_prod = ps * (1 - ps) * pt * (1 - pt)
                if var_prod <= 0:
                    continue
                D_required = float(np.sqrt(true_r2 * var_prod))
                maxD = float(min(ps * (1 - pt), (1 - ps) * pt))
                if D_required > maxD + 1e-6:
                    continue
                valid_scenarios.append((ps, pt, D_required, float(true_r2)))

    if seeds is None:
        # deterministic-ish default: one seed per scenario
        seeds = range(len(valid_scenarios))

    seeds = list(seeds)
    if len(seeds) < len(valid_scenarios):
        raise ValueError("Provide at least as many seeds as valid scenarios, or leave seeds=None.")

    results_list = Parallel(n_jobs=n_jobs)(
        delayed(_simulate_one_scenario_all_estimators)(
            n,
            N_replicates,
            ps,
            pt,
            D,
            true_r2,
            estimators_to_calibrate,
            seed=seeds[k],
        )
        for k, (ps, pt, D, true_r2) in enumerate(tqdm(valid_scenarios, desc=f"Simulating for n={n}"))
    )

    final_models: Dict[str, Any] = {name: defaultdict(lambda: defaultdict(float)) for name in estimators_to_calibrate.keys()}

    # aggregate
    for mac_key, true_r2_bin, scenario_results in results_list:
        for name, mean_obs in scenario_results.items():
            final_models[name][mac_key][true_r2_bin] = mean_obs

    # convert defaultdicts to plain dicts
    return {name: {k: dict(v) for k, v in model.items()} for name, model in final_models.items()}


def apply_calibration(
    G: np.ndarray,
    base_estimator: EstimatorFn,
    calibration_model_for_n: dict,
    calibration_type: CalibrationType = "general",
) -> float:
    """
    Applies one of the three calibration methods to one pair-genotype matrix G (n x 2).
    """
    r2obs = float(base_estimator(G))
    if np.isnan(r2obs):
        return np.nan

    n = int(G.shape[0])

    macs = int(min(np.sum(G[:, 0]), 2 * n - np.sum(G[:, 0])))
    mact = int(min(np.sum(G[:, 1]), 2 * n - np.sum(G[:, 1])))
    mac_key = tuple(sorted((macs, mact)))

    if mac_key not in calibration_model_for_n:
        return r2obs

    scenario_data = calibration_model_for_n[mac_key]
    if (scenario_data is None) or (len(scenario_data) < 2):
        return r2obs

    y_true = np.array(list(scenario_data.keys()), dtype=float)       # true r2
    x_mean = np.array(list(scenario_data.values()), dtype=float)     # mean observed r2

    # (optional) make sure monotone in x for interpolation
    order = np.argsort(x_mean)
    x_mean = x_mean[order]
    y_true = y_true[order]

    if calibration_type == "general":
        calibrated = float(np.interp(r2obs, x_mean, y_true))
        return max(0.0, calibrated)

    if calibration_type == "independence":
        # your notebook uses the first point as an "error term" under independence [file:25]
        err = float(x_mean[0])
        return float(1.0 - (1.0 - r2obs) / (1.0 - err))

    if calibration_type == "meancorrection":
        mean_error = x_mean - y_true
        pred_err = float(np.interp(r2obs, x_mean, mean_error))
        return float(r2obs - pred_err)

    raise ValueError(f"Unknown calibration_type: {calibration_type}")


def create_calibrated_estimator(
    base_estimator: EstimatorFn,
    estimator_name: str,
    master_model: dict,
    calibration_type: CalibrationType = "general",
) -> EstimatorFn:
    """
    Returns a function G -> calibrated r2, using master_model[estimator_name][n].
    """
    suffix = {"general": "c", "independence": "ic", "meancorrection": "mc"}[calibration_type]
    new_name = f"{estimator_name}{suffix}"

    def calibrated(G: np.ndarray) -> float:
        n = int(G.shape[0])
        if estimator_name in master_model and n in master_model[estimator_name]:
            model_for_n = master_model[estimator_name][n]
            return apply_calibration(G, base_estimator, model_for_n, calibration_type=calibration_type)
        return float(base_estimator(G))

    calibrated.__name__ = new_name
    calibrated.__doc__ = f"{calibration_type} calibrated version of {estimator_name}"
    return calibrated
