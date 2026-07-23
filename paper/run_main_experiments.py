"""Main-text experiments: regenerate the cached result CSVs used by the figure
notebooks (Figures 2 and 4, and Supplementary Sections S1--S3).

For each dataset it writes to paper/output/:
    metrics_{pop}.csv   bias / variance / RMSE by distance bin   (Figure 2, Tables S1/S2)
    f1_{pop}.csv        F1 of LD classification by threshold      (Figure 4)
    ldscore_{pop}.csv   LD-score RMSE                             (Supp. S3)
    summary_{pop}.csv   metrics averaged over replicates          (Tables S1/S2)

Datasets:
    CEU  -- 1000 Genomes EUR, chr22, MAF>5%   (needs `magenpy`)
    AFR  -- stdpopsim Africa_1T12 cache paper/data/AFR_chr22.npz  (no extra deps)

Usage:
    python paper/run_main_experiments.py --pop AFR
    python paper/run_main_experiments.py --pop CEU
    python paper/run_main_experiments.py --pop AFR --quick   # fast smoke test

Everything is seeded, so a given (--pop, reps) configuration is reproducible.
The figures themselves are drawn from these CSVs in paper/notebooks/main_figures.ipynb.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

from ld_estimates.estimators import r2, r2_BS, r2_Rag, r2_Supp, r2_batch
from ld_estimates.calibration import (
    build_calibration_models, create_calibrated_estimator,
    build_flex_models, create_flex_estimator,
)
from paper.experiments import (
    run_experiment, collect_bias_variance_results,
    compute_f1_by_threshold, compute_ld_scores,
)

OUT = os.path.join(HERE, "output")

# ---- parameters (defaults mirror the paper; --quick lowers the reps) ----
N_VALUES      = [5, 10, 25]
R2_GRID       = np.round(np.linspace(0, 1, 21), 3)
N_REP_BUILD   = 5000       # replicates for the Cal / mCal calibration curves
N_REP_FLEX    = 2000       # replicates for the Flex polynomial fit (per MAC)
FLEX_DEGREE   = 2
DISTANCE_BINS = np.array([0, 1000, 10000, 100000, 1000000])
N_REP_EXP     = 100        # bootstrap replicates for the RMSE/bias experiment
MAX_PAIRS_BIN = 1000
F1_THRESHOLDS = [0.2, 0.5, 0.8]
LD_N_VARIANTS = 1000
LD_N_REPS     = 50
LD_WINDOW_BP  = 500_000
SEED          = 42
N_JOBS        = -1

t0 = time.time()
def log(m): print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)


def maf_filter(G, positions, maf_threshold=0.05):
    """Filter an unphased (0/1/2) genotype matrix + positions by MAF."""
    if G.shape[1] == 0:
        return G, positions
    alt = np.sum(G, axis=0)
    nchrom = 2 * G.shape[0]
    af = alt / nchrom
    maf = np.minimum(af, 1 - af)
    mask = (maf >= maf_threshold) & (alt > 0) & (alt < nchrom)
    return G[:, mask], positions[mask]


def load_dataset(pop: str):
    """Return (G, pos) for pop in {CEU, AFR}."""
    if pop == "CEU":
        import magenpy as mgp
        gdl = mgp.GWADataLoader(mgp.tgp_eur_data_path())
        G = gdl.genotype[22].to_numpy().astype(float)
        pos = np.asarray(gdl.genotype[22].bp_pos)
        return maf_filter(G, pos, 0.05)
    if pop == "AFR":
        cache = os.path.join(HERE, "data", "AFR_chr22.npz")
        if not os.path.exists(cache):
            raise FileNotFoundError(f"{cache} not found; run paper/gen_afr.py to create it.")
        d = np.load(cache)
        return d["G"].astype(float), d["pos"]
    raise ValueError(f"unknown pop {pop!r} (expected CEU or AFR)")


def build_estimators(n_rep_build, n_rep_flex):
    """Build the seven r^2 estimators used in the main text, per sample size."""
    master = {}
    cal_by_n, mcal_by_n, flex_master = {}, {}, {}
    for n in N_VALUES:
        bm = build_calibration_models(
            n=n, N_replicates=n_rep_build, r2_grid_to_model=R2_GRID,
            estimators_to_calibrate={"r2": r2}, batch_estimators={"r2": r2_batch},
            n_jobs=N_JOBS)
        master.setdefault("r2", {})[n] = bm["r2"]
        cal = create_calibrated_estimator(r2, "r2", master, "cal")

        cm = build_calibration_models(
            n=n, N_replicates=n_rep_build, r2_grid_to_model=R2_GRID,
            estimators_to_calibrate={"r2_cal": cal}, n_jobs=N_JOBS)
        master.setdefault("r2_cal", {})[n] = cm["r2_cal"]
        mcal = create_calibrated_estimator(cal, "r2_cal", master, "indep")

        flex_master[n] = build_flex_models(
            n=n, cal_estimator=cal, r2_grid=R2_GRID,
            N_replicates=n_rep_flex, degree=FLEX_DEGREE, n_jobs=N_JOBS)
        cal_by_n[n], mcal_by_n[n] = cal, mcal
        log(f"  built calibrators for n={n}")

    def Cal(G):  return cal_by_n[G.shape[0]](G)
    def mCal(G): return mcal_by_n[G.shape[0]](G)
    Flex = create_flex_estimator(lambda G: cal_by_n[G.shape[0]](G), flex_master)

    return {"Samp": r2, "BS": r2_BS, "Rag": r2_Rag, "Supp": r2_Supp,
            "Cal": Cal, "mCal": mCal, "Flex": Flex}


def main(pop: str, quick: bool):
    os.makedirs(OUT, exist_ok=True)
    n_rep_build = 200 if quick else N_REP_BUILD
    n_rep_flex  = 100 if quick else N_REP_FLEX
    n_rep_exp   = 5   if quick else N_REP_EXP
    ld_n_reps   = 3   if quick else LD_N_REPS
    ld_n_var    = 100 if quick else LD_N_VARIANTS

    G, pos = load_dataset(pop)
    log(f"{pop}: G={G.shape}")

    estimators = build_estimators(n_rep_build, n_rep_flex)
    log("estimators ready")

    # --- Figure 2 / Tables S1-S2: bias, variance, RMSE by distance bin ---
    res = run_experiment(G, pos, N_VALUES, estimators, DISTANCE_BINS,
                         max_pairs_per_bin=MAX_PAIRS_BIN, n_rep=n_rep_exp,
                         seed=SEED, n_jobs=N_JOBS)
    metrics = collect_bias_variance_results(res, N_VALUES, DISTANCE_BINS, estimators)
    binlabel = {i: f"{int(DISTANCE_BINS[i])}-{int(DISTANCE_BINS[i+1])}"
                for i in range(len(DISTANCE_BINS) - 1)}
    metrics["binlabel"] = metrics["bin"].map(binlabel)
    metrics["pop"] = pop
    metrics.to_csv(os.path.join(OUT, f"metrics_{pop}.csv"), index=False)
    log(f"wrote metrics_{pop}.csv ({len(metrics)} rows)")

    summary = (metrics.groupby(["n", "binlabel", "method"])[["bias", "variance", "rmse"]]
               .mean().reset_index())
    summary.to_csv(os.path.join(OUT, f"summary_{pop}.csv"), index=False)
    log(f"wrote summary_{pop}.csv")

    # --- Figure 4: F1 of LD classification by threshold (reuses the same pairs) ---
    f1 = compute_f1_by_threshold(res, N_VALUES, DISTANCE_BINS, estimators, F1_THRESHOLDS)
    f1.to_csv(os.path.join(OUT, f"f1_{pop}.csv"), index=False)
    log(f"wrote f1_{pop}.csv ({len(f1)} rows)")

    # --- Supp. S3: LD-score RMSE ---
    ld_estimators = {k: v for k, v in estimators.items() if k != "Samp"}
    ld = compute_ld_scores(G, pos, ld_estimators, N_VALUES,
                           n_variants=ld_n_var, n_reps=ld_n_reps,
                           window_bp=LD_WINDOW_BP, seed=SEED, n_jobs=N_JOBS)
    ld.to_csv(os.path.join(OUT, f"ldscore_{pop}.csv"), index=False)
    log(f"wrote ldscore_{pop}.csv ({len(ld)} rows)")
    log("DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pop", choices=["CEU", "AFR"], required=True)
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke test with reduced replicate counts")
    args = ap.parse_args()
    main(args.pop, args.quick)
