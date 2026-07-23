"""
Preview of the flexible second-step calibration (Flex) inside the SCoLD pipeline.

Generates, on a synthetic chromosome with distance-decaying LD (real CEU/AFR need
magenpy/stdpopsim in the analysis env):
  (a) paper/figures/flex_bias_by_distance.png  -- bias by distance bin (Flex's home turf)
  (b) paper/figures/flex_ld_score_rmse.png     -- LD-score RMSE (Supp S3 style)

Run from the repo root:  python paper/flex_preview.py
"""
from __future__ import annotations
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ld_estimates.estimators import r2, r2_BS, r2_batch
from ld_estimates.calibration import (
    build_calibration_models, create_calibrated_estimator,
    build_flex_models, create_flex_estimator,
)
from paper.experiments import (
    run_experiment, collect_bias_variance_results, compute_ld_scores,
)
from paper.plotting import plot_rmse_distribution_by_bin, plot_metric_distribution

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
N_VALUES = [5, 10, 25]
R2GRID = np.round(np.arange(0.0, 1.0001, 0.05), 3)
DEGREE = 2


def norm_ppf(p):
    """Acklam inverse-normal-CDF approximation (numpy only)."""
    p = np.asarray(p, dtype=float)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    x = np.empty_like(p)
    lo, hi = p < plow, p > phigh
    mid = ~(lo | hi)
    q = np.sqrt(-2 * np.log(p[lo]))
    x[lo] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = np.sqrt(-2 * np.log(1 - p[hi]))
    x[hi] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p[mid] - 0.5
    r = q * q
    x[mid] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    return x


def simulate_chromosome(n_ind=300, n_var=3000, length=2_000_000, ld_scale=40_000, seed=1):
    """Synthetic diploid genotypes (n_ind, n_var) with LD decaying over ~ld_scale bp."""
    rng = np.random.default_rng(seed)
    pos = np.sort(rng.uniform(0, length, size=n_var))
    af = rng.uniform(0.05, 0.5, size=n_var)
    H = 2 * n_ind
    z = np.empty((H, n_var))
    z[:, 0] = rng.standard_normal(H)
    for k in range(1, n_var):
        rho = np.exp(-(pos[k] - pos[k-1]) / ld_scale)
        z[:, k] = rho * z[:, k-1] + np.sqrt(1 - rho**2) * rng.standard_normal(H)
    thr = norm_ppf(1 - af)
    hap = (z > thr[None, :]).astype(np.int8)
    G = (hap[0::2] + hap[1::2]).astype(float)
    return G, pos


def build_estimators():
    """Returns dict-of-callables Cal/mCal/Flex, valid across N_VALUES."""
    cal_by_n, mcal_by_n, flex_master = {}, {}, {}
    for n in N_VALUES:
        master = {}
        bm = build_calibration_models(
            n=n, N_replicates=1500, r2_grid_to_model=R2GRID,
            estimators_to_calibrate={"r2": r2}, batch_estimators={"r2": r2_batch}, n_jobs=-1)
        master.setdefault("r2", {})[n] = bm["r2"]
        cal = create_calibrated_estimator(r2, "r2", master, "cal")
        cm = build_calibration_models(
            n=n, N_replicates=1500, r2_grid_to_model=R2GRID,
            estimators_to_calibrate={"r2_cal": cal}, n_jobs=-1)
        master.setdefault("r2_cal", {})[n] = cm["r2_cal"]
        mcal = create_calibrated_estimator(cal, "r2_cal", master, "indep")
        flex_master[n] = build_flex_models(
            n=n, cal_estimator=cal, r2_grid=R2GRID,
            N_replicates=800, degree=DEGREE, n_jobs=-1)
        cal_by_n[n], mcal_by_n[n] = cal, mcal
        print(f"[build n={n}] flex MACs={len(flex_master[n])}", flush=True)

    def Cal(G):  return cal_by_n[G.shape[0]](G)
    def mCal(G): return mcal_by_n[G.shape[0]](G)
    flex = create_flex_estimator(lambda G: cal_by_n[G.shape[0]](G), flex_master)
    return Cal, mCal, flex


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    t0 = time.time()
    Cal, mCal, Flex = build_estimators()
    G, pos = simulate_chromosome()
    print(f"[sim] G={G.shape}  built in {time.time()-t0:.1f}s", flush=True)

    # ---------- (a) bias by distance ----------
    bins = np.array([0, 1000, 10000, 100000, 1000000])
    est_bias = {"Samp": r2, "Cal": Cal, "mCal": mCal, "Flex": Flex}
    cmap_bias = {"Samp": "tab:gray", "Cal": "tab:blue", "mCal": "tab:cyan", "Flex": "tab:purple"}

    res = run_experiment(G, pos, N_VALUES, est_bias, bins,
                         max_pairs_per_bin=400, n_rep=40, seed=123, n_jobs=-1)
    df = collect_bias_variance_results(res, N_VALUES, bins, est_bias)
    df["binlabel"] = df["bin"].map({i: f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)})
    print("\n mean |bias| by n:\n",
          df.groupby(["n", "method"]).bias.apply(lambda s: np.mean(np.abs(s))).unstack("method").round(4).to_string())
    plot_rmse_distribution_by_bin(
        df, statistic="bias", title="Synthetic (bias by distance)",
        colormap=cmap_bias, outpath=os.path.join(FIGDIR, "flex_bias_by_distance.png"))
    print("saved flex_bias_by_distance.png", flush=True)

    # ---------- (b) LD-score RMSE ----------
    est_ld = {"BS": r2_BS, "Cal": Cal, "mCal": mCal, "Flex": Flex}
    cmap_ld = {"BS": "tab:orange", "Cal": "tab:blue", "mCal": "tab:cyan", "Flex": "tab:purple"}
    ld = compute_ld_scores(G, pos, est_ld, N_VALUES,
                           n_variants=60, n_reps=12, window_bp=150_000, seed=42, n_jobs=-1)
    print("\n mean LD-score RMSE by n:\n",
          ld.groupby(["n", "method"]).rmse.mean().unstack("method").round(3).to_string())
    plot_metric_distribution(
        ld, statistic="rmse", title="Synthetic (LD-score RMSE)",
        colormap=cmap_ld, outpath=os.path.join(FIGDIR, "flex_ld_score_rmse.png"))
    print(f"saved flex_ld_score_rmse.png\n total {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
