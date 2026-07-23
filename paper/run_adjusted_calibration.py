"""Calibrating an LD matrix without the genotypes: validation on structured 1000G data.

Design (structured sample of Bercovich et al. 2025: 50 CEU + 50 YRI + 50 ASW, chr22, MAF>5%):

  truth      : r2_adj computed on all 150 individuals (k=1 PC)
  replicate  : subsample 9 CEU + 9 YRI + 8 ASW = 26 individuals, refit the PCA on the
               subsample, recompute r2_adj, and calibrate it.

  k = 1 PC  =>  n_eff = 26 - 1 = 25, so we reuse the n=25 curves of the main text.

All four estimators are sample-size corrections applied to the SAME adjusted LD matrix, and
all of them use only the matrix and n -- never the genotypes:

  Samp   no sample-size correction (the raw adjusted r2)
  BS     the Bulik-Sullivan correction r2 - (1-r2)/(n_eff-2), which is what Bercovich et al.
         (2025) already use to correct the adjusted LD; the incumbent
  Cal    our one-step calibration at (p_eff, n_eff)
  mCal   our two-step calibration (mean-centred under independence)

`std` (the standard, unadjusted r2) is also recorded, as a reference for the structure bias.
Sites with effective frequency below P_EFF_MIN are discarded.

Writes paper/output/adjcal_metrics.csv (+ adjcal_raw.csv.gz).
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ld_estimates.estimators import r2, r2_batch
from ld_estimates.calibration import build_calibration_models
from ld_estimates.adjusted import (
    fit_residuals, adjusted_D_from_residuals, effective_freq, effective_mac,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
DATA = os.path.join(HERE, "data", "1000G_struct_chr22.npz")

K_PCS = 1
DESIGNS = [
    [("CEU", 5), ("YRI", 5), ("ASW", 6)],     # n = 16 -> n_eff = 15
    [("CEU", 9), ("YRI", 9), ("ASW", 8)],     # n = 26 -> n_eff = 25
]
N_REP = 100
R2_GRID = np.round(np.linspace(0, 1, 21), 3)
N_REP_BUILD = 5000
N_REP_BUILD_CAL = 2000
DISTANCE_BINS = np.array([0, 1000, 10000, 100000, 1000000])
MAX_PAIRS_PER_BIN = 400
P_EFF_MIN = 0.05
SEED = 7

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)


def _curve(mac_key, model):
    sc = model.get(mac_key)
    if not sc or len(sc) < 2:
        return None
    y = np.array(list(sc.keys()), dtype=float)      # true rho^2
    x = np.array(list(sc.values()), dtype=float)    # mean observed
    o = np.argsort(x)
    return x[o], y[o]


def cal_step(r2obs, mac_key, model_r2):
    """One-step calibration: invert the mean curve. Input is a value + a MAC key, not genotypes."""
    c = _curve(mac_key, model_r2)
    if c is None or not np.isfinite(r2obs):
        return float(r2obs) if np.isfinite(r2obs) else np.nan
    x, y = c
    return max(0.0, float(np.interp(r2obs, x, y)))


def bs_step(r2obs, n_eff):
    """Bulik-Sullivan sample-size correction: r2 - (1-r2)/(n-2). Needs only the value and n.

    This is the correction Bercovich et al. (2025) apply to the adjusted LD, so it is the
    incumbent we compare against. We give it the effective sample size n_eff = n - k, which is
    the residual degrees of freedom, rather than n -- the fairer choice for the adjusted matrix.
    """
    if not np.isfinite(r2obs) or n_eff <= 2:
        return np.nan
    return float(r2obs - (1.0 - r2obs) / (n_eff - 2))


def mcal_step(cal_value, mac_key, model_cal):
    """Second step: mean-centre under independence, h(x) = 1 - (1-x)/(1-e), e = E[Cal | rho2=0]."""
    c = _curve(mac_key, model_cal)
    if c is None or not np.isfinite(cal_value):
        return float(cal_value) if np.isfinite(cal_value) else np.nan
    x, _ = c
    e = float(x[0])
    if not np.isfinite(e) or e >= 1.0:
        return float(cal_value)
    return float(1.0 - (1.0 - cal_value) / (1.0 - e))


def pairs_by_bin(pos, idx, rng):
    idx = np.asarray(idx)
    p = pos[idx]
    o = np.argsort(p)
    idx, p = idx[o], p[o]
    out = {}
    for b in range(len(DISTANCE_BINS) - 1):
        lo, hi = DISTANCE_BINS[b], DISTANCE_BINS[b + 1]
        got, tries = [], 0
        while len(got) < MAX_PAIRS_PER_BIN and tries < MAX_PAIRS_PER_BIN * 60:
            tries += 1
            i = rng.integers(0, len(idx) - 1)
            j0 = np.searchsorted(p, p[i] + lo, side="left")
            j1 = np.searchsorted(p, p[i] + hi, side="right")
            if j1 <= max(j0, i + 1):
                continue
            j = rng.integers(max(j0, i + 1), j1)
            if j < len(idx):
                got.append((int(idx[i]), int(idx[j])))
        out[b] = got
    return out


def one_rep(rep, G, pos, pop, R_full, design, model_r2, model_cal):
    rng = np.random.default_rng(SEED + 1000 * sum(s for _, s in design) + rep)
    n_full = G.shape[0]
    sel = np.concatenate([rng.choice(np.where(pop == q)[0], size=s, replace=False)
                          for q, s in design])
    Gs = G[sel]
    n = Gs.shape[0]
    n_eff = n - K_PCS

    ac = Gs.sum(axis=0)
    poly = np.where((ac > 0) & (ac < 2 * n))[0]
    if len(poly) < 100:
        return []

    Rs = fit_residuals(Gs, k=K_PCS)

    recs = []
    for b, prs in pairs_by_bin(pos, poly, rng).items():
        for (i, j) in prs:
            # ---- truth: adjusted r2 on the full 150 ----
            Df = adjusted_D_from_residuals(R_full[:, [i, j]], n_full, K_PCS)
            df_den = Df[0, 0] * Df[1, 1]
            if not np.isfinite(df_den) or df_den <= 0:
                continue
            truth = float(Df[0, 1] ** 2 / df_den)

            # ---- the subsample's adjusted covariance: the ONLY input from here on ----
            D = adjusted_D_from_residuals(Rs[:, [i, j]], n, K_PCS)
            den = D[0, 0] * D[1, 1]
            if not np.isfinite(den) or den <= 0:
                continue
            v_adj = float(D[0, 1] ** 2 / den)

            # effective frequency filter (computable from D alone)
            pe_s, pe_t = effective_freq(D[0, 0]), effective_freq(D[1, 1])
            if not (np.isfinite(pe_s) and np.isfinite(pe_t)):
                continue
            if pe_s < P_EFF_MIN or pe_t < P_EFF_MIN:
                continue

            mac = (effective_mac(D[0, 0], n_eff), effective_mac(D[1, 1], n_eff))
            if None in mac:
                continue
            key = tuple(sorted(mac))

            v_bs = bs_step(v_adj, n_eff)
            v_cal = cal_step(v_adj, key, model_r2)
            v_mcal = mcal_step(v_cal, key, model_cal)

            v_std = r2(Gs[:, [i, j]])
            recs.append((rep, n, b, truth, v_std, v_adj, v_bs, v_cal, v_mcal))
    return recs


def main():
    os.makedirs(OUT, exist_ok=True)
    d = np.load(DATA, allow_pickle=True)
    G, pos, pop = d["G"].astype(float), d["pos"], d["pop"]
    R_full = fit_residuals(G, k=K_PCS)
    log(f"data {G.shape}; full-sample residuals fitted (truth)")

    from ld_estimates.calibration import create_calibrated_estimator
    all_rows = []
    for design in DESIGNS:
        n = sum(s for _, s in design)
        n_eff = n - K_PCS
        log(f"=== design {design} -> n={n}, n_eff={n_eff} ===")

        master = {}
        model_r2 = build_calibration_models(
            n=n_eff, N_replicates=N_REP_BUILD, r2_grid_to_model=R2_GRID,
            estimators_to_calibrate={"r2": r2}, batch_estimators={"r2": r2_batch},
            n_jobs=-1)["r2"]
        master["r2"] = {n_eff: model_r2}
        cal_est = create_calibrated_estimator(r2, "r2", master, "cal")
        model_cal = build_calibration_models(
            n=n_eff, N_replicates=N_REP_BUILD_CAL, r2_grid_to_model=R2_GRID,
            estimators_to_calibrate={"r2_cal": cal_est}, n_jobs=-1)["r2_cal"]
        log(f"  curves built ({len(model_r2)} MAC pairs)")

        res = Parallel(n_jobs=-1)(
            delayed(one_rep)(rep, G, pos, pop, R_full, design, model_r2, model_cal)
            for rep in range(N_REP))
        rows = [r for sub in res for r in sub]
        all_rows.extend(rows)
        log(f"  {len(rows):,} pair-observations")

    df = pd.DataFrame(all_rows, columns=["rep", "n", "bin", "truth",
                                         "std", "Samp", "BS", "Cal", "mCal"])
    df.to_csv(os.path.join(OUT, "adjcal_raw.csv.gz"), index=False, compression="gzip")

    recs = []
    for m in ["std", "Samp", "BS", "Cal", "mCal"]:
        g = df.dropna(subset=[m, "truth"])
        for (rep, n, b), sub in g.groupby(["rep", "n", "bin"]):
            e = sub[m].to_numpy() - sub["truth"].to_numpy()
            bias = float(np.mean(e)); var = float(np.var(e))
            recs.append(dict(rep=rep, n=n, bin=b, method=m, bias=bias,
                             variance=var, rmse=float(np.sqrt(bias ** 2 + var))))
    md = pd.DataFrame(recs)
    md["binlabel"] = md["bin"].map(
        {i: f"{int(DISTANCE_BINS[i])}-{int(DISTANCE_BINS[i+1])}" for i in range(len(DISTANCE_BINS)-1)})
    md.to_csv(os.path.join(OUT, "adjcal_metrics.csv"), index=False)

    order = ["std", "Samp", "BS", "Cal", "mCal"]
    for stat in ("bias", "rmse"):
        print(f"\n===== mean {stat.upper()} vs full-sample truth =====", flush=True)
        print(md.pivot_table(index=["n", "binlabel"], columns="method", values=stat)[order]
                .round(4).to_string())
    log("DONE")


if __name__ == "__main__":
    main()
