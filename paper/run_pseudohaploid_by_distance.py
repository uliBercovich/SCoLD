"""Pseudohaploid estimators by genomic distance, on real data (Figure 2 / S4 / S5 style).

The pseudohaploid figure in the reviewer letter is the synthetic view: expected value and
RMSE against the true rho^2 at fixed allele frequencies. This is the complementary view --
the paper's own bootstrap design, on real data, binned by genomic distance.

  truth      : standard diploid r2 on the full dataset (what we are trying to recover)
  replicate  : subsample n individuals, draw ONE allele per site per individual
               (H_is ~ Bernoulli(G_is/2)), then estimate from the 0/1 matrix.

Estimators (all built/applied under the pseudohaploid model):
  4r2       naive pseudohaploid r2 rescaled by 4 (corrects the fourfold attenuation)
  4r2_HB    the same, hard-bounded to [0,1]
  Cal       one-step calibration of 4r2, curves simulated under the pseudohaploid model
  mCal      two-step calibration

Writes paper/output/pseudo_metrics.csv.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ld_estimates.estimators import r2, r2_batch
from ld_estimates.calibration import build_calibration_models, apply_calibration

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")

SAMPLE_SIZES = [5, 10, 25]
R2_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 3)
NREP_MODEL, NREP_MODEL_CAL = 5000, 2000
DISTANCE_BINS = np.array([0, 1000, 10000, 100000, 1000000])
MAX_PAIRS_PER_BIN = 400
N_REP = 100
SEED = 11

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)


def r2_x4(G):
    return 4.0 * r2(G)


def r2_x4_batch(Gb):
    return 4.0 * r2_batch(Gb)


def r2_x4_hb(G):
    """4*r2 hard-bounded to [0,1] -- the simple alternative the calibration must beat."""
    v = 4.0 * r2(G)
    if not np.isfinite(v):
        return np.nan
    return float(min(max(v, 0.0), 1.0))


def build_pseudo_calibrators():
    """Cal and mCal under the pseudohaploid model, per sample size."""
    master_base, master_cal = {}, {}
    for n in SAMPLE_SIZES:
        master_base[n] = build_calibration_models(
            n=n, N_replicates=NREP_MODEL, r2_grid_to_model=R2_GRID,
            estimators_to_calibrate={"b": r2_x4}, batch_estimators={"b": r2_x4_batch},
            pseudohaploid=True, n_jobs=-1)["b"]
        log(f"  base pseudohaploid curves built for n={n}")

    def mk_cal(n):
        model = master_base[n]
        def cal(G):
            return apply_calibration(G, r2_x4, model, calibration_type="cal", pseudohaploid=True)
        return cal
    cal = {n: mk_cal(n) for n in SAMPLE_SIZES}

    for n in SAMPLE_SIZES:
        master_cal[n] = build_calibration_models(
            n=n, N_replicates=NREP_MODEL_CAL, r2_grid_to_model=R2_GRID,
            estimators_to_calibrate={"c": cal[n]}, pseudohaploid=True, n_jobs=-1)["c"]
        log(f"  Cal curves built for n={n} (2nd step)")

    def mk_mcal(n):
        c, model = cal[n], master_cal[n]
        def cal_mcal(G):
            return apply_calibration(G, c, model, calibration_type="indep", pseudohaploid=True)
        return cal_mcal
    return cal, {n: mk_mcal(n) for n in SAMPLE_SIZES}


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


def one_rep(rep, G, pos, n, cal, mcal):
    """Subsample n individuals, pseudohaploidise, estimate; truth is the full diploid r2."""
    rng = np.random.default_rng(SEED + 1000 * n + rep)
    sel = rng.choice(G.shape[0], size=n, replace=False)
    Gs = G[sel]

    # pseudohaploid: draw one allele per individual per site, H ~ Bernoulli(G/2)
    H = (rng.random(Gs.shape) < (Gs / 2.0)).astype(float)

    ac = H.sum(axis=0)
    poly = np.where((ac > 0) & (ac < n))[0]        # polymorphic in the pseudohaploid sample
    if len(poly) < 100:
        return []

    recs = []
    for b, prs in pairs_by_bin(pos, poly, rng).items():
        for (i, j) in prs:
            truth = r2(G[:, [i, j]])               # full-sample diploid r2
            if not np.isfinite(truth):
                continue
            Hp = H[:, [i, j]]
            v4 = r2_x4(Hp)
            vhb = r2_x4_hb(Hp)
            vc = cal[n](Hp)
            vm = mcal[n](Hp)
            if not np.isfinite(v4):
                continue
            recs.append((rep, n, b, truth, v4, vhb, vc, vm))
    return recs


def main():
    os.makedirs(OUT, exist_ok=True)
    import magenpy as mgp

    datasets = {}
    gdl = mgp.GWADataLoader(mgp.tgp_eur_data_path())
    Gc, pc = gdl.genotype[22].to_numpy().astype(float), np.asarray(gdl.genotype[22].bp_pos)
    ac = Gc.sum(0); nch = 2 * Gc.shape[0]; maf = np.minimum(ac / nch, 1 - ac / nch)
    k = (maf >= 0.05) & (ac > 0) & (ac < nch)
    datasets["CEU"] = (Gc[:, k], pc[k])

    d = np.load(os.path.join(HERE, "data", "AFR_chr22.npz"))
    datasets["AFR"] = (d["G"].astype(float), d["pos"])
    for name, (g, _) in datasets.items():
        log(f"{name}: {g.shape}")

    log("building pseudohaploid calibrators ...")
    cal, mcal = build_pseudo_calibrators()

    rows = []
    for name, (Gd, posd) in datasets.items():
        for n in SAMPLE_SIZES:
            res = Parallel(n_jobs=-1)(
                delayed(one_rep)(rep, Gd, posd, n, cal, mcal) for rep in range(N_REP))
            for sub in res:
                for r in sub:
                    rows.append((name,) + r)
            log(f"{name} n={n}: {sum(len(s) for s in res):,} pair-observations")

    df = pd.DataFrame(rows, columns=["pop", "rep", "n", "bin", "truth",
                                     "4r2", "4r2_HB", "Cal", "mCal"])
    df.to_csv(os.path.join(OUT, "pseudo_raw.csv.gz"), index=False, compression="gzip")

    recs = []
    for m in ["4r2", "4r2_HB", "Cal", "mCal"]:
        g = df.dropna(subset=[m, "truth"])
        for (pop, rep, n, b), sub in g.groupby(["pop", "rep", "n", "bin"]):
            e = sub[m].to_numpy() - sub["truth"].to_numpy()
            bias = float(np.mean(e)); var = float(np.var(e))
            recs.append(dict(pop=pop, rep=rep, n=n, bin=b, method=m, bias=bias,
                             variance=var, rmse=float(np.sqrt(bias ** 2 + var))))
    md = pd.DataFrame(recs)
    md["binlabel"] = md["bin"].map(
        {i: f"{int(DISTANCE_BINS[i])}-{int(DISTANCE_BINS[i+1])}" for i in range(len(DISTANCE_BINS)-1)})
    md.to_csv(os.path.join(OUT, "pseudo_metrics.csv"), index=False)

    order = ["4r2", "4r2_HB", "Cal", "mCal"]
    for stat in ("bias", "rmse"):
        print(f"\n===== mean {stat.upper()} vs full-sample diploid r2 =====", flush=True)
        print(md.pivot_table(index=["pop", "n", "binlabel"], columns="method", values=stat)[order]
                .round(4).to_string())
    log("DONE")


if __name__ == "__main__":
    main()
