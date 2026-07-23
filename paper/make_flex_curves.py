"""
Bias curves for the calibrated estimators: Cal, mCal and Flex.

Same construction as the pseudohaploid bias-curve figure (make_reviewer1_figures.py):
one panel, sample size encoded by colour, estimator by line style.

  bias_curves_cal_mcal_flex   — E[estimate] against the true rho^2
  bias_residual_cal_mcal_flex — the bias itself, E[estimate] - rho^2

The second panel is what the reviewer is asking about: mCal is anchored at rho^2 = 0
and rho^2 = 1 but is affine in between, so it over-corrects at intermediate rho^2;
Cal keeps an upward bias near independence; Flex fits the curvature and stays flat.

Run from the repository root:  python paper/make_flex_curves.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ld_estimates.calibration import (
    build_calibration_models, create_calibrated_estimator,
    build_flex_models, create_flex_estimator,
)
from ld_estimates.estimators import r2, r2_batch
from paper.experiments import simulate_bias_curve_data, simulate_metrics_curve
from paper.plotting import plot_bias_curves_single_panel

FIGDIR = os.path.join(os.path.dirname(__file__), "figures")

SAMPLE_SIZES = [5, 10, 25]
R2_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 3)
P_S, P_T = 0.5, 0.5

NREP_CURVE = 5000      # replicates per grid point for the curves
NREP_METRIC = 5000     # replicates per grid point for bias/variance/RMSE
NREP_MODEL = 5000      # replicates for the Cal / mCal calibration models
NREP_FLEX = 8000       # replicates for the Flex polynomial fit (2000 left a visible
                       # kink at n=5; the moment-matching fit is noisy at small n)
FLEX_DEGREE = 2        # degree 1 would reproduce mCal

METHODS = ["Cal", "mCal", "Flex"]
LINESTYLES = {"Cal": "-", "mCal": "--", "Flex": ":"}


def build_estimators():
    """Cal, mCal and Flex for each sample size (diploid)."""
    master, cal_by_n, mcal_by_n, flex_master = {}, {}, {}, {}

    for n in SAMPLE_SIZES:
        bm = build_calibration_models(
            n=n, N_replicates=NREP_MODEL, r2_grid_to_model=R2_GRID,
            estimators_to_calibrate={"r2": r2}, batch_estimators={"r2": r2_batch},
            n_jobs=-1,
        )
        master.setdefault("r2", {})[n] = bm["r2"]
        cal = create_calibrated_estimator(r2, "r2", master, "cal")

        cm = build_calibration_models(
            n=n, N_replicates=NREP_MODEL, r2_grid_to_model=R2_GRID,
            estimators_to_calibrate={"r2_cal": cal}, n_jobs=-1,
        )
        master.setdefault("r2_cal", {})[n] = cm["r2_cal"]
        mcal = create_calibrated_estimator(cal, "r2_cal", master, "indep")

        flex_master[n] = build_flex_models(
            n=n, cal_estimator=cal, r2_grid=R2_GRID,
            N_replicates=NREP_FLEX, degree=FLEX_DEGREE, n_jobs=-1,
        )

        cal_by_n[n], mcal_by_n[n] = cal, mcal
        print(f"[build] n={n}: Flex fitted for {len(flex_master[n])} MAC pairs", flush=True)

    flex = create_flex_estimator(lambda G: cal_by_n[G.shape[0]](G), flex_master)
    return cal_by_n, mcal_by_n, flex


def _curve(n, estimator, seed):
    return n, simulate_bias_curve_data(
        n=n, p_s=P_S, p_t=P_T, r2_grid=R2_GRID, Nrep=NREP_CURVE,
        r2_estimator=estimator, seed=seed,
    )


def plot_bias_residuals(curves, outpath):
    """Companion panel: the bias E[estimate] - rho^2, where the differences are visible."""
    colors = sns.color_palette("colorblind", n_colors=len(SAMPLE_SIZES))
    n_colors = dict(zip(SAMPLE_SIZES, colors))

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.axhline(0.0, color="red", linewidth=1.5, zorder=1)

    for label, data_by_n in curves.items():
        for n in SAMPLE_SIZES:
            x, y_mean, _, _ = data_by_n[n]
            x = np.asarray(x)
            ax.plot(
                x, np.asarray(y_mean) - x,
                linestyle=LINESTYLES[label], color=n_colors[n], linewidth=2, zorder=3,
            )

    handles = [Line2D([0], [0], color="red", linewidth=1.5, label="Unbiased")]
    handles += [Line2D([0], [0], color=n_colors[n], linewidth=2, label=f"n = {n}")
                for n in SAMPLE_SIZES]
    handles += [Line2D([0], [0], color="grey", linestyle=LINESTYLES[m], linewidth=2, label=m)
                for m in METHODS]

    ax.set_xlim(0, 1)
    ax.set_xlabel(r"True population $\rho^2$")
    ax.set_ylabel(r"Bias:  $\mathbb{E}[\widehat{r^2}] - \rho^2$")
    ax.legend(handles=handles, frameon=True, loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    return fig, ax


def _metrics(n, estimator, seed):
    return n, simulate_metrics_curve(
        n=n, p_s=P_S, p_t=P_T, r2_grid=R2_GRID, Nrep=NREP_METRIC,
        r2_estimator=estimator, seed=seed,
    )


def plot_metrics_panels(metrics, outpath):
    """Bias, variance and RMSE against the true rho^2 — accuracy and precision side by side.

    This is the figure the reviewer is asking for: the bias panel alone flatters Flex,
    and only the variance panel shows what that accuracy costs.
    """
    colors = sns.color_palette("colorblind", n_colors=len(SAMPLE_SIZES))
    n_colors = dict(zip(SAMPLE_SIZES, colors))

    titles = ["Bias", "Variance", "RMSE"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for k, (ax, title) in enumerate(zip(axes, titles)):
        if k == 0:
            ax.axhline(0.0, color="red", linewidth=1.5, zorder=1)
        for label, data_by_n in metrics.items():
            for n in SAMPLE_SIZES:
                x, bias, var, rmse = data_by_n[n]
                y = [bias, var, rmse][k]
                ax.plot(np.asarray(x), np.asarray(y),
                        linestyle=LINESTYLES[label], color=n_colors[n],
                        linewidth=2, zorder=3)
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_xlabel(r"True population $\rho^2$")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel(r"$\mathbb{E}[\widehat{r^2}] - \rho^2$")
    axes[1].set_ylabel(r"Var$(\widehat{r^2})$")
    axes[2].set_ylabel(r"RMSE")

    handles = [Line2D([0], [0], color=n_colors[n], linewidth=2, label=f"n = {n}")
               for n in SAMPLE_SIZES]
    handles += [Line2D([0], [0], color="grey", linestyle=LINESTYLES[m], linewidth=2, label=m)
                for m in METHODS]
    axes[2].legend(handles=handles, frameon=True, loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    return fig, axes


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    cal_by_n, mcal_by_n, flex = build_estimators()

    est_by_method = {
        "Cal": lambda G: cal_by_n[G.shape[0]](G),
        "mCal": lambda G: mcal_by_n[G.shape[0]](G),
        "Flex": flex,
    }

    print("[curves] simulating ...", flush=True)
    jobs = [(m, n) for m in METHODS for n in SAMPLE_SIZES]
    out = Parallel(n_jobs=len(jobs))(
        delayed(_curve)(n, est_by_method[m], 7000 + 100 * i) for i, (m, n) in enumerate(jobs)
    )

    curves = {m: {} for m in METHODS}
    for (m, _), (n, data) in zip(jobs, out):
        curves[m][n] = data

    for ext in ("pdf", "png"):
        plot_bias_curves_single_panel(
            curves=curves,
            sample_sizes=SAMPLE_SIZES,
            ylim=(0, 1),
            linestyles=LINESTYLES,
            outpath=os.path.join(FIGDIR, f"bias_curves_cal_mcal_flex.{ext}"),
        )
        plt.close("all")
        plot_bias_residuals(curves, os.path.join(FIGDIR, f"bias_residual_cal_mcal_flex.{ext}"))
        plt.close("all")
    print("[curves] done", flush=True)

    # --- bias / variance / RMSE against rho^2: accuracy AND precision ---
    print("[metrics] simulating ...", flush=True)
    out_m = Parallel(n_jobs=len(jobs))(
        delayed(_metrics)(n, est_by_method[m], 8000 + 100 * i) for i, (m, n) in enumerate(jobs)
    )
    metrics = {m: {} for m in METHODS}
    for (m, _), (n, data) in zip(jobs, out_m):
        metrics[m][n] = data

    for ext in ("pdf", "png"):
        plot_metrics_panels(metrics, os.path.join(FIGDIR, f"metrics_cal_mcal_flex.{ext}"))
        plt.close("all")

    print("\n=== variance and RMSE at selected rho^2 (what the bias panel hides) ===")
    for n in SAMPLE_SIZES:
        print(f"\n n = {n}")
        print(f"{'rho2':>6} " + "".join(f"{m+'_var':>11}{m+'_rmse':>11}" for m in METHODS))
        x = np.asarray(metrics["Cal"][n][0])
        for i, rho2 in enumerate(x):
            if not np.isclose(rho2 % 0.25, 0, atol=1e-6):
                continue
            row = f"{rho2:>6.2f} "
            for m in METHODS:
                row += f"{metrics[m][n][2][i]:>11.4f}{metrics[m][n][3][i]:>11.4f}"
            print(row)

    # print the curve values so the claims can be checked against numbers
    print("\n=== bias E[est] - rho^2 at selected rho^2 ===")
    for n in SAMPLE_SIZES:
        print(f"\n n = {n}")
        print(f"{'rho2':>6} " + "".join(f"{m:>9}" for m in METHODS))
        x = np.asarray(curves["Cal"][n][0])
        for i, rho2 in enumerate(x):
            if not np.isclose(rho2 % 0.1, 0, atol=1e-6) and not np.isclose(rho2 % 0.1, 0.1, atol=1e-6):
                continue
            row = f"{rho2:>6.2f} "
            for m in METHODS:
                row += f"{curves[m][n][1][i] - rho2:>9.4f}"
            print(row)


if __name__ == "__main__":
    main()
