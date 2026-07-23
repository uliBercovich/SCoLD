"""
Figures for the reviewer 1 revision.

1. calibration_curve      — main-text Figure 1, redrawn as a single panel with
                            the sample sizes as separate lines (comment 1).
2. bias_curves_pseudohap  — supplementary: bias curves for diploid r², naive
                            pseudohaploid r² and rescaled pseudohaploid 4·r²
                            (comment 4).
3. metrics_pseudohap      — supplementary: bias/variance/RMSE of the calibrated
                            estimators under the pseudohaploid model (comment 4).

Run from the repository root:  python paper/make_reviewer1_figures.py
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ld_estimates.calibration import apply_calibration, build_calibration_models
from ld_estimates.estimators import r2, r2_batch
from paper.experiments import simulate_bias_curve_data, simulate_metrics_curve
from paper.plotting import plot_bias_and_variance_panels, plot_bias_curves_single_panel

FIGDIR = os.path.join(os.path.dirname(__file__), "figures")

SAMPLE_SIZES = [5, 10, 25]

# Main-text bias/variance figure: the grid used in the original analysis notebook.
FIG1_GRID = np.round(np.linspace(0, 1, 11), 3)

# Calibration models and the supplementary pseudohaploid figures use a finer grid,
# since the inverse interpolation benefits from more anchor points.
R2_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 3)

P_S, P_T = 0.5, 0.5
NREP_CURVE = 5000
NREP_MODEL = 5000
NREP_MODEL_CAL = 2000
NREP_METRIC = 3000


def r2_x4(G: np.ndarray) -> float:
    """Pseudohaploid r² rescaled by 4: sampling one allele per site halves D,
    so the naive pseudohaploid r² estimates ρ²/4 (see Supplementary Note)."""
    return 4.0 * r2(G)


def r2_x4_batch(G_batch: np.ndarray) -> np.ndarray:
    return 4.0 * r2_batch(G_batch)


def r2_x4_HB(G: np.ndarray) -> float:
    """4·r² hard-bounded to [0, 1], as done for the other estimators in the supplement."""
    return float(min(max(4.0 * r2(G), 0.0), 1.0))


def _curve(n, estimator, pseudohaploid, seed):
    return n, simulate_bias_curve_data(
        n=n, p_s=P_S, p_t=P_T, r2_grid=R2_GRID, Nrep=NREP_CURVE,
        r2_estimator=estimator, pseudohaploid=pseudohaploid, seed=seed,
    )


def figure_1():
    """Main-text Figure 1: bias panel and variance panel, one line per sample size."""
    print("[fig 1] simulating diploid bias curves ...", flush=True)
    curve_results = Parallel(n_jobs=len(SAMPLE_SIZES))(
        delayed(_curve)(n, r2, False, 1000 + n) for n in SAMPLE_SIZES
    )
    curves = {"r²": dict(curve_results)}

    print("[fig 1] simulating diploid bias/variance metrics ...", flush=True)
    # bias and variance come from the same replicate draws, so the panels agree
    metric_results = Parallel(n_jobs=len(SAMPLE_SIZES))(
        delayed(simulate_metrics_curve)(
            n=n, p_s=P_S, p_t=P_T, r2_grid=FIG1_GRID, Nrep=NREP_CURVE,
            r2_estimator=r2, pseudohaploid=False, seed=1000 + n,
        )
        for n in SAMPLE_SIZES
    )
    metrics = dict(zip(SAMPLE_SIZES, metric_results))

    for ext in ("pdf", "png"):
        plot_bias_and_variance_panels(
            metrics=metrics,
            sample_sizes=SAMPLE_SIZES,
            outpath=os.path.join(FIGDIR, f"calibration_curve.{ext}"),
        )
        plt.close("all")
    print("[fig 1] done", flush=True)
    return curves


def figure_pseudohaploid_curves(diploid_curves):
    """Supplementary: diploid vs pseudohaploid bias curves."""
    print("[fig S-pseudo] simulating pseudohaploid bias curves ...", flush=True)
    jobs = [(n, r2, True, 2000 + n) for n in SAMPLE_SIZES]
    jobs += [(n, r2_x4, True, 3000 + n) for n in SAMPLE_SIZES]
    out = Parallel(n_jobs=len(jobs))(delayed(_curve)(*j) for j in jobs)

    naive = dict(out[: len(SAMPLE_SIZES)])
    rescaled = dict(out[len(SAMPLE_SIZES):])

    curves = {
        "Diploid  r²": diploid_curves["r²"],
        "Pseudohaploid  r²": naive,
        "Pseudohaploid  4·r²": rescaled,
    }

    for ext in ("pdf", "png"):
        plot_bias_curves_single_panel(
            curves=curves,
            sample_sizes=SAMPLE_SIZES,
            # 4·r² is not bounded by 1 at small n, so the y-range must be wider
            ylim=(0, 1.7),
            equal_aspect=False,
            linestyles={
                "Diploid  r²": "-",
                "Pseudohaploid  r²": ":",
                "Pseudohaploid  4·r²": "--",
            },
            outpath=os.path.join(FIGDIR, f"bias_curves_pseudohaploid.{ext}"),
        )
        plt.close("all")
    print("[fig S-pseudo] done", flush=True)


def _build_pseudo_calibrators():
    """Cal and Cal+mCal under the pseudohaploid model, for each sample size."""
    master_base, master_cal = {}, {}

    for n in SAMPLE_SIZES:
        master_base[n] = build_calibration_models(
            n=n, N_replicates=NREP_MODEL, r2_grid_to_model=R2_GRID,
            estimators_to_calibrate={"b": r2_x4}, batch_estimators={"b": r2_x4_batch},
            pseudohaploid=True, n_jobs=-1,
            seeds=range(4000 + 100000 * n, 4000 + 100000 * n + 200000),
        )["b"]

    def mk_cal(n):
        model = master_base[n]

        def cal(G):
            return apply_calibration(G, r2_x4, model, calibration_type="cal", pseudohaploid=True)
        return cal

    cal = {n: mk_cal(n) for n in SAMPLE_SIZES}

    # the mCal step needs a calibration model of Cal itself
    for n in SAMPLE_SIZES:
        master_cal[n] = build_calibration_models(
            n=n, N_replicates=NREP_MODEL_CAL, r2_grid_to_model=R2_GRID,
            estimators_to_calibrate={"c": cal[n]},
            pseudohaploid=True, n_jobs=-1,
            seeds=range(9000 + 100000 * n, 9000 + 100000 * n + 200000),
        )["c"]

    def mk_mcal(n):
        c, model = cal[n], master_cal[n]

        def cal_mcal(G):
            return apply_calibration(G, c, model, calibration_type="indep", pseudohaploid=True)
        return cal_mcal

    return cal, {n: mk_mcal(n) for n in SAMPLE_SIZES}


def figure_pseudohaploid_calibrated_curves():
    """Bias curves under the pseudohaploid model: does calibration beat the 4x rescaling?"""
    print("[fig S-pseudo-cal] building pseudohaploid calibrators ...", flush=True)
    cal, mcal = _build_pseudo_calibrators()

    estimators = {
        "r²  (naive)":     lambda n: r2,
        "4·r²  (rescaled)": lambda n: r2_x4,
        "Cal":              lambda n: cal[n],
        "Cal + mCal":       lambda n: mcal[n],
    }

    print("[fig S-pseudo-cal] simulating expected-value curves ...", flush=True)
    jobs = [(lbl, n) for lbl in estimators for n in SAMPLE_SIZES]
    out = Parallel(n_jobs=len(jobs))(
        delayed(simulate_metrics_curve)(
            n=n, p_s=P_S, p_t=P_T, r2_grid=R2_GRID, Nrep=NREP_METRIC,
            r2_estimator=estimators[lbl](n), pseudohaploid=True, seed=7000 + n,
        )
        for lbl, n in jobs
    )
    M = {k: v for k, v in zip(jobs, out)}

    palette = sns.color_palette("colorblind", n_colors=len(estimators))
    colors = dict(zip(estimators, palette))
    markers = dict(zip(estimators, ["o", "s", "^", "D"]))

    fig, axes = plt.subplots(1, len(SAMPLE_SIZES), figsize=(5.6 * len(SAMPLE_SIZES), 5.6),
                             sharex=True, sharey=True)

    for j, n in enumerate(SAMPLE_SIZES):
        ax = axes[j]
        ax.plot([0, 1], [0, 1], color="red", linestyle="-", linewidth=1.8,
                label="Ideal (no bias)", zorder=5)
        # the value the naive statistic converges to
        ax.plot([0, 1], [0, 0.25], color="grey", linestyle=":", linewidth=1.5,
                label="ρ²/4  (naive limit)", zorder=4)

        for lbl in estimators:
            x, bias, _, _ = M[(lbl, n)]
            x = np.asarray(x)
            ax.plot(x, np.asarray(bias) + x, color=colors[lbl], marker=markers[lbl],
                    linestyle="--", linewidth=1.8, markersize=6, markevery=2,
                    markeredgecolor="white", markeredgewidth=0.6, label=lbl, zorder=3)

        ax.set_title(f"n = {n}", fontsize=16)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.75)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.tick_params(axis="both", labelsize=12)
        ax.set_xlabel("True population ρ²", fontsize=14)
        if j == 0:
            ax.set_ylabel("Expected value of the estimator", fontsize=14)
            ax.legend(fontsize=11, loc="upper left", framealpha=0.92)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"bias_curves_pseudohaploid.{ext}"),
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("[fig S-pseudo-cal] done", flush=True)


def figure_pseudohaploid_reply():
    """
    Figure for the reviewer response (comment 4).

    2 rows x 3 columns: columns are the sample sizes, the top row is the expected
    value of the estimator against the true rho^2 (as in main-text Figure 1) and
    the bottom row is the RMSE. Estimators: Samp, 4r^2, Cal, mCal.
    """
    print("[fig reply] building pseudohaploid calibrators ...", flush=True)
    cal, mcal = _build_pseudo_calibrators()

    estimators = {
        "Samp":       lambda n: r2,
        "4·r²":       lambda n: r2_x4,
        "4·r² (HB)":  lambda n: r2_x4_HB,
        "Cal":        lambda n: cal[n],
        "mCal":       lambda n: mcal[n],
    }

    print("[fig reply] simulating ...", flush=True)
    jobs = [(lbl, n) for lbl in estimators for n in SAMPLE_SIZES]
    out = Parallel(n_jobs=len(jobs))(
        delayed(simulate_metrics_curve)(
            n=n, p_s=P_S, p_t=P_T, r2_grid=R2_GRID, Nrep=8000,
            r2_estimator=estimators[lbl](n), pseudohaploid=True, seed=7000 + n,
        )
        for lbl, n in jobs
    )
    M = dict(zip(jobs, out))

    palette = sns.color_palette("colorblind", n_colors=len(estimators))
    colors = dict(zip(estimators, palette))
    markers = dict(zip(estimators, ["o", "s", "P", "^", "D"]))

    fig, axes = plt.subplots(2, len(SAMPLE_SIZES), figsize=(5.4 * len(SAMPLE_SIZES), 9.4),
                             sharex=True, sharey="row")

    for j, n in enumerate(SAMPLE_SIZES):
        ax_top, ax_bot = axes[0, j], axes[1, j]

        ax_top.plot([0, 1], [0, 1], color="red", linestyle="-", linewidth=1.8,
                    label="Ideal (no bias)", zorder=5)

        for lbl in estimators:
            x, bias, _, rmse = M[(lbl, n)]
            x = np.asarray(x)
            style = dict(color=colors[lbl], marker=markers[lbl], linestyle="--",
                         linewidth=1.8, markersize=6, markevery=2,
                         markeredgecolor="white", markeredgewidth=0.6, zorder=3)
            ax_top.plot(x, np.asarray(bias) + x, label=lbl, **style)
            ax_bot.plot(x, np.asarray(rmse), label=lbl, **style)

        ax_top.set_title(f"n = {n}", fontsize=17)
        for ax in (ax_top, ax_bot):
            ax.set_xlim(0, 1)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.tick_params(axis="both", labelsize=12)
        ax_bot.set_xlabel("True population ρ²", fontsize=15)
        ax_bot.set_ylim(bottom=0)

        if j == 0:
            ax_top.set_ylabel("Expected observed r²", fontsize=15)
            ax_bot.set_ylabel("RMSE", fontsize=15)
            ax_top.legend(fontsize=12, loc="upper left", framealpha=0.92)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"pseudohaploid_reply.{ext}"),
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("[fig reply] done", flush=True)


def figure_pseudohaploid_metrics():
    """Supplementary: does calibration fix the small-n bias of pseudohaploid 4·r²?"""
    print("[fig S-pseudo-metrics] building pseudohaploid calibration models ...", flush=True)

    master = {"r2_x4": {}, "r2_x4_cal": {}}

    for n in SAMPLE_SIZES:
        master["r2_x4"][n] = build_calibration_models(
            n=n,
            N_replicates=NREP_MODEL,
            r2_grid_to_model=R2_GRID,
            estimators_to_calibrate={"r2_x4": r2_x4},
            batch_estimators={"r2_x4": r2_x4_batch},
            pseudohaploid=True,
            n_jobs=-1,
            seeds=range(4000 + 100000 * n, 4000 + 100000 * n + 200000),
        )["r2_x4"]

    # The mCal (independence) step needs the calibration model of Cal itself.
    def make_cal(n):
        model = master["r2_x4"][n]

        def cal(G):
            return apply_calibration(G, r2_x4, model, calibration_type="cal", pseudohaploid=True)
        return cal

    cal_by_n = {n: make_cal(n) for n in SAMPLE_SIZES}

    print("[fig S-pseudo-metrics] building mCal models on top of Cal ...", flush=True)
    for n in SAMPLE_SIZES:
        master["r2_x4_cal"][n] = build_calibration_models(
            n=n,
            N_replicates=NREP_MODEL_CAL,
            r2_grid_to_model=R2_GRID,
            estimators_to_calibrate={"r2_x4_cal": cal_by_n[n]},
            pseudohaploid=True,
            n_jobs=-1,
            seeds=range(9000 + 100000 * n, 9000 + 100000 * n + 200000),
        )["r2_x4_cal"]

    def make_cal_mcal(n):
        cal = cal_by_n[n]
        model = master["r2_x4_cal"][n]

        def cal_mcal(G):
            return apply_calibration(G, cal, model, calibration_type="indep", pseudohaploid=True)
        return cal_mcal

    # Cal(4·r²) and Cal(r²) are identical: calibration inverts a monotone map, so
    # the ×4 rescaling is absorbed by the inverse. We therefore label them "Cal".
    estimators = {
        "r² (naive)": lambda n: r2,
        "4·r² (rescaled)": lambda n: r2_x4,
        "Cal": lambda n: cal_by_n[n],
        "Cal + mCal": make_cal_mcal,
    }

    print("[fig S-pseudo-metrics] simulating metric curves ...", flush=True)
    jobs = [(label, n) for label in estimators for n in SAMPLE_SIZES]
    out = Parallel(n_jobs=len(jobs))(
        delayed(simulate_metrics_curve)(
            n=n, p_s=P_S, p_t=P_T, r2_grid=R2_GRID, Nrep=NREP_METRIC,
            r2_estimator=estimators[label](n), pseudohaploid=True, seed=7000 + n,
        )
        for label, n in jobs
    )
    metrics = {label: {} for label in estimators}
    for (label, n), res in zip(jobs, out):
        metrics[label][n] = res

    metric_labels = ["Bias  (E[est] − ρ²)", "Variance", "RMSE"]
    palette = sns.color_palette("colorblind", n_colors=len(metrics))
    label_colors = dict(zip(metrics.keys(), palette))

    fig, axes = plt.subplots(
        3, len(SAMPLE_SIZES), figsize=(5.2 * len(SAMPLE_SIZES), 11.5),
        sharex=True, sharey="row",
    )

    for row_i, metric_label in enumerate(metric_labels):
        for col_i, n in enumerate(SAMPLE_SIZES):
            ax = axes[row_i, col_i]
            for label, data_by_n in metrics.items():
                x, bias, var, rmse = data_by_n[n]
                y = (bias, var, rmse)[row_i]
                ax.plot(x, y, color=label_colors[label], linewidth=2, label=label)
            if row_i == 0:
                ax.axhline(0.0, color="red", linestyle="--", linewidth=1.2)
                ax.set_title(f"n = {n}", fontsize=16)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.tick_params(axis="both", labelsize=12)
            if col_i == 0:
                ax.set_ylabel(metric_label, fontsize=14)
            if row_i == len(metric_labels) - 1:
                ax.set_xlabel("True population ρ²", fontsize=14)

    axes[0, 0].legend(fontsize=11, loc="upper right")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"metrics_pseudohaploid.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("[fig S-pseudo-metrics] done", flush=True)


if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    diploid_curves = figure_1()
    figure_pseudohaploid_curves(diploid_curves)
    figure_pseudohaploid_metrics()
    print("All figures written to", FIGDIR, flush=True)
