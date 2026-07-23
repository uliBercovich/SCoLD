"""Downstream figures (LD score, LD pruning F1) restricted to Cal, mCal and Flex.

Mirrors the paper's LD-score (Supp S3) and F1 (Figure 3) figures, CEU top / AFR bottom.

  flex_ldscore_by_n.{pdf,png}   <- LD-score RMSE per sample size
  flex_f1_by_threshold.{pdf,png}<- pruning F1 per r^2 threshold

Reads paper/output/ldscore_{CEU,AFR}.csv and f1flex_{CEU,AFR}.csv.
Run:  python paper/make_flex_downstream.py
"""
from __future__ import annotations
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np, pandas as pd, seaborn as sns

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "output")
FIGDIR = os.path.join(HERE, "figures")

METHODS = ["Cal", "mCal", "Flex"]
COLORMAP = {"Cal": "tab:blue", "mCal": "tab:cyan", "Flex": "tab:purple"}
POPS = ["CEU", "AFR"]
N_VALUES = [5, 10, 25]


def _legend(fig):
    handles = [Patch(color=COLORMAP[m], label=m) for m in METHODS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=len(METHODS), title="Estimator", fontsize=20, title_fontsize=22, frameon=True)


def ldscore_figure():
    """One boxplot per method, per sample size; y = LD-score RMSE. CEU top, AFR bottom."""
    fig, axes = plt.subplots(len(POPS), len(N_VALUES),
                             figsize=(7 * len(N_VALUES), 6.5 * len(POPS)), sharey=False)
    for i, pop in enumerate(POPS):
        df = pd.read_csv(os.path.join(OUT, f"ldscore_{pop}.csv"))
        df = df[df["method"].isin(METHODS)]
        ymax = df["rmse"].max() * 1.05
        for j, n in enumerate(N_VALUES):
            ax = axes[i, j]
            sns.boxplot(x="method", y="rmse", data=df[df["n"] == n], ax=ax,
                        order=METHODS, palette=COLORMAP, showmeans=False,
                        boxprops={"edgecolor": "none"},
                        medianprops={"linewidth": 1.5, "color": "white", "alpha": 0.8})
            if i == 0:
                ax.set_title(f"Sample Size n={n}", fontsize=22)
            ax.set_ylabel("LD-score RMSE" if j == 0 else "", fontsize=20)
            ax.set_xlabel("")
            ax.set_ylim(0, ymax)
            ax.tick_params(labelsize=18)
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        axes[i, 0].annotate(pop, xy=(-0.24, 1.04), xycoords="axes fraction",
                            fontsize=28, fontweight="bold", ha="left", va="bottom")
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"flex_ldscore_by_n.{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("[ldscore] written", flush=True)


def f1_figure():
    """Boxplots of F1 by r^2 threshold, hue = method; 3 sample-size columns. CEU top, AFR bottom."""
    fig, axes = plt.subplots(len(POPS), len(N_VALUES),
                             figsize=(7 * len(N_VALUES), 6.5 * len(POPS)), sharex=True, sharey=False)
    for i, pop in enumerate(POPS):
        df = pd.read_csv(os.path.join(OUT, f"f1flex_{pop}.csv"))
        df = df[df["method"].isin(METHODS)]
        df["thr"] = df["thr"].map(lambda t: f"{t:g}")
        thr_order = sorted(df["thr"].unique(), key=float)
        ymin, ymax = df["f1"].min(), df["f1"].max()
        pad = 0.05 * (ymax - ymin)
        for j, n in enumerate(N_VALUES):
            ax = axes[i, j]
            sns.boxplot(x="thr", y="f1", hue="method", data=df[df["n"] == n], ax=ax,
                        order=thr_order, hue_order=METHODS, palette=COLORMAP, showmeans=False,
                        boxprops={"edgecolor": "none"},
                        medianprops={"linewidth": 1.5, "color": "white", "alpha": 0.8}, legend=False)
            if i == 0:
                ax.set_title(f"Sample Size n={n}", fontsize=22)
            ax.set_ylabel("F1 Score" if j == 0 else "", fontsize=20)
            ax.set_xlabel(r"$r^2$ pruning threshold" if i == len(POPS) - 1 else "", fontsize=20)
            ax.set_ylim(ymin - pad, ymax + pad)
            ax.tick_params(labelsize=18)
            ax.grid(axis="y", linestyle="--", alpha=0.7)
        axes[i, 0].annotate(pop, xy=(-0.24, 1.04), xycoords="axes fraction",
                            fontsize=28, fontweight="bold", ha="left", va="bottom")
    _legend(fig)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"flex_f1_by_threshold.{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("[f1] written", flush=True)


if __name__ == "__main__":
    ldscore_figure()
    f1_figure()
