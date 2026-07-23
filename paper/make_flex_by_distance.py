"""Figure 2 / S4 / S5, restricted to Cal, mCal and Flex.

Reproduces the paper's RMSE / bias / variance boxplots by genomic distance, with CEU on the
top row and AFR on the bottom, using the metrics already cached in paper/output/. Each
statistic is written as one self-contained two-row figure.

  flex_rmse_by_distance.{pdf,png}      <- Figure 2 style
  flex_bias_by_distance.{pdf,png}      <- Figure S4 style
  flex_variance_by_distance.{pdf,png}  <- Figure S5 style

Run:  python paper/make_flex_by_distance.py
"""
from __future__ import annotations
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, FixedLocator, FixedFormatter
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paper.plotting import create_sci_bin_labels

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "output")
FIGDIR = os.path.join(HERE, "figures")

METHODS = ["Cal", "mCal", "Flex"]
COLORMAP = {"Cal": "tab:blue", "mCal": "tab:cyan", "Flex": "tab:purple"}
POPS = ["CEU", "AFR"]
N_VALUES = [5, 10, 25]
YLABEL = {"rmse": "Root Mean Squared Error", "bias": "Bias", "variance": "Variance"}


def load(pop):
    df = pd.read_csv(os.path.join(OUT, f"metrics_{pop}.csv"))
    return df[df["method"].isin(METHODS)].copy()


def make_figure(statistic):
    data = {pop: load(pop) for pop in POPS}

    fig, axes = plt.subplots(len(POPS), len(N_VALUES),
                             figsize=(7 * len(N_VALUES), 6.5 * len(POPS)),
                             sharex=True, sharey=False)

    for i, pop in enumerate(POPS):
        df = data[pop]
        # shared y-range across the three sample sizes of one population
        ymin, ymax = df[statistic].min(), df[statistic].max()
        pad = 0.05 * (ymax - ymin)
        for j, n in enumerate(N_VALUES):
            ax = axes[i, j]
            sub = df[df["n"] == n]
            bin_order = sorted(sub["binlabel"].unique(), key=lambda x: int(str(x).split("-")[0]))
            edges = sorted({int(v) for lab in bin_order for v in str(lab).split("-")})
            scilabs = create_sci_bin_labels(np.array(edges, dtype=float))

            sns.boxplot(x="binlabel", y=statistic, hue="method", data=sub, ax=ax,
                        order=bin_order, hue_order=METHODS, palette=COLORMAP,
                        showmeans=False, boxprops={"edgecolor": "none"},
                        medianprops={"linewidth": 1.5, "color": "white", "alpha": 0.8},
                        legend=False)

            ax.set_ylim(ymin - pad, ymax + pad)
            if i == 0:
                ax.set_title(f"Sample Size n={n}", fontsize=22)
            ax.set_ylabel(YLABEL[statistic] if j == 0 else "", fontsize=20)
            ax.tick_params(axis="y", labelsize=18)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            ax.xaxis.set_major_locator(FixedLocator(np.arange(len(bin_order))))
            ax.xaxis.set_major_formatter(FixedFormatter(scilabs))
            ax.set_xlabel("Distance Bin (bp)" if i == len(POPS) - 1 else "", fontsize=20)
            ax.tick_params(axis="x", labelsize=18)

        # population label on the left, like the paper's CEU/AFR
        axes[i, 0].annotate(pop, xy=(-0.24, 1.04), xycoords="axes fraction",
                            fontsize=28, fontweight="bold", ha="left", va="bottom")

    handles = [Patch(color=COLORMAP[m], label=m) for m in METHODS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=len(METHODS), title="Estimator", fontsize=20, title_fontsize=22,
               frameon=True)
    plt.tight_layout(rect=(0, 0.05, 1, 1))

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"flex_{statistic}_by_distance.{ext}"),
                    bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[{statistic}] written", flush=True)


if __name__ == "__main__":
    for stat in ("rmse", "bias", "variance"):
        make_figure(stat)
