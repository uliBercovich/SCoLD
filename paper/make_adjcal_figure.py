"""Figure for the supplementary section: sample-size corrections of the adjusted LD matrix.

Two stacked panels (bias, RMSE) against genomic distance, in the paper's house style.
All four estimators are corrections of the SAME adjusted LD matrix, using only the matrix
and the sample size.

  adjcal_bias.{pdf,png}   adjcal_rmse.{pdf,png}   (and a combined adjcal_both.{pdf,png})

Run:  python paper/make_adjcal_figure.py
"""
from __future__ import annotations
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, FixedLocator, FixedFormatter
import numpy as np, pandas as pd, seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paper.plotting import create_sci_bin_labels

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
FIGDIR = os.path.join(HERE, "figures")

METHODS = ["Samp", "Cal", "mCal"]
COLORMAP = {"Samp": "tab:gray", "BS": "tab:orange", "Cal": "tab:blue", "mCal": "tab:cyan"}
ORDER = ["0-1000", "1000-10000", "10000-100000", "100000-1000000"]
YLABEL = {"bias": "Bias", "rmse": "Root Mean Squared Error"}


def _panel(ax, df, stat, show_x):
    sub = df[df["method"].isin(METHODS)]
    edges = sorted({int(v) for lab in ORDER for v in lab.split("-")})
    scilabs = create_sci_bin_labels(np.array(edges, dtype=float))
    if stat == "bias":
        ax.axhline(0.0, color="red", linewidth=1.5, zorder=1)
    sns.boxplot(x="binlabel", y=stat, hue="method", data=sub, ax=ax,
                order=ORDER, hue_order=METHODS, palette=COLORMAP, showmeans=False,
                boxprops={"edgecolor": "none"},
                medianprops={"linewidth": 1.5, "color": "white", "alpha": 0.8},
                legend=False, zorder=3)
    ax.set_ylabel(YLABEL[stat], fontsize=20)
    ax.tick_params(axis="y", labelsize=18)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.xaxis.set_major_locator(FixedLocator(np.arange(len(ORDER))))
    ax.xaxis.set_major_formatter(FixedFormatter(scilabs))
    ax.set_xlabel("Distance Bin (bp)" if show_x else "", fontsize=20)
    ax.tick_params(axis="x", labelsize=18)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    df = pd.read_csv(os.path.join(OUT, "adjcal_metrics.csv"))
    ns = sorted(df["n"].unique())

    # bias on top, RMSE below; one column per sample size
    fig, axes = plt.subplots(2, len(ns), figsize=(9 * len(ns), 11), sharex=True)
    if len(ns) == 1:
        axes = axes.reshape(2, 1)
    for j, n in enumerate(ns):
        sub = df[df["n"] == n]
        _panel(axes[0, j], sub, "bias", show_x=False)
        _panel(axes[1, j], sub, "rmse", show_x=True)
        axes[0, j].set_title(f"Sample Size n={n}", fontsize=22)
        if j > 0:
            axes[0, j].set_ylabel(""); axes[1, j].set_ylabel("")
    handles = [Patch(color=COLORMAP[m], label=m) for m in METHODS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.055),
               ncol=len(METHODS), title="Estimator", fontsize=18, title_fontsize=20, frameon=True)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"adjcal_both.{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("[both] written", flush=True)


if __name__ == "__main__":
    main()
