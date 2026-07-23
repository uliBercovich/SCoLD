"""Pseudohaploid estimators by genomic distance: bias and RMSE, CEU/AFR rows, n columns.

Same layout as the paper's Figure 2 / S4 / S5 and as the Flex reviewer figures.

  pseudo_bias_by_distance.{pdf,png}
  pseudo_rmse_by_distance.{pdf,png}

Run:  python paper/make_pseudo_by_distance.py
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

METHODS = ["4r2", "4r2_HB", "Cal", "mCal"]
LABELS = {"4r2": r"$4\cdot r^2$", "4r2_HB": r"$4\cdot r^2$ (HB)", "Cal": "Cal", "mCal": "mCal"}
COLORMAP = {"4r2": "tab:orange", "4r2_HB": "tab:green", "Cal": "tab:blue", "mCal": "tab:cyan"}
POPS = ["CEU", "AFR"]
N_VALUES = [5, 10, 25]
ORDER = ["0-1000", "1000-10000", "10000-100000", "100000-1000000"]
YLABEL = {"bias": "Bias", "rmse": "Root Mean Squared Error"}


def make_figure(statistic, clip_4r2=False):
    df = pd.read_csv(os.path.join(OUT, "pseudo_metrics.csv"))
    methods = METHODS if not clip_4r2 else [m for m in METHODS if m != "4r2"]
    df = df[df["method"].isin(methods)]

    fig, axes = plt.subplots(len(POPS), len(N_VALUES),
                             figsize=(7 * len(N_VALUES), 6.5 * len(POPS)),
                             sharex=True, sharey=False)
    for i, pop in enumerate(POPS):
        d = df[df["pop"] == pop]
        ymin, ymax = d[statistic].min(), d[statistic].max()
        pad = 0.05 * (ymax - ymin)
        for j, n in enumerate(N_VALUES):
            ax = axes[i, j]
            sub = d[d["n"] == n]
            edges = sorted({int(v) for lab in ORDER for v in lab.split("-")})
            scilabs = create_sci_bin_labels(np.array(edges, dtype=float))
            if statistic == "bias":
                ax.axhline(0.0, color="red", linewidth=1.5, zorder=1)
            sns.boxplot(x="binlabel", y=statistic, hue="method", data=sub, ax=ax,
                        order=ORDER, hue_order=methods, palette=COLORMAP, showmeans=False,
                        boxprops={"edgecolor": "none"},
                        medianprops={"linewidth": 1.5, "color": "white", "alpha": 0.8},
                        legend=False, zorder=3)
            ax.set_ylim(ymin - pad, ymax + pad)
            if i == 0:
                ax.set_title(f"Sample Size n={n}", fontsize=22)
            ax.set_ylabel(YLABEL[statistic] if j == 0 else "", fontsize=20)
            ax.tick_params(axis="y", labelsize=18)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            ax.xaxis.set_major_locator(FixedLocator(np.arange(len(ORDER))))
            ax.xaxis.set_major_formatter(FixedFormatter(scilabs))
            ax.set_xlabel("Distance Bin (bp)" if i == len(POPS) - 1 else "", fontsize=20)
            ax.tick_params(axis="x", labelsize=18)
        axes[i, 0].annotate(pop, xy=(-0.24, 1.04), xycoords="axes fraction",
                            fontsize=28, fontweight="bold", ha="left", va="bottom")

    handles = [Patch(color=COLORMAP[m], label=LABELS[m]) for m in methods]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.03),
               ncol=len(methods), title="Estimator", fontsize=20, title_fontsize=22, frameon=True)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    tag = statistic + ("_no4r2" if clip_4r2 else "")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"pseudo_{tag}_by_distance.{ext}"),
                    bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[{tag}] written", flush=True)


if __name__ == "__main__":
    for stat in ("bias", "rmse"):
        make_figure(stat)
        make_figure(stat, clip_4r2=True)   # 4r2 is off-scale; also emit a readable version
