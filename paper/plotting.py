from __future__ import annotations

from typing import Dict, Optional, Sequence
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FixedFormatter, MaxNLocator


def create_sci_bin_labels(bins: np.ndarray) -> list[str]:
    """
    Creates bin labels like: "0 - 1e3", "1e3 - 1e4", ...
    bins: array of edges (len = n_bins + 1)
    """
    bins = np.asarray(bins)
    if bins.ndim != 1 or len(bins) < 2:
        raise ValueError("bins must be a 1D array with len >= 2.")

    labels: list[str] = []
    for i in range(len(bins) - 1):
        start = float(bins[i])
        end = float(bins[i + 1])

        if start == 0:
            start_str = "0"
        else:
            start_pow = int(np.floor(np.log10(start)))
            start_str = f"1e{start_pow}"

        end_pow = int(np.floor(np.log10(end))) if end > 0 else 0
        end_str = f"1e{end_pow}" if end > 0 else "0"

        labels.append(f"{start_str} - {end_str}")
    return labels


def _default_colormap_from_order(order: Sequence[str]) -> Dict[str, tuple]:
    palette = sns.color_palette("tab10", n_colors=len(order))
    return dict(zip(order, palette))


def plot_rmse_distribution_by_bin(
    df: pd.DataFrame,
    statistic: str = "rmse",
    title: Optional[str] = None,
    subtitle: bool = True,
    colormap: Optional[Dict[str, tuple]] = None,
    show_xlabel: bool = True,
    show_legend: bool = True,
    outpath: Optional[str] = None,
):
    """
    Reproduces your binned “grid of boxplots” figure.

    Expected columns in df:
      - n (int)
      - method (str)
      - binlabel (str)  e.g. "0-1000"  (NOTE: not sci label; we convert for ticks)
      - rmse / bias / variance (depending on statistic)
    """
    if statistic not in df.columns:
        raise ValueError(f"statistic '{statistic}' not found in df columns.")

    samplesizes = sorted(df["n"].unique())
    numcols = len(samplesizes)

    hue_order = list(df["method"].unique())

    if colormap is None:
        colormap = _default_colormap_from_order(hue_order)

    fig, axes = plt.subplots(1, numcols, figsize=(7 * numcols, 7), sharex=True, sharey=False)
    if numcols == 1:
        axes = [axes]

    if title:
        fig.suptitle(title, fontsize=28, fontweight="bold", x=0.03, y=0.88)

    for j, n in enumerate(samplesizes):
        ax = axes[j]
        sub = df[df["n"] == n].copy()

        # Keep the original order of bins by numeric start
        bin_order = sorted(sub["binlabel"].unique(), key=lambda x: int(str(x).split("-")[0]))

        # Extract edges from labels like "0-1000"
        binedges = sorted({int(v) for lab in bin_order for v in str(lab).split("-")})
        scilabs = create_sci_bin_labels(np.array(binedges, dtype=float))
        ticklocs = np.arange(len(bin_order))

        sns.boxplot(
            x="binlabel",
            y=statistic,
            hue="method",
            data=sub,
            ax=ax,
            order=bin_order,
            hue_order=hue_order,
            palette=colormap,
            showmeans=False,
            boxprops={"edgecolor": "none"},
            medianprops={"linewidth": 1.5, "color": "white", "alpha": 0.8},
            legend=False,
        )

        if subtitle:
            ax.set_title(f"Sample Size n={n}", fontsize=22)
        else:
            ax.set_title(f"Sample Size n={n}", fontsize=22, color="white")

        if j == 0:
            if statistic == "rmse":
                ax.set_ylabel("Root Mean Squared Error", fontsize=20)
            elif statistic == "bias":
                ax.set_ylabel("Bias", fontsize=20)
            elif statistic == "variance":
                ax.set_ylabel("Variance", fontsize=20)
            else:
                ax.set_ylabel(statistic, fontsize=20)
        else:
            ax.set_ylabel("")

        ax.tick_params(axis="y", labelsize=18)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8, integer=False))
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        ax.set_xlabel("Distance Bin (bp)", fontsize=20)
        ax.xaxis.set_major_locator(FixedLocator(ticklocs))
        ax.xaxis.set_major_formatter(FixedFormatter(scilabs))

        if show_xlabel:
            ax.tick_params(axis="x", labelsize=18)
        else:
            ax.tick_params(axis="x", labelsize=18, colors="white")
            ax.xaxis.label.set_color("white")

    if show_legend:
        legend_patches = [Patch(color=colormap[k], label=k) for k in hue_order]
        fig.legend(
            handles=legend_patches,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.035),
            ncol=len(hue_order),
            title="Estimator",
            fontsize=20,
            title_fontsize=22,
            frameon=True,
        )
        plt.tight_layout(rect=(0, 0.1, 1, 0.96 if title else 1))
    else:
        plt.tight_layout(rect=(0, 0.05, 1, 0.96 if title else 1))

    if outpath:
        plt.savefig(outpath, format=outpath.split(".")[-1], bbox_inches="tight", dpi=300)

    return fig, axes


def plot_metric_distribution(
    df: pd.DataFrame,
    statistic: str = "rmse",
    title: Optional[str] = None,
    subtitle: bool = True,
    colormap: Optional[Dict[str, tuple]] = None,
    outpath: Optional[str] = None,
):
    """
    Non-binned metric plot: one boxplot per method, faceted by sample size n.

    Expected columns:
      - n (int)
      - method (str)
      - rmse/bias/variance (statistic)
    """
    if statistic not in df.columns:
        raise ValueError(f"statistic '{statistic}' not found in df columns.")

    samplesizes = sorted(df["n"].unique())
    numcols = len(samplesizes)

    hue_order = list(df["method"].unique())
    if colormap is None:
        colormap = _default_colormap_from_order(hue_order)

    fig, axes = plt.subplots(1, numcols, figsize=(7 * numcols, 7), sharey=False)
    if numcols == 1:
        axes = [axes]

    if title:
        fig.suptitle(title, fontsize=28, fontweight="bold", x=0.03, y=0.88)

    for j, n in enumerate(samplesizes):
        ax = axes[j]
        sub = df[df["n"] == n].copy()

        sns.boxplot(
            x="method",
            y=statistic,
            data=sub,
            ax=ax,
            order=hue_order,
            palette=colormap,
            showmeans=False,
            boxprops={"edgecolor": "none"},
            medianprops={"linewidth": 1.5, "color": "white", "alpha": 0.8},
        )

        if subtitle:
            ax.set_title(f"Sample Size n={n}", fontsize=22)
        else:
            ax.set_title(f"Sample Size n={n}", fontsize=22, color="white")

        if j == 0:
            if statistic == "rmse":
                ax.set_ylabel("Root Mean Squared Error", fontsize=20)
            elif statistic == "bias":
                ax.set_ylabel("Bias", fontsize=20)
            elif statistic == "variance":
                ax.set_ylabel("Variance", fontsize=20)
            else:
                ax.set_ylabel(statistic, fontsize=20)
        else:
            ax.set_ylabel("")

        ax.tick_params(axis="y", labelsize=18)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))

        ax.set_xlabel("")
        ax.tick_params(axis="x", bottom=False)
        ax.set_xticklabels([])

    plt.tight_layout(rect=(0, 0.05, 1, 0.95 if title else 1))

    if outpath:
        plt.savefig(outpath, format=outpath.split(".")[-1], bbox_inches="tight", dpi=300)

    return fig, axes


def create_estimator_legend(
    colormap: Dict[str, tuple],
    outpath: Optional[str] = None,
    title: str = "Estimator",
    ncol: Optional[int] = None,
):
    """
    Standalone horizontal legend (your notebook helper).
    """
    if ncol is None:
        ncol = len(colormap)

    legend_patches = [Patch(color=color, label=label) for label, color in colormap.items()]

    fig = plt.figure(figsize=(12, 1.5))
    leg = fig.legend(
        handles=legend_patches,
        loc="center",
        ncol=ncol,
        title=title,
        fontsize=22,
        title_fontsize=24,
        frameon=True,
        edgecolor="lightgrey",
        labelspacing=1.0,
        columnspacing=2.0,
    )
    plt.gca().set_axis_off()

    if outpath:
        fig.savefig(outpath, bbox_inches="tight", pad_inches=0.1, dpi=300)
        plt.close(fig)

    return fig, leg
