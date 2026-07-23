from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
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


def plot_bias_curves(
    curves: Dict[str, Dict[int, Tuple[List[float], List[float], List[float], List[float]]]],
    sample_sizes: Sequence[int],
    show_spread: bool = False,
    ylim: Optional[Tuple[float, float]] = (0, 1),
    outpath: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plots bias curves for one or more data types on a shared grid of subplots.

    Parameters
    ----------
    curves : dict
        Mapping of label → {n → (x_data, y_mean, y_low, y_high)}.
        y_low and y_high are quantile bounds from simulate_bias_curve_data.
        Example::

            {
                "Diploid":      {5: (x5, y_mean5, y_low5, y_high5), ...},
                "Pseudohaploid":{5: (x5, y_mean5, y_low5, y_high5), ...},
            }

    sample_sizes : sequence of int
        The n values that define the columns (must match the keys in each
        inner dict of *curves*).

    show_spread : bool
        If True, shade the quantile band (y_low, y_high) around each curve.
        The band is naturally asymmetric and stays within [0, 1].

    outpath : str, optional
        If given, save to this path (format inferred from extension, dpi=300).
    """
    colors = sns.color_palette("tab10", n_colors=len(curves))
    label_colors = dict(zip(curves.keys(), colors))

    ncols = len(sample_sizes)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), sharex=True, sharey=True)
    if ncols == 1:
        axes = np.array([axes])

    for j, n in enumerate(sample_sizes):
        ax = axes[j]
        ax.plot([0, 1], [0, 1], color="red", linestyle="--", label="Ideal (No Bias)", zorder=1)

        for label, data_by_n in curves.items():
            if n not in data_by_n:
                continue
            x_data, y_mean, y_low, y_high = data_by_n[n]
            color = label_colors[label]
            x = np.asarray(x_data)

            ax.plot(x, y_mean, linestyle="-", color=color, label=label, zorder=2)
            if show_spread:
                ax.fill_between(
                    x, y_low, y_high,
                    color=color,
                    alpha=0.2,
                    zorder=1,
                )

        ax.set_title(f"Sample Size n = {n}", fontsize=18)
        ax.set_xlim(0, 1)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.tick_params(axis="both", labelsize=14)
        ax.set_xlabel("True Population ρ²", fontsize=16)
        if j == 0:
            ax.set_ylabel("Expected Observed r²", fontsize=16)
        ax.legend(fontsize=12)

    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, bbox_inches="tight", dpi=300)

    return fig, axes


def plot_bias_curves_single_panel(
    curves: Dict[str, Dict[int, Tuple[List[float], List[float], List[float], List[float]]]],
    sample_sizes: Sequence[int],
    show_spread: bool = False,
    ylim: Optional[Tuple[float, float]] = (0, 1),
    linestyles: Optional[Dict[str, str]] = None,
    equal_aspect: bool = True,
    outpath: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots all bias curves on a single axes: sample size is encoded by colour and
    the data type (the keys of *curves*) by line style.

    Same input format as plot_bias_curves. With a single key this collapses to
    one curve per sample size, which is the layout used for the main-text figure.

    Set equal_aspect=False when the y-range is wider than the x-range, e.g. for
    estimators that are not bounded by 1.
    """
    colors = sns.color_palette("colorblind", n_colors=len(sample_sizes))
    n_colors = dict(zip(sample_sizes, colors))

    default_styles = ["-", "--", ":", "-."]
    if linestyles is None:
        linestyles = {label: default_styles[i % len(default_styles)] for i, label in enumerate(curves)}

    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=1.5, zorder=1)

    for label, data_by_n in curves.items():
        for n in sample_sizes:
            if n not in data_by_n:
                continue
            x_data, y_mean, y_low, y_high = data_by_n[n]
            x = np.asarray(x_data)
            color = n_colors[n]

            ax.plot(
                x, y_mean,
                linestyle=linestyles[label],
                color=color,
                linewidth=2,
                zorder=3,
            )
            if show_spread:
                ax.fill_between(x, y_low, y_high, color=color, alpha=0.15, zorder=2)

    # Legend: one entry per sample size (colour), one per data type (style),
    # plus the identity line. Styles are only listed when there is more than one.
    handles = [Line2D([0], [0], color="red", linestyle="--", label="Ideal (no bias)")]
    handles += [
        Line2D([0], [0], color=n_colors[n], linestyle="-", linewidth=2, label=f"n = {n}")
        for n in sample_sizes
    ]
    if len(curves) > 1:
        handles += [
            Line2D([0], [0], color="grey", linestyle=linestyles[label], linewidth=2, label=label)
            for label in curves
        ]

    ax.set_xlim(0, 1)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis="both", labelsize=13)
    ax.set_xlabel("True population ρ²", fontsize=15)
    ax.set_ylabel("Expected observed r²", fontsize=15)
    ax.legend(handles=handles, fontsize=12, loc="upper left", framealpha=0.9)

    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, bbox_inches="tight", dpi=300)

    return fig, ax


def plot_bias_and_variance_panels(
    metrics: Dict[int, Tuple[List[float], List[float], List[float], List[float]]],
    sample_sizes: Sequence[int],
    outpath: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Two-panel main-text figure, with one line per sample size in each panel.

    metrics maps n -> (x, bias, variance, rmse), as returned by
    simulate_metrics_curve. The left panel shows the bias curve, i.e. the
    expected observed r² (= bias + ρ²) against the true ρ², together with the
    identity line. The right panel shows the variance of r² against the true ρ².

    The identity line is solid, and each sample size is a dashed line carrying its
    own marker shape as well as its own colour, so the curves stay distinguishable
    for colour-blind readers and in greyscale print.
    """
    colors = sns.color_palette("colorblind", n_colors=len(sample_sizes))
    n_colors = dict(zip(sample_sizes, colors))

    markers = ["o", "s", "^", "D", "v"]
    n_markers = {n: markers[i % len(markers)] for i, n in enumerate(sample_sizes)}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    ax_bias, ax_var = axes

    ax_bias.plot([0, 1], [0, 1], color="red", linestyle="-", linewidth=1.8,
                 label="Ideal (no bias)", zorder=1)

    style = dict(linestyle="--", linewidth=1.8, markersize=7,
                 markeredgecolor="white", markeredgewidth=0.7, zorder=3)

    for n in sample_sizes:
        x, bias, var, _ = metrics[n]
        x = np.asarray(x)
        mean_obs = np.asarray(bias) + x

        ax_bias.plot(x, mean_obs, color=n_colors[n], marker=n_markers[n],
                     label=f"n = {n}", **style)
        ax_var.plot(x, np.asarray(var), color=n_colors[n], marker=n_markers[n],
                    label=f"n = {n}", **style)

    ax_bias.set_ylabel("Expected observed r²", fontsize=15)
    ax_bias.set_ylim(0, 1)
    ax_bias.set_aspect("equal", adjustable="box")
    ax_bias.set_title("Bias", fontsize=16)

    ax_var.set_ylabel("Variance of r²", fontsize=15)
    ax_var.set_ylim(bottom=0)
    ax_var.set_title("Variance", fontsize=16)

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.tick_params(axis="both", labelsize=13)
        ax.set_xlabel("True population ρ²", fontsize=15)
        ax.legend(fontsize=12, framealpha=0.9)

    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, bbox_inches="tight", dpi=300)

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
