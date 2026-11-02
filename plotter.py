"""
Plots win rate vs. cost for each model family, with four curves per plot:
HelpSteer3 (human), SHP (human), Anthropic HH (human), and Synthetic
(AI-generated).

Fill in the WIN_RATES dict below with real numbers before running --
placeholders are marked None and will be skipped (that point just won't
be plotted, rather than erroring), so you can run this incrementally as
results come in.

Each point's win rate can be given as:
  - None                  -> point skipped entirely
  - a single number        -> plotted with no error bar (only one run so far)
  - a list of numbers       -> plotted as mean, with min/max shown as an
                               error bar (works for 2, 3, or more runs)

This also applies to BASE_WIN_RATE (the base-model point at x=BASE_X_POSITION):
a scalar plots with no error bar, a list plots mean +/- min/max, and the
dashed reference line uses the mean of that list.

The figure has two rows per model: the top row shows mean +/- min/max (as
before), the bottom row shows every individual run overlaid as its own line
(one line per run index, same color per source, differentiated by linestyle),
so you can see the raw per-seed curves rather than only the summarized one.

Usage:
    python plot_winrate_vs_cost.py
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
import numpy as np

# ── Paper style ──────────────────────────────────────────────────────────────
# Figures are drawn at the exact size they are printed in the ACL template, so
# the font sizes below are the real printed sizes. Include the saved PNG in
# LaTeX at width=\textwidth (no extra scaling).
TEXT_WIDTH = 6.30   # inches, ACL \textwidth (figure*)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "axes.edgecolor": "#555555",
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5, "xtick.minor.width": 0.4,
    "lines.linewidth": 1.3, "lines.markersize": 4, "errorbar.capsize": 1.5,
    "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# One color per data source, shared by every figure in the paper.
C_HUMAN = "#B0205E"      # HelpSteer3 / TL;DR
C_SHP = "#5B6FB5"        # SHP
C_HH = "#2A9D8F"         # Anthropic HH
C_SYNTH = "#E08A2C"      # Synthetic
C_MIXED = "#2B2B2B"
C_BASE = "#8A8A8A"


def dollar_log_axis(ax, ticks=(100, 1000, 10000)):
    """Log x-axis labelled $100, $1k, $10k instead of 10^2, 10^3, 10^4."""
    ax.set_xscale("log")
    ax.set_xticks(list(ticks))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x/1000:g}k" if x >= 1000 else f"${x:g}"))


def save(fig, png_path):
    fig.savefig(png_path)
    print(f"saved {png_path}")

# ── Cost basis ────────────────────────────────────────────────────────────
HUMAN_COST_PER_SAMPLE = 1.30  # $/sample, as given

# ── Data ─────────────────────────────────────────────────────────────────
# Structure: MODEL -> SOURCE -> {budget_label: (cost_dollars, win_rate)}
# win_rate is None, a single number, or a list of numbers (one per run) --
# see module docstring above.
#
# For HelpSteer3, SHP, and HH, budgets are given directly in $ for the $100-$1000
# tiers (matching your synthetic tiers), plus a 10k-sample point whose cost
# is derived from HUMAN_COST_PER_SAMPLE (10,000 * $1.30 = $13,000).
#
# For Synthetic, use the actual dollar cost tiers you trained at.

WIN_RATES = {
    "Olmo3-7B": {
        "HelpSteer3": {
            "$100 (76 samples)": (100, [51.81, 55.57]),
            "$200 (153 samples)": (200, [57.86, 56.94]),
            "$300 (230 samples)": (300, [63.96, 62.3]),
            "$400 (307 samples)": (400, [68.41, 69.09]),
            "$1000 (769 samples)": (1000, [72.17, 72.51]),
            "10,000 samples": (10000 * HUMAN_COST_PER_SAMPLE, [76.81, 77.93]),
        },
        "SHP": {
            "$100": (100, [53.86, 56.35]),
            "$200": (200, [55.57, 56.64]),
            "$300": (300, [58.40, 60.45]),
            "$400": (400, [60.2, 62.89]),
            "$1000": (1000, [51.86, 51.9]),
            "10,000 samples": (10000 * HUMAN_COST_PER_SAMPLE, [26.12, 23.39]),
        },
        "HH": {
            "$100": (100, [53.96, 55.91]),
            "$200": (200, [55.71, 59.42]),
            "$300": (300, [57.91, 61.33]),
            "$400": (400, [60.2, 63.13]),
            "$1000": (1000, [59.52, 59.52]),
            "10,000 samples": (10000 * HUMAN_COST_PER_SAMPLE, [77.83, 76.22]),
        },
        "Synthetic": {
            "$20 (1,949 samples)": (20, [68.46, 68.94]),
            "$40 (3,899 samples)": (40, [72.41, 73.00]),
            "$60 (5,849 samples)": (60, [74.51, 71.48]),
            "$80 (7,799 samples)": (80, [76.61, 76.71]),
            "$100 (9,749 samples)": (100, [79.3, 80.57]),
            "$200 (19,498 samples)": (200, [82.42, 87.84]),
            "$300 (28,966 samples)": (300, [87.16, 85.64]),
            "$400 (38,621 samples)": (400, [87.5, 87.79]),
            "$1000 (97,494 samples)": (1000, [78.96, 79.79])
        },
    },
    "Llama3.1-8B": {
        "HelpSteer3": {
            "$100 (76 samples)": (100, [60.15, 62.45]),
            "$200 (153 samples)": (200, [73.19, 76.29]),
            "$300 (230 samples)": (300, [81.15, 82.12]),
            "$400 (307 samples)": (400, [80.71, 80.62]),
            "$1000 (769 samples)": (1000, [80.85, 82.18]),
            "10,000 samples": (10000 * HUMAN_COST_PER_SAMPLE, [94.24, 94.82]),
        },
        "SHP": {
            "$100": (100, [54.27, 60.35]),
            "$200": (200, [59.71, 63.96]),
            "$300": (300, [57.91, 64.36]),
            "$400": (400, [59.96, 61.82]),
            "$1000": (1000, [54.00, 52.64]),
            "10,000 samples": (10000 * HUMAN_COST_PER_SAMPLE, [45.21, 47.31]),
        },
        "HH": {
            "$100": (100, [54.1, 61.13]),
            "$200": (200, [58.25, 65.04]),
            "$300": (300, [58.64, 62.01]),
            "$400": (400, [59.96, 60.11]),
            "$1000": (1000, [60.16, 57.42]),
            "10,000 samples": (10000 * HUMAN_COST_PER_SAMPLE, [77.25, 76.81]),
        },
        "Synthetic": {
            "$20 (1,949 samples)": (20, [80.57, 78.86]),
            "$40 (3,899 samples)": (40, [85.17, 81.59]),
            "$60 (5,849 samples)": (60, [83.01, 82.42]),
            "$80 (7,799 samples)": (80, [82.91, 82.71]),
            "$100 (9,749 samples)": (100, [83.11, 82.57]),
            "$200 (19,498 samples)": (200, [86.33, 78.41]),
            "$300 (29,248 samples)": (300, [85.5, 81.15]),
            "$400 (38,997 samples)": (400, [80.57, 82.08]),
            "$1000 (97,494 samples)": (1000, [85.17, 85.89])
        },
    },
    "Qwen2.5-7B": {
        "HelpSteer3": {
            "$100 (76 samples)": (100, [58.98, 60.15]),
            "$200 (153 samples)": (200, [63.67, 62.89]),
            "$300 (230 samples)": (300, [66.02, 66.99]),
            "$400 (307 samples)": (400, [71.34, 71.09]),
            "$1000 (769 samples)": (1000, [77.34, 75.63]),
            "10,000 samples": (10000 * HUMAN_COST_PER_SAMPLE, [89.84, 88.92]),
        },
        "SHP": {
            "$100": (100, [55.08, 60.79]),
            "$200": (200, [52.39, 52.78]),
            "$300": (300, [49.95, 48.88]),
            "$400": (400, [48.82, 48.58]),
            "$1000": (1000, [44.14, 44.82]),
            "10,000 samples": (10000 * HUMAN_COST_PER_SAMPLE, [40.63, 42.38]),
        },
        "HH": {
            "$100": (100, [56.4, 57.91]),
            "$200": (200, [54.73, 54.69]),
            "$300": (300, [52, 51.81]),
            "$400": (400, [50.68, 48.54]), 
            "$1000": (1000, [53.37, 54.2]),
            "10,000 samples": (10000 * HUMAN_COST_PER_SAMPLE, [70.85, 66.6]),
        },
        "Synthetic": {
            "$20 (1,949 samples)": (20, [64.36, 64.70]),
            "$40 (3,899 samples)": (40, [74.37, 71.04]),
            "$60 (5,849 samples)": (60, [77.78, 76.32]),
            "$80 (7,799 samples)": (80, [80.37, 74.51]),
            "$100 (9,749 samples)": (100, [74.32, 76.95]),
            "$200 (19,498 samples)": (200, [84.81, 82.91]),
            "$300 (28,966 samples)": (300, [85.45, 84.57]),
            "$400 (38,621 samples)": (400, [84.13, 86.33]),
            "$1000 (97,494 samples)": (1000, [83.64, 83.69])
        },
    },

}

# Now supports the same scalar / list-of-runs format as WIN_RATES entries.
# e.g. "Olmo3-7B": [51.31, 52.4, 50.9]  -> plotted as mean with min/max error bar.
BASE_WIN_RATE = {
    "Olmo3-7B": [51.31, 55.27],
    "Llama3.1-8B": [52.93, 58.01],
    "Qwen2.5-7B": [59.86, 61.87],
}

# ── Styling ─────────────────────────────────────────────────────────────
# Color = data source (same colors in every figure). Marker shape also
# differs per source so the curves stay distinguishable in grayscale.
STYLE = {
    "HelpSteer3": dict(color=C_HUMAN, marker="o", label="HelpSteer3 (human)"),
    "SHP": dict(color=C_SHP,   marker="s", label="SHP (human)"),
    "HH": dict(color=C_HH,     marker="^", label="Anthropic HH (human)"),
    "Synthetic": dict(color=C_SYNTH, marker="X", label="Synthetic (AI-generated)"),
}

# Per-run styling for the all-runs figure. Run 1 is drawn thick and solid
# underneath; later runs are thinner and dashed on top, so both stay visible
# where the seeds agree.
RUN_COLORS = ["#2F6DB5", "#E08A2C", "#1B9E77", "#7570B3", "#E7298A"]
RUN_MARKERS = ["o", "X", "s", "^", "D"]
RUN_LINESTYLES = ["-", (0, (3, 1.5)), ":", "-."]
RUN_LINEWIDTHS = [2.0, 1.1, 1.1, 1.1, 1.1]

DISPLAY_NAME = {"TLDR": "TL;DR", "HH": "Anthropic HH"}


def _mean_min_max(rate):
    """Normalize a win-rate entry (None / scalar / list) into (mean, min, max).
    Returns None if the entry is None (point should be skipped)."""
    if rate is None:
        return None
    if isinstance(rate, (list, tuple)):
        vals = [v for v in rate if v is not None]
        if not vals:
            return None
        arr = np.array(vals, dtype=float)
        return arr.mean(), arr.min(), arr.max()
    return float(rate), float(rate), float(rate)


def _run_value(rate, run_idx):
    """Value for a specific run index out of a win-rate entry, or None."""
    if rate is None:
        return None
    if isinstance(rate, (list, tuple)):
        if run_idx >= len(rate):
            return None
        v = rate[run_idx]
        return None if v is None else float(v)
    return float(rate) if run_idx == 0 else None


def _style_axis(ax, title, yticks=np.arange(0, 101, 20)):
    """Shared axis styling. Win rates are stored in percent (0-100) and
    labelled as fractions (0.0-1.0), as before."""
    dollar_log_axis(ax)
    ax.set_xlim(12, 2.2e4)
    ax.set_ylim(0, 100)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{v/100:.1f}" for v in yticks])
    ax.set_title(title, pad=3)


def _draw_base(ax, base_rate):
    if base_rate is not None:
        base_stats = _mean_min_max(base_rate)
        if base_stats is not None:
            ax.axhline(base_stats[0], color=C_BASE, linestyle=":", linewidth=0.9, zorder=1)


def _max_runs(*entries):
    n = 1
    for rate in entries:
        if isinstance(rate, (list, tuple)):
            n = max(n, len(rate))
    return n


def plot_model(ax, model_name, sources, base_rate):
    """Mean per point, with min/max across runs shown as an error bar."""
    for source_name, points in sources.items():
        style = STYLE[source_name]
        costs, means, err_low, err_high = [], [], [], []
        for label, (cost, rate) in points.items():
            stats = _mean_min_max(rate)
            if stats is None:
                continue
            mean, lo, hi = stats
            costs.append(cost); means.append(mean)
            err_low.append(mean - lo); err_high.append(hi - mean)
        if not costs:
            continue
        order = np.argsort(costs)
        ax.errorbar(
            np.array(costs)[order], np.array(means)[order],
            yerr=[np.array(err_low)[order], np.array(err_high)[order]],
            color=style["color"], marker=style["marker"], label=style["label"],
            mec="white", mew=0.5, elinewidth=0.7, zorder=3,
        )
    _draw_base(ax, base_rate)
    _style_axis(ax, model_name)


def plot_source_runs(ax, model_name, source_name, points, base_rate):
    """One subplot: every individual run for a single data source."""
    max_runs = _max_runs(base_rate, *[rate for _, (_, rate) in points.items()])
    for run_idx in range(max_runs):
        costs, vals = [], []
        for label, (cost, rate) in points.items():
            v = _run_value(rate, run_idx)
            if v is None:
                continue
            costs.append(cost); vals.append(v)
        if not costs:
            continue
        order = np.argsort(costs)
        ax.plot(
            np.array(costs)[order], np.array(vals)[order],
            color=RUN_COLORS[run_idx % len(RUN_COLORS)],
            marker=RUN_MARKERS[run_idx % len(RUN_MARKERS)],
            linestyle=RUN_LINESTYLES[run_idx % len(RUN_LINESTYLES)],
            linewidth=RUN_LINEWIDTHS[run_idx % len(RUN_LINEWIDTHS)],
            markersize=4.5 if run_idx == 0 else 4, mec="white", mew=0.5,
            label=f"Run {run_idx + 1}", zorder=3 + run_idx,
        )
    _draw_base(ax, base_rate)
    _style_axis(ax, f"{model_name} — {DISPLAY_NAME.get(source_name, source_name)}",
                yticks=np.arange(0, 101, 25))
    ax.set_yticklabels(["0", "", "0.5", "", "1"])


def make_all_runs_by_dataset_figure():
    """Rows = models, columns = data sources, each subplot showing every
    individual run as its own line."""
    source_order = ['HelpSteer3', 'SHP', 'HH', 'Synthetic']
    n_models = len(WIN_RATES)
    n_sources = len(source_order)
    fig, axes = plt.subplots(n_models, n_sources, figsize=(TEXT_WIDTH, 4.1),
                             sharex=True, sharey=True, squeeze=False)

    for row, (model_name, sources) in enumerate(WIN_RATES.items()):
        base_rate = BASE_WIN_RATE.get(model_name)
        for col, source_name in enumerate(source_order):
            plot_source_runs(axes[row][col], model_name, source_name,
                             sources.get(source_name, {}), base_rate)
        axes[row][0].set_ylabel(f"{model_name}\nWin rate vs. ref.")
    # With 4 columns the "Model — Source" titles collide, so label columns by
    # source (top row only) and rows by model (y-axis label) instead.
    for row in range(n_models):
        for col, source_name in enumerate(source_order):
            axes[row][col].set_title(
                DISPLAY_NAME.get(source_name, source_name) if row == 0 else "", pad=3)
    for ax in axes[-1]:
        ax.set_xlabel("Cost")

    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(h))
    fig.tight_layout(h_pad=0.9, w_pad=0.6)
    save(fig, "winrate_all_runs_by_dataset.png")


def main():
    n_models = len(WIN_RATES)
    fig, axes = plt.subplots(1, n_models, figsize=(TEXT_WIDTH, 1.95), sharey=True, squeeze=False)
    axes = axes[0]

    for ax, (model_name, sources) in zip(axes, WIN_RATES.items()):
        plot_model(ax, model_name, sources, BASE_WIN_RATE.get(model_name))
    axes[0].set_ylabel("Win rate vs. ref.")
    axes[len(axes) // 2].set_xlabel("Annotation budget (log scale)")

    seen = {}
    for ax in axes:
        for hi, li in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(li, hi)
    seen["Untrained base"] = mlines.Line2D([], [], color=C_BASE, linestyle=":", linewidth=0.9)
    fig.legend(list(seen.values()), list(seen.keys()), loc="lower center",
               bbox_to_anchor=(0.5, 1.0), ncol=len(seen), handlelength=1.8, columnspacing=1.5)

    fig.tight_layout(w_pad=0.8)
    save(fig, "winrate_vs_cost.png")

    make_all_runs_by_dataset_figure()


if __name__ == "__main__":
    main()