"""
Plots win rate vs. cost for each model family, with three curves per plot:
HelpSteer3 (human), SHP (human), and Synthetic (AI-generated).

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
# For HelpSteer3 and SHP, budgets are given directly in $ for the $100-$1000
# tiers (matching your synthetic tiers), plus a 10k-sample point whose cost
# is derived from HUMAN_COST_PER_SAMPLE (10,000 * $1.30 = $13,000).
#
# For Synthetic, use the actual dollar cost tiers you trained at.

WIN_RATES = {
    "Olmo3-7B": {
        "TLDR": {
            "$100 (76 samples)": (100, [4.64, 0.63]),
            "$200 (153 samples)": (200, [4.69, 0.98]),
            "$300 (230 samples)": (300, [4.88, 0.63]),
            "$400 (307 samples)": (400, [6.15, 0.49]),
            "$1000 (769 samples)": (1000, [9.08, 1.32]),
            "10,000 samples": (10000 * HUMAN_COST_PER_SAMPLE, [31.34, 27.73]),
        },
        "Synthetic": {
            "$20 (1,949 samples)": (20, [37.6, 46]),
            "$40 (3,899 samples)": (40, [43.02, 51.42]),
            "$60 (5,849 samples)": (60, [49.46, 58.35]),
            "$80 (7,799 samples)": (80, [55.32, 59.47]),
            "$100 (9,749 samples)": (100, [58.11, 61.72]),
            "$200 (19,498 samples)": (200, [58.40, 67.33]),
            "$300 (28,966 samples)": (300, [64.60, 65.62]),
            "$400 (38,621 samples)": (400, [64.60, 66.7]),
            "$1000 (97,494 samples)": (1000, [61.57, 58.15])
        },
    },
    "Llama3.1-8B": {
        "TLDR": {
            "$100 (76 samples)": (100, [4.79, 5.13]),
            "$200 (153 samples)": (200, [6.30, 6.64]),
            "$300 (230 samples)": (300, [6.79, 7.67]),
            "$400 (307 samples)": (400, [5.66, 7.57]),
            "$1000 (769 samples)": (1000, [10.94, 10.16]),
            "10,000 samples": (10000 * HUMAN_COST_PER_SAMPLE, [60.89, 58.54]),
        },
        "Synthetic": {
            "$20 (1,949 samples)": (20, [46.48, 58.3]),
            "$40 (3,899 samples)": (40, [56.05, 66.6]),
            "$60 (5,849 samples)": (60, [64.40, 70.75]),
            "$80 (7,799 samples)": (80, [69.87, 73.39]),
            "$100 (9,749 samples)": (100, [73.29, 70.17]),
            "$200 (19,498 samples)": (200, [70.36, 70.61]),
            "$300 (29,248 samples)": (300, [73.04, 70.41]),
            "$400 (38,997 samples)": (400, [67.78, 70.94]),
            "$1000 (97,494 samples)": (1000, [67.04, 67.72])
        },
    },
    "Qwen2.5-7B": {
        "TLDR": {
            "$100 (76 samples)": (100, [38.82, 26.76]),
            "$200 (153 samples)": (200, [23.63, 21.44]),
            "$300 (230 samples)": (300, [23.93, 16.36]),
            "$400 (307 samples)": (400, [19.97, 22.85]),
            "$1000 (769 samples)": (1000, [28.32, 28.13]),
            "10,000 samples": (10000 * HUMAN_COST_PER_SAMPLE, [51.22, 52.34]),
        },
        "Synthetic": {
            "$20 (1,949 samples)": (20, [58.54, 58.07]),
            "$40 (3,899 samples)": (40, [58.30, 56.69]),
            "$60 (5,849 samples)": (60, [63.62, 62.84]),
            "$80 (7,799 samples)": (80, [63.82, 62.26]),
            "$100 (9,749 samples)": (100, [66.06, 65.33]),
            "$200 (19,498 samples)": (200, [66.36, 65.09]),
            "$300 (28,966 samples)": (300, [67.33, 62.11]),
            "$400 (38,621 samples)": (400, [66.02, 62.55]),
            "$1000 (97,494 samples)": (1000, [63.48, 63.82])
        },
    },

}

# Now supports the same scalar / list-of-runs format as WIN_RATES entries.
# e.g. "Olmo3-7B": [51.31, 52.4, 50.9]  -> plotted as mean with min/max error bar.
BASE_WIN_RATE = {
    "Olmo3-7B": [3.66, 0.78],
    "Llama3.1-8B": [1.95, 4.79],
    "Qwen2.5-7B": [49.85, 31.88],
}

# ── Styling ─────────────────────────────────────────────────────────────
# Color = data source (same colors in every figure). Marker shape also
# differs per source so the curves stay distinguishable in grayscale.
STYLE = {
    "TLDR": dict(color=C_HUMAN, marker="o", label="TL;DR (human)"),
    "Synthetic": dict(color=C_SYNTH, marker="X", label="Synthetic (AI-generated)"),
}

# Per-run styling for the all-runs figure. Run 1 is drawn thick and solid
# underneath; later runs are thinner and dashed on top, so both stay visible
# where the seeds agree.
RUN_COLORS = ["#2F6DB5", "#E08A2C", "#1B9E77", "#7570B3", "#E7298A"]
RUN_MARKERS = ["o", "X", "s", "^", "D"]
RUN_LINESTYLES = ["-", (0, (3, 1.5)), ":", "-."]
RUN_LINEWIDTHS = [2.0, 1.1, 1.1, 1.1, 1.1]

DISPLAY_NAME = {"TLDR": "TL;DR"}


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
    source_order = ['TLDR', 'Synthetic']
    n_models = len(WIN_RATES)
    n_sources = len(source_order)
    fig, axes = plt.subplots(n_models, n_sources, figsize=(TEXT_WIDTH * 2 / 3, 4.1),
                             sharex=True, sharey=True, squeeze=False)

    for row, (model_name, sources) in enumerate(WIN_RATES.items()):
        base_rate = BASE_WIN_RATE.get(model_name)
        for col, source_name in enumerate(source_order):
            plot_source_runs(axes[row][col], model_name, source_name,
                             sources.get(source_name, {}), base_rate)
        axes[row][0].set_ylabel("Win rate vs. ref.")
    for ax in axes[-1]:
        ax.set_xlabel("Cost")

    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(h))
    fig.tight_layout(h_pad=0.9, w_pad=0.6)
    save(fig, "winrate_all_runs_by_dataset_sum.png")


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
    save(fig, "winrate_vs_cost_sum.png")

    make_all_runs_by_dataset_figure()


if __name__ == "__main__":
    main()