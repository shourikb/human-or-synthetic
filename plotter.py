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

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
})

# ── Cost basis ────────────────────────────────────────────────────────────
HUMAN_COST_PER_SAMPLE = 1.30  # $/sample, as given

# A true $0 point can't be shown on a log axis, so the base-model rate is
# plotted at this nominal x position instead (mirroring how reference plots
# of this kind put their "0-shot"/base point at x=10^0=1 rather than at 0).
BASE_X_POSITION = 1

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
        "Synthetic": {
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
        "Synthetic": {
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
        "Synthetic": {
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
# marker: 'x' = synthetic (AI-generated), 'o' = human -- per request, marker
# shape encodes human-vs-AI regardless of which specific source it is; color
# still distinguishes the specific dataset (HelpSteer3 vs SHP vs Synthetic).
STYLE = {
    "HelpSteer3":  dict(color="#C2185B", marker="o", label="HelpSteer3 (human)"),
    "SHP":         dict(color="#E57373", marker="o", label="SHP (human)"),
    "Synthetic":   dict(color="#F4A460", marker="x", label="Synthetic (AI-generated)"),
}

# Linestyle per run index, for the "all runs overlaid" row -- cycles if more
# runs exist than styles listed here.
RUN_LINESTYLES = ["-", "--", ":", "-."]


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
    # single scalar -- no error bar
    return float(rate), float(rate), float(rate)


def _run_value(rate, run_idx):
    """Return the value for a specific run index out of a win-rate entry
    (None / scalar / list), or None if that run doesn't exist / is missing
    for this point."""
    if rate is None:
        return None
    if isinstance(rate, (list, tuple)):
        if run_idx >= len(rate):
            return None
        v = rate[run_idx]
        return None if v is None else float(v)
    # scalar counts as "run 0" only
    return float(rate) if run_idx == 0 else None


def _style_axis(ax, model_name, title_suffix=""):
    ax.set_xscale("log")
    ax.set_xlabel("Cost ($)")
    ax.set_ylabel("Win Rate v/s Ref.")
    ax.set_title(f"{model_name}{title_suffix}")
    # Win rates in WIN_RATES are given as 0-100; display axis as 0.0-1.0
    # to match the reference plot's style without changing underlying data.
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_yticklabels([f"{v/100:.1f}" for v in np.arange(0, 101, 10)])
    ax.grid(True, which="major", linestyle="-", linewidth=0.4, alpha=0.4)
    ax.grid(True, which="minor", linestyle="-", linewidth=0.2, alpha=0.2)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_model(ax, model_name, sources, base_rate):
    """Top-row plot: mean per point, with min/max shown as an error bar."""
    base_stats = _mean_min_max(base_rate)

    for source_name, points in sources.items():
        style = STYLE[source_name]
        costs, means, err_low, err_high = [], [], [], []

        if base_stats is not None:
            base_mean, base_lo, base_hi = base_stats
            costs.append(BASE_X_POSITION)
            means.append(base_mean)
            err_low.append(base_mean - base_lo)
            err_high.append(base_hi - base_mean)

        for label, (cost, rate) in points.items():
            stats = _mean_min_max(rate)
            if stats is None:
                continue
            mean, lo, hi = stats
            costs.append(cost)
            means.append(mean)
            err_low.append(mean - lo)
            err_high.append(hi - mean)

        if not costs:
            continue

        order = np.argsort(costs)
        costs = np.array(costs)[order]
        means = np.array(means)[order]
        err_low = np.array(err_low)[order]
        err_high = np.array(err_high)[order]

        ax.errorbar(
            costs, means, yerr=[err_low, err_high],
            color=style["color"], marker=style["marker"],
            linestyle="-", label=style["label"],
            markersize=6, markeredgewidth=1.5, linewidth=1.3, alpha=0.9,
            capsize=3, elinewidth=1,
        )

    if base_stats is not None:
        ax.axhline(base_stats[0], color="gray", linestyle=":", linewidth=1, zorder=0)

    _style_axis(ax, model_name)


def plot_model_runs(ax, model_name, sources, base_rate):
    """Bottom-row plot: every individual run overlaid as its own line."""
    max_runs = 1
    for points in sources.values():
        for _, (_, rate) in points.items():
            if isinstance(rate, (list, tuple)):
                max_runs = max(max_runs, len(rate))
    if isinstance(base_rate, (list, tuple)):
        max_runs = max(max_runs, len(base_rate))

    for source_name, points in sources.items():
        style = STYLE[source_name]
        for run_idx in range(max_runs):
            costs, vals = [], []

            base_val = _run_value(base_rate, run_idx)
            if base_val is not None:
                costs.append(BASE_X_POSITION)
                vals.append(base_val)

            for label, (cost, rate) in points.items():
                v = _run_value(rate, run_idx)
                if v is None:
                    continue
                costs.append(cost)
                vals.append(v)

            if not costs:
                continue

            order = np.argsort(costs)
            costs = np.array(costs)[order]
            vals = np.array(vals)[order]
            linestyle = RUN_LINESTYLES[run_idx % len(RUN_LINESTYLES)]

            ax.plot(
                costs, vals,
                color=style["color"], marker=style["marker"],
                linestyle=linestyle, markersize=5, markeredgewidth=1.3,
                linewidth=1.1, alpha=0.85,
            )

    if base_rate is not None:
        base_stats = _mean_min_max(base_rate)
        if base_stats is not None:
            ax.axhline(base_stats[0], color="gray", linestyle=":", linewidth=1, zorder=0)

    _style_axis(ax, model_name, title_suffix=" — All Runs")


# Distinct colors per run (used only in the second, per-dataset figure below,
# where color no longer needs to encode data source since each subplot only
# contains one source already).
RUN_COLORS = ["#1B6CA8", "#D95F02", "#1B9E77", "#7570B3", "#E7298A"]


def plot_source_runs(ax, model_name, source_name, points, base_rate):
    """One subplot: every individual run for a single data source, each run
    as its own colored line."""
    style = STYLE[source_name]

    max_runs = 1
    for _, (_, rate) in points.items():
        if isinstance(rate, (list, tuple)):
            max_runs = max(max_runs, len(rate))
    if isinstance(base_rate, (list, tuple)):
        max_runs = max(max_runs, len(base_rate))

    for run_idx in range(max_runs):
        costs, vals = [], []

        base_val = _run_value(base_rate, run_idx)
        if base_val is not None:
            costs.append(BASE_X_POSITION)
            vals.append(base_val)

        for label, (cost, rate) in points.items():
            v = _run_value(rate, run_idx)
            if v is None:
                continue
            costs.append(cost)
            vals.append(v)

        if not costs:
            continue

        order = np.argsort(costs)
        costs = np.array(costs)[order]
        vals = np.array(vals)[order]
        color = RUN_COLORS[run_idx % len(RUN_COLORS)]

        ax.plot(
            costs, vals,
            color=color, marker=style["marker"], linestyle="-",
            markersize=6, markeredgewidth=1.5, linewidth=1.3, alpha=0.9,
            label=f"Run {run_idx + 1}",
        )

    if base_rate is not None:
        base_stats = _mean_min_max(base_rate)
        if base_stats is not None:
            ax.axhline(base_stats[0], color="gray", linestyle=":", linewidth=1, zorder=0)

    _style_axis(ax, model_name, title_suffix=f" — {source_name}")


def make_all_runs_by_dataset_figure():
    """Second figure: rows = models, columns = data sources (HelpSteer3, SHP,
    Synthetic), each subplot showing every individual run as its own colored
    line. Saved separately from the main winrate_vs_cost.png figure."""
    source_order = ["HelpSteer3", "SHP", "Synthetic"]
    n_models = len(WIN_RATES)
    n_sources = len(source_order)

    fig, axes = plt.subplots(n_models, n_sources, figsize=(4.3 * n_sources, 4 * n_models), squeeze=False)

    max_runs_overall = 1
    for sources in WIN_RATES.values():
        for points in sources.values():
            for _, (_, rate) in points.items():
                if isinstance(rate, (list, tuple)):
                    max_runs_overall = max(max_runs_overall, len(rate))
    for base_rate in BASE_WIN_RATE.values():
        if isinstance(base_rate, (list, tuple)):
            max_runs_overall = max(max_runs_overall, len(base_rate))

    for row, (model_name, sources) in enumerate(WIN_RATES.items()):
        base_rate = BASE_WIN_RATE.get(model_name)
        for col, source_name in enumerate(source_order):
            points = sources.get(source_name, {})
            plot_source_runs(axes[row][col], model_name, source_name, points, base_rate)

    run_handles = [
        mlines.Line2D([], [], color=RUN_COLORS[i % len(RUN_COLORS)],
                      linestyle="-", marker="o", label=f"Run {i + 1}")
        for i in range(max_runs_overall)
    ]
    fig.legend(run_handles, [h.get_label() for h in run_handles], loc="lower center",
               ncol=len(run_handles), bbox_to_anchor=(0.5, -0.02), frameon=False)

    fig.tight_layout()
    fig.savefig("winrate_all_runs_by_dataset.png", dpi=200, bbox_inches="tight")
    print("Saved winrate_all_runs_by_dataset.png")


def main():
    n_models = len(WIN_RATES)
    fig, axes = plt.subplots(1, n_models, figsize=(4.3 * n_models, 4), squeeze=False)
    axes = axes[0]

    for ax, (model_name, sources) in zip(axes, WIN_RATES.items()):
        plot_model(ax, model_name, sources, BASE_WIN_RATE.get(model_name))

    seen = {}
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            seen.setdefault(li, hi)
    labels = list(seen.keys())
    handles = list(seen.values())
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   bbox_to_anchor=(0.5, -0.08), frameon=False)

    fig.tight_layout()
    fig.savefig("winrate_vs_cost.png", dpi=200, bbox_inches="tight")
    print("Saved winrate_vs_cost.png")

    make_all_runs_by_dataset_figure()


if __name__ == "__main__":
    main()