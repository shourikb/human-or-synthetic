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
        "TLDR": {
            "$100 (76 samples)": (100, [4.64, 0.63]),
            "$200 (153 samples)": (200, [4.69, 0.98]),
            "$300 (230 samples)": (300, [4.88, 0.63]),
            "$400 (307 samples)": (400, [6.15, 0.49]),
            "$1000 (769 samples)": (1000, [9.08, 1.32]),
            "10,000 samples": (10000 * HUMAN_COST_PER_SAMPLE, [31.34, 27.73]),
        },
        "Synthetic": {
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
# marker: 'x' = synthetic (AI-generated), 'o' = human -- per request, marker
# shape encodes human-vs-AI regardless of which specific source it is; color
# still distinguishes the specific dataset (HelpSteer3 vs SHP vs Synthetic).
STYLE = {
    "TLDR":  dict(color="#C2185B", marker="o", label="TLDR (human)"),
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
    source_order = ["TLDR", "Synthetic"]
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
    fig.savefig("winrate_all_runs_by_dataset_sum.png", dpi=200, bbox_inches="tight")
    print("Saved winrate_all_runs_by_dataset_sum.png")


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
    fig.savefig("winrate_vs_cost_sum.png", dpi=200, bbox_inches="tight")
    print("Saved winrate_vs_cost_sum.png")

    make_all_runs_by_dataset_figure()


if __name__ == "__main__":
    main()