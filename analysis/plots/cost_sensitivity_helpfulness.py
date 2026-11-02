"""
Human-cost sensitivity analysis for Helpfulness -- 3 models side by side.

Each subplot re-prices that model's human (HelpSteer3) cost curve across
a sweep of assumed per-sample human annotation costs, alongside a single
fixed synthetic curve for reference. No retraining involved: sample
counts and win rates are fixed (measured); only the human cost axis is
recomputed.

Edit the MODELS list below (one entry per model), then run:
    python sensitivity_helpfulness.py
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

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

# ============================== EDIT THIS ==============================

# One dict per model/panel. Add, remove, or reorder entries freely --
# the figure adapts to however many are in this list.
MODELS = [
    {
        "name": "Olmo3-7B",
        # Human (HelpSteer3) sample counts.
        "human_samples": np.array([76, 153, 230, 307, 769]),
        # Human win rates: 2 values per sample count (one per seed/run).
        # Averaged automatically below before plotting.
        "human_winrate_runs": np.array([
            [51.81, 55.57],   # 76 samples: run 1, run 2
            [57.86, 56.94],   # 153 samples
            [63.96, 62.30],   # 230 samples
            [68.41, 69.09],   # 307 samples
            [72.17, 72.51],   # 769 samples
        ]) / 100,
        # Synthetic points: (cost in $, win rate). Fixed, drawn once.
        "synthetic_cost": np.array([100, 200, 300, 400, 1000]),
        "synthetic_winrate_runs": np.array([
            [79.30, 80.57],
            [82.42, 87.84],
            [87.16, 85.64],
            [87.5, 87.79],
            [78.96, 79.79],
        ]) / 100,
    },
    {
        "name": "Llama3.1-8B",
        "human_samples": np.array([76, 153, 230, 307, 769]),
        "human_winrate_runs": np.array([
            [60.15, 62.45],
            [73.19, 76.29],
            [81.15, 82.12],
            [80.71, 80.62],
            [80.85, 82.18],
        ]) / 100,
        "synthetic_cost": np.array([100, 200, 300, 400, 1000]),
        "synthetic_winrate_runs": np.array([
            [83.11, 82.57],
            [86.33, 78.41],
            [85.5, 81.15],
            [80.57, 82.08],
            [85.17, 85.89],
        ]) / 100,
    },
    {
        "name": "Qwen2.5-7B",
        "human_samples": np.array([76, 153, 230, 307, 769]),
        "human_winrate_runs": np.array([
            [58.98, 60.15],
            [63.67, 62.89],
            [66.02, 66.99],
            [71.34, 71.09],
            [77.34, 75.63],
        ]) / 100,
        "synthetic_cost": np.array([100, 200, 300, 400, 1000]),
        "synthetic_winrate_runs": np.array([
            [74.32, 76.95],
            [84.81, 82.91],
            [85.45, 84.57],
            [84.13, 86.33],
            [83.64, 83.69],
        ]) / 100,
    },
]

# Price sweep for the human cost assumption ($/sample). Shared across panels.
PRICE_MIN, PRICE_MAX, PRICE_STEP = 1.0, 1.6, 0.15

OUTPUT_PATH = "human_cost_sensitivity_helpfulness.png"

# =========================================================================



# Human cost curves use one hue from light (cheapest) to dark (most
# expensive) so their order reads at a glance; synthetic is orange, matching
# the other figures.
HUMAN_COST_RAMP = LinearSegmentedColormap.from_list("human_cost", ["#E7A0BC", "#5E0C31"])


def main():
    prices = np.round(np.arange(PRICE_MIN, PRICE_MAX + 1e-9, PRICE_STEP), 2)
    colors = HUMAN_COST_RAMP(np.linspace(0, 1, len(prices)))

    n = len(MODELS)
    # sharey=True: one y-range for the row (computed from all models below).
    fig, axes = plt.subplots(1, n, figsize=(TEXT_WIDTH, 2.45), sharey=True)
    if n == 1:
        axes = [axes]

    all_y, all_x = [], []
    for ax, model in zip(axes, MODELS):
        samples = model["human_samples"]
        # Average the two per-seed runs into the value actually plotted.
        winrate = model["human_winrate_runs"].mean(axis=1)
        synthetic_winrate = model["synthetic_winrate_runs"].mean(axis=1)

        for price, color in zip(prices, colors):
            ax.plot(samples * price, winrate, color=color, marker="o",
                    ms=2.6, lw=1.0, mec="none", zorder=3)
            all_x.append(samples * price)

        ax.plot(model["synthetic_cost"], synthetic_winrate,
                color=C_SYNTH, linestyle="--", marker="X", ms=4.5, lw=1.3,
                mec="white", mew=0.5, zorder=4)
        all_x.append(model["synthetic_cost"])

        dollar_log_axis(ax, ticks=(100, 1000))
        ax.set_title(model["name"], pad=3)
        all_y.append(np.concatenate([winrate, synthetic_winrate]))

    all_x = np.concatenate(all_x)
    axes[0].set_xlim(all_x.min() / 1.25, all_x.max() * 1.25)
    all_y = np.concatenate(all_y)
    y_min, y_max = all_y.min(), all_y.max()
    pad = max((y_max - y_min) * 0.08, 0.02)
    axes[0].set_ylim(max(0, y_min - pad), min(1, y_max + pad))

    axes[0].set_ylabel("Win rate vs. ref.")
    axes[len(axes) // 2].set_xlabel("Annotation budget (log scale)")

    # One-row legend: a text lead-in, one entry per price, then synthetic.
    lead = plt.Line2D([], [], color="none")
    price_handles = [plt.Line2D([], [], color=c, lw=1.0, marker="o", ms=2.6, mec="none") for c in colors]
    synthetic_handle = plt.Line2D([], [], color=C_SYNTH, lw=1.3, linestyle="--", marker="X", ms=4.5,
                                  mec="white", mew=0.5)
    fig.legend(
        [lead] + price_handles + [synthetic_handle],
        ["Human cost / sample:"] + [f"${p:.2f}" for p in prices] + ["Synthetic"],
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(prices) + 2,
        handlelength=1.6, columnspacing=1.0, handletextpad=0.4,
    )

    fig.tight_layout(w_pad=0.8)
    save(fig, OUTPUT_PATH)

    for model in MODELS:
        print(f"\n=== {model['name']} ===")
        winrate = model["human_winrate_runs"].mean(axis=1)
        synthetic_winrate = model["synthetic_winrate_runs"].mean(axis=1)
        print("-- human --")
        print(f"samples | run1 | run2 | avg | cost @${PRICE_MIN:.2f} | cost @${PRICE_MAX:.2f}")
        for s, (r1, r2), w in zip(model["human_samples"], model["human_winrate_runs"], winrate):
            print(f"{s:>7} | {r1:.4f} | {r2:.4f} | {w:.4f} | {s*PRICE_MIN:>10.0f} | {s*PRICE_MAX:>10.0f}")
        print("-- synthetic --")
        print("cost | run1 | run2 | avg")
        for c, (r1, r2), w in zip(model["synthetic_cost"], model["synthetic_winrate_runs"], synthetic_winrate):
            print(f"{c:>4} | {r1:.4f} | {r2:.4f} | {w:.4f}")
    shift = np.log10(PRICE_MAX / PRICE_MIN)
    print(f"\nhorizontal shift across full price range (all panels): {shift:.4f} decades")



if __name__ == "__main__":
    main()