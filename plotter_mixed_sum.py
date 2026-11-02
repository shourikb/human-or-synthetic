"""
Mixed-run Summarization plot -- 3 models side by side.

X-axis: discrete human/synthetic splits (e.g. 95/5, 96/4, ..., 99/1).
Y-axis: win rate for that mixed configuration, given as 2 runs per split
        (plotted as mean, with min/max shown as an error bar).

Each subplot also draws 3 horizontal dotted reference lines:
  - best 100% human win rate     (2 values, averaged)
  - best 100% synthetic win rate (2 values, averaged)
  - base (untrained) win rate    (2 values, averaged)

Edit the MODELS list below, then run:
    python mixed_summarization.py
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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

SPLIT_LABELS = ["95/5", "96/4", "97/3", "98/2", "99/1"]

# One dict per model/panel. Add, remove, or reorder entries freely --
# the figure adapts to however many are in this list.
MODELS = [
    {
        "name": "Olmo3-7B",
        # Win rate for each split above, 2 runs each (mean +/- min/max).
        "mixed_winrate_runs": np.array([
            [36.62, 41.6],   # 95/5
            [36.08, 40.53],   # 96/4
            [39.01, 40.38],   # 97/3
            [38.67, 41.5],   # 98/2
            [36.38, 41.85],   # 99/1
        ]) / 100,
        "best_human_runs": np.array([31.34, 46.0]) / 100,       # best 100% human win rate
        "best_synthetic_runs": np.array([64.6, 66.7]) / 100,   # best 100% synthetic win rate
        "base_runs": np.array([3.66, 0.78]) / 100,             # untrained baseline
    },
    {
        "name": "Llama3.1-8B",
        "mixed_winrate_runs": np.array([
            [51.86, 53.03],
            [53.76, 55.66],
            [56.88, 51.12],
            [51.51, 52.34],
            [54.83, 53.17],
        ]) / 100,
        "best_human_runs": np.array([60.89, 58.54]) / 100,
        "best_synthetic_runs": np.array([73.04, 70.41]) / 100,
        "base_runs": np.array([1.95, 4.79]) / 100,
    },
    {
        "name": "Qwen2.5-7B",
        "mixed_winrate_runs": np.array([
            [41.8, 43.46],
            [40.97, 42.77],
            [39.26, 41.41],
            [41.5, 44.87],
            [46.53, 50.83],
        ]) / 100,
        "best_human_runs": np.array([51.22, 52.34]) / 100,
        "best_synthetic_runs": np.array([66.36, 65.09]) / 100,
        "base_runs": np.array([49.85, 31.88]) / 100,
    },
]

OUTPUT_PATH = "mixed_summarization_by_split.png"

# =========================================================================

REF_STYLE = {
    "human":     dict(color=C_HUMAN, linestyle="--", label="Best 100% human"),
    "synthetic": dict(color=C_SYNTH, linestyle="--", label="Best 100% synthetic"),
    "base":      dict(color=C_BASE,  linestyle=":",  label="Untrained base"),
}


def _mean_min_max(vals):
    arr = np.array(vals, dtype=float)
    return arr.mean(), arr.min(), arr.max()


def main():
    n = len(MODELS)
    # sharey=True: all panels share one y-range (computed from all models
    # below) so the gaps between lines can be compared across models.
    fig, axes = plt.subplots(1, n, figsize=(TEXT_WIDTH, 1.95), sharey=True)
    if n == 1:
        axes = [axes]

    x = np.arange(len(SPLIT_LABELS))
    all_y = []

    for ax, model in zip(axes, MODELS):
        runs = model["mixed_winrate_runs"]
        means = runs.mean(axis=1)
        err_low = means - runs.min(axis=1)
        err_high = runs.max(axis=1) - means

        for key in ["human", "synthetic", "base"]:
            mean, _, _ = _mean_min_max(model[f"best_{key}_runs"] if key != "base" else model["base_runs"])
            style = REF_STYLE[key]
            ax.axhline(mean, color=style["color"], linestyle=style["linestyle"],
                       linewidth=0.9, label=style["label"], zorder=1)

        ax.errorbar(
            x, means, yerr=[err_low, err_high],
            color=C_MIXED, marker="o", mec="white", mew=0.5,
            elinewidth=0.7, label="Mixed human + synthetic", zorder=3,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(SPLIT_LABELS)
        ax.set_xlim(-0.3, len(x) - 0.7)
        ax.set_title(model["name"], pad=3)
        ax.grid(axis="x", visible=False)

        all_y.append(np.concatenate([
            runs.flatten(), model["best_human_runs"],
            model["best_synthetic_runs"], model["base_runs"],
        ]))

    all_y = np.concatenate(all_y)
    y_min, y_max = all_y.min(), all_y.max()
    pad = max((y_max - y_min) * 0.06, 0.02)
    axes[0].set_ylim(max(0, y_min - pad), min(1, y_max + pad))

    axes[0].set_ylabel("Win rate vs. ref.")
    axes[len(axes) // 2].set_xlabel("Budget split (human / synthetic, %)")

    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index("Mixed human + synthetic")] + [i for i, l in enumerate(labels) if l != "Mixed human + synthetic"]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, columnspacing=1.4)

    fig.tight_layout(w_pad=0.8)
    save(fig, OUTPUT_PATH)

    for model in MODELS:
        print(f"\n=== {model['name']} ===")
        runs = model["mixed_winrate_runs"]
        means = runs.mean(axis=1)
        for label, (r1, r2), m in zip(SPLIT_LABELS, runs, means):
            print(f"  {label}: run1={r1:.4f} run2={r2:.4f} mean={m:.4f}")
        for key, vals in [("best human", model["best_human_runs"]),
                           ("best synthetic", model["best_synthetic_runs"]),
                           ("base", model["base_runs"])]:
            mean, lo, hi = _mean_min_max(vals)
            print(f"  {key}: {vals} -> mean={mean:.4f}")



if __name__ == "__main__":
    main()