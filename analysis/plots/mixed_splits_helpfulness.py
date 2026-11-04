import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

TEXT_WIDTH = 6.30

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

C_HUMAN = "#B0205E"
C_SHP = "#5B6FB5"
C_SYNTH = "#E08A2C"
C_MIXED = "#2B2B2B"
C_BASE = "#8A8A8A"


def dollar_log_axis(ax, ticks=(100, 1000, 10000)):
    ax.set_xscale("log")
    ax.set_xticks(list(ticks))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x/1000:g}k" if x >= 1000 else f"${x:g}"))


def save(fig, png_path):
    fig.savefig(png_path)
    print(f"saved {png_path}")


SPLIT_LABELS = ["95/5", "96/4", "97/3", "98/2", "99/1"]

MODELS = [
    {
        "name": "Olmo3-7B",
        "mixed_winrate_runs": np.array([
            [79.74, 80.62],
            [76.71, 83.06],
            [79.1, 84.67],
            [78.81, 83.20],
            [75.53, 80.22],
        ]) / 100,
        "best_human_runs": np.array([76.81, 77.93]) / 100,
        "best_synthetic_runs": np.array([87.5, 87.79]) / 100,
        "base_runs": np.array([51.31, 55.27]) / 100,
    },
    {
        "name": "Llama3.1-8B",
        "mixed_winrate_runs": np.array([
            [85.69, 85.54],
            [87.94, 86.47],
            [88.38, 89.99],
            [89.94, 86.82],
            [87.21, 88.57],
        ]) / 100,
        "best_human_runs": np.array([94.24, 94.82]) / 100,
        "best_synthetic_runs": np.array([85.17, 85.89]) / 100,
        "base_runs": np.array([52.39, 58.01]) / 100,
    },
    {
        "name": "Qwen2.5-7B",
        "mixed_winrate_runs": np.array([
            [87.16, 87.26],
            [90.77, 87.06],
            [80.96, 89.36],
            [90.82, 87.84],
            [85.35, 87.60],
        ]) / 100,
        "best_human_runs": np.array([89.84, 88.92]) / 100,
        "best_synthetic_runs": np.array([85.45, 84.57]) / 100,
        "base_runs": np.array([59.86, 61.87]) / 100,
    },
]


REF_STYLE = {
    "human":     dict(color=C_HUMAN, linestyle="--", label="Best 100% human"),
    "synthetic": dict(color=C_SYNTH, linestyle="--", label="Best 100% synthetic"),
    "base":      dict(color=C_BASE,  linestyle=":",  label="Untrained base"),
}


def _mean_min_max(vals):
    arr = np.array(vals, dtype=float)
    return arr.mean(), arr.min(), arr.max()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="mixed_helpfulness_by_split.png", help="Where to save the figure")
    args = parser.parse_args()

    n = len(MODELS)
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
    save(fig, args.output)

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
