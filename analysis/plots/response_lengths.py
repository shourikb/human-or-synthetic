import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

TASK_CONDITIONS = {
    "helpfulness": [
        ("base", "Base"),
        ("shp", "SHP"),
        ("helpsteer", "HelpSteer"),
        ("synthetic", "Synthetic"),
        ("mixed", "Mixed"),
    ],
    "summarization": [
        ("base", "Base"),
        ("tldr", "TL;DR"),
        ("synthetic", "Synthetic"),
        ("mixed", "Mixed"),
    ],
}
TASK_LABELS = {"helpfulness": "Helpfulness", "summarization": "Summarization"}

MODEL_TOKENIZERS = [
    ("olmo", "allenai/Olmo-3-1025-7B"),
    ("llama", "unsloth/Meta-Llama-3.1-8B"),
    ("qwen", "unsloth/Qwen2.5-7B"),
]

COLORS = {
    "base": "#9e9e9e",
    "shp": "#2c3e7a",
    "helpsteer": "#5a78c2",
    "tldr": "#3b528b",
    "synthetic": "#21918c",
    "mixed": "#5ec962",
}


_TOKENIZER_CACHE = {}


def tokenizer_for(path: Path):
    name = str(path).lower()
    for substring, tok_id in MODEL_TOKENIZERS:
        if substring in name:
            if tok_id not in _TOKENIZER_CACHE:
                from transformers import AutoTokenizer
                print(f"  loading tokenizer {tok_id}")
                _TOKENIZER_CACHE[tok_id] = AutoTokenizer.from_pretrained(tok_id)
            return _TOKENIZER_CACHE[tok_id]
    return None


def token_lengths(texts: list[str], tokenizer) -> list[int]:
    ids = tokenizer(texts, add_special_tokens=False)["input_ids"]
    return [len(x) for x in ids]


def lengths_in_folder(folder: Path, field: str, root: Path) -> list[int]:
    lengths = []
    files = sorted(folder.rglob("*.jsonl"))
    if not files:
        print(f"  [warn] no .jsonl files under {folder}")
    for f in files:
        tokenizer = tokenizer_for(f.relative_to(root))
        if tokenizer is None:
            print(f"  [skip] {f.relative_to(folder.parent)}: no model in MODEL_TOKENIZERS matches this path")
            continue
        texts, n_bad = [], 0
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    text = json.loads(line)[field]
                    if not isinstance(text, str):
                        raise TypeError
                    texts.append(text)
                except (json.JSONDecodeError, KeyError, TypeError):
                    n_bad += 1
        if texts:
            lengths.extend(token_lengths(texts, tokenizer))
        print(f"  read {f.relative_to(folder.parent)}" + (f" ({n_bad} bad lines skipped)" if n_bad else ""))
    return lengths


def compute_stats(root: Path, field: str) -> dict:
    stats = {t: {} for t in TASK_CONDITIONS}
    for task, conds in TASK_CONDITIONS.items():
        for key, _ in conds:
            folder = root / f"{key}-{task}"
            if not folder.is_dir():
                print(f"[skip] {folder} not found")
                continue
            print(f"[{key}-{task}]")
            lens = lengths_in_folder(folder, field, root)
            if lens:
                stats[task][key] = {
                    "avg": statistics.mean(lens),
                    "median": statistics.median(lens),
                    "min": min(lens),
                    "max": max(lens),
                    "n": len(lens),
                }
    return stats


def save_csv(stats: dict, path: Path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "condition", "n_responses", "avg_tokens", "median_tokens", "min_tokens", "max_tokens"])
        for task, conds in TASK_CONDITIONS.items():
            for key, _ in conds:
                s = stats[task].get(key)
                if s:
                    w.writerow([task, key, s["n"], round(s["avg"], 2), s["median"], s["min"], s["max"]])
    print(f"wrote {path}")


def plot(stats: dict, out_path: Path):
    plt.rcParams.update({
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.4,
        "grid.linewidth": 0.6,
    })

    tasks = [t for t in TASK_CONDITIONS if stats[t]]
    counts = [sum(1 for k, _ in TASK_CONDITIONS[t] if k in stats[t]) for t in tasks]
    fig, axes = plt.subplots(1, len(tasks), figsize=(1.6 * sum(counts) + 1, 3.8),
                             sharey=True, gridspec_kw={"width_ratios": counts})
    axes = np.atleast_1d(axes)

    width = 0.38
    for ax, task in zip(axes, tasks):
        conds = [(k, lbl) for k, lbl in TASK_CONDITIONS[task] if k in stats[task]]
        x = np.arange(len(conds))
        colors = [COLORS.get(k, "#777777") for k, _ in conds]
        means = [stats[task][k]["avg"] for k, _ in conds]
        medians = [stats[task][k]["median"] for k, _ in conds]

        b1 = ax.bar(x - width / 2, means, width, color=colors, edgecolor=colors, linewidth=1.2, label="Mean")
        b2 = ax.bar(x + width / 2, medians, width, color="white", edgecolor=colors, linewidth=1.2,
                    hatch="///", label="Median")
        ax.bar_label(b1, labels=[f"{v:.0f}" for v in means], fontsize=7.5, padding=2)
        ax.bar_label(b2, labels=[f"{v:.0f}" for v in medians], fontsize=7.5, padding=2)

        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in conds])
        ax.set_title(TASK_LABELS[task])
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.12)

    legend_handles = [Patch(facecolor="#555555", edgecolor="#555555", label="Mean"),
                      Patch(facecolor="white", edgecolor="#555555", hatch="///", label="Median")]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08), frameon=False)

    axes[0].set_ylabel("Response length (tokens)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--responses_dir", type=Path, required=True, help="Folder holding the <key>-<task>/ response folders")
    p.add_argument("--out", type=Path, default=Path("response_lengths.png"))
    p.add_argument("--field", default="response", help="JSON key holding the model response")
    args = p.parse_args()

    stats = compute_stats(args.responses_dir, args.field)
    save_csv(stats, args.out.with_suffix(".csv"))
    plot(stats, args.out)


if __name__ == "__main__":
    main()
