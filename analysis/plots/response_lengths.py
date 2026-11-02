"""
Plot mean and median response length (in tokens) for each training condition, per task.

Each response is tokenized with the tokenizer of the model that generated it. The model
is detected from the file path (e.g. "responses_models_llama3-sft-..." -> Llama3.1-8B);
see MODEL_TOKENIZERS below.

Helpfulness has two human-data sources (SHP, HelpSteer); summarization has one (TL;DR),
so each task has its own list of conditions (see TASK_CONDITIONS below).

The folders are expected to sit in the same directory as this script
(folder = "<key>-<task>"):

    <script directory>/
        base-helpfulness/         seed-42-123/*.jsonl  seed-55-555/*.jsonl
        shp-helpfulness/          ...
        helpsteer-helpfulness/    ...
        synthetic-helpfulness/    ...
        mixed-helpfulness/        ...
        base-summarization/       ...
        tldr-summarization/       ...
        synthetic-summarization/  ...
        mixed-summarization/      ...

If your folders are named differently, just edit the keys in TASK_CONDITIONS.
Every .jsonl file found (recursively) inside a folder is read, and the stats are
pooled over all responses in that folder (all seeds, all files, all models).

Usage:
    python plot_response_lengths.py
    python plot_response_lengths.py --demo          # example values, no data needed
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# (folder key, display label) per task, in plotting order
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

# substring found in the file path (lowercased) -> HF tokenizer id or local path.
# Checked in order; the first match wins. Point these at the exact checkpoints you
# trained from (a local path to your model directory works too).
MODEL_TOKENIZERS = [
    ("olmo", "allenai/Olmo-3-1025-7B"),
    ("llama", "unsloth/Meta-Llama-3.1-8B"),
    ("qwen", "unsloth/Qwen2.5-7B"),
]

# human-data sources share a blue family so they read as one "human" group
COLORS = {
    "base": "#9e9e9e",
    "shp": "#2c3e7a",
    "helpsteer": "#5a78c2",
    "tldr": "#3b528b",
    "synthetic": "#21918c",
    "mixed": "#5ec962",
}

ROOT = Path(__file__).resolve().parent

_TOKENIZER_CACHE = {}


def tokenizer_for(path: Path):
    """Return the tokenizer for the model that produced this file, or None if unknown."""
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
    """Number of tokens per response (no BOS/EOS or other special tokens added)."""
    ids = tokenizer(texts, add_special_tokens=False)["input_ids"]
    return [len(x) for x in ids]


def lengths_in_folder(folder: Path, field: str) -> list[int]:
    lengths = []
    files = sorted(folder.rglob("*.jsonl"))
    if not files:
        print(f"  [warn] no .jsonl files under {folder}")
    for f in files:
        tokenizer = tokenizer_for(f.relative_to(ROOT))
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
    """stats[task][key] = {"avg", "median", "min", "max", "n"}; missing folders are skipped."""
    stats = {t: {} for t in TASK_CONDITIONS}
    for task, conds in TASK_CONDITIONS.items():
        for key, _ in conds:
            folder = root / f"{key}-{task}"
            if not folder.is_dir():
                print(f"[skip] {folder} not found")
                continue
            print(f"[{key}-{task}]")
            lens = lengths_in_folder(folder, field)
            if lens:
                stats[task][key] = {
                    "avg": statistics.mean(lens),
                    "median": statistics.median(lens),
                    "min": min(lens),
                    "max": max(lens),
                    "n": len(lens),
                }
    return stats


def demo_stats() -> dict:
    """Illustrative placeholder values (tokens) just to preview the figure layout."""
    return {
        "helpfulness": {
            "base": {"avg": 178.0, "median": 123.0, "n": 4096}, "shp": {"avg": 156.0, "median": 113.0, "n": 4096},
            "helpsteer": {"avg": 211.0, "median": 160.0, "n": 4096}, "synthetic": {"avg": 114.0, "median": 86.0, "n": 4096},
            "mixed": {"avg": 130.0, "median": 96.0, "n": 4096},
        },
        "summarization": {
            "base": {"avg": 121.0, "median": 111.0, "n": 4096}, "tldr": {"avg": 61.0, "median": 57.0, "n": 4096},
            "synthetic": {"avg": 75.0, "median": 70.0, "n": 4096}, "mixed": {"avg": 69.0, "median": 65.0, "n": 4096},
        },
    }


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
    p.add_argument("--out", type=Path, default=Path("response_lengths.png"))
    p.add_argument("--field", default="response", help="JSON key holding the model response")
    p.add_argument("--demo", action="store_true", help="plot illustrative example values instead of reading data")
    args = p.parse_args()

    stats = demo_stats() if args.demo else compute_stats(ROOT, args.field)
    if not args.demo:
        save_csv(stats, args.out.with_suffix(".csv"))
    plot(stats, args.out)


if __name__ == "__main__":
    main()