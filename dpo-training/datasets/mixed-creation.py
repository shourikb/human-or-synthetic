"""
Generates 5 progressively-shifting SFT/DPO training mixes from two source
JSONL datasets (human-authored preference data + synthetic preference data).

Unlike independently re-sampling each split from scratch, this keeps the
splits NESTED:
  - Human samples ACCUMULATE across splits: 96/4's human set = 95/5's human
    set + exactly the new samples needed to reach 96/4's target count.
  - Synthetic samples SHRINK across splits: 96/4's synthetic set is a
    subsample drawn FROM 95/5's synthetic set (not from the full pool), so
    every later split's synthetic data is a strict subset of every earlier
    split's synthetic data.

Usage:
    python combine_datasets.py
"""
import json
import random

SFT_HUMAN_FILE = "sft-reddit.jsonl"
SFT_SYNTHETIC_FILE = "sft-synsum.jsonl"

DPO_HUMAN_FILE = "pt-reddit.jsonl"
DPO_SYNTHETIC_FILE = "pt-synsum.jsonl"

# Each entry: (split_label, n_human, n_synthetic), in the order you want them
# processed (order doesn't matter here -- they get re-sorted internally).
SFT_SPLITS = [
    ("955", 9499, 33817),
    ("964", 9599, 27054),
    ("973", 9699, 20290),
    ("982", 9799, 13527),
    ("991", 9899, 6763),
]

DPO_SPLITS = [
    ("955", 9499, 33754),
    ("964", 9599, 27003),
    ("973", 9699, 20252),
    ("982", 9799, 13501),
    ("991", 9899, 6750),
]

SFT_SEED = 55  
DPO_SEED = 55 


def load_lines(path):
    with open(path, "r") as f:
        return [line for line in f if line.strip()]


def build_growing_samples(splits_asc, pool, rng):
    """
    splits_asc: [(label, n_needed), ...] sorted ascending by n_needed.
    Returns {label: [lines]} where each split's list contains the previous
    split's list plus newly sampled lines to make up the difference, so
    every split is a strict superset of the one before it.
    """
    results = {}
    chosen = []
    remaining = list(pool)
    for label, n_needed in splits_asc:
        n_new = n_needed - len(chosen)
        if n_new < 0:
            raise ValueError(
                f"{label}: target {n_needed} is less than the {len(chosen)} "
                f"already accumulated -- growing splits must increase."
            )
        if n_new > len(remaining):
            raise ValueError(
                f"{label}: need {n_new} new lines but only {len(remaining)} "
                f"remain unused in the pool."
            )
        new_lines = rng.sample(remaining, n_new)
        new_ids = {id(x) for x in new_lines}
        remaining = [l for l in remaining if id(l) not in new_ids]
        chosen = chosen + new_lines
        results[label] = list(chosen)
    return results


def build_shrinking_samples(splits_desc, pool, rng):
    """
    splits_desc: [(label, n_needed), ...] sorted descending by n_needed.
    Returns {label: [lines]} where each split's list is a subsample of the
    PREVIOUS split's list (not the full pool), so every split is a strict
    subset of the one before it.
    """
    results = {}
    current = list(pool)
    for label, n_needed in splits_desc:
        if n_needed > len(current):
            raise ValueError(
                f"{label}: target {n_needed} exceeds the {len(current)} "
                f"available from the previous (larger) split."
            )
        current = rng.sample(current, n_needed)
        results[label] = list(current)
    return results


def generate_mode(mode_name, splits, seed, human_file, synthetic_file):
    rng = random.Random(seed)

    human_pool = load_lines(human_file)
    synthetic_pool = load_lines(synthetic_file)

    max_human = max(n_h for _, n_h, _ in splits)
    max_synthetic = max(n_s for _, _, n_s in splits)

    if len(human_pool) < max_human:
        raise ValueError(
            f"{human_file} only has {len(human_pool)} lines, need at least {max_human}."
        )
    if len(synthetic_pool) < max_synthetic:
        raise ValueError(
            f"{synthetic_file} only has {len(synthetic_pool)} lines, "
            f"need at least {max_synthetic}."
        )

    human_asc = sorted(((label, n_h) for label, n_h, _ in splits), key=lambda t: t[1])
    synthetic_desc = sorted(((label, n_s) for label, _, n_s in splits), key=lambda t: -t[1])

    human_by_label = build_growing_samples(human_asc, human_pool, rng)
    synthetic_by_label = build_shrinking_samples(synthetic_desc, synthetic_pool, rng)

    for label, n_h, n_s in splits:
        human_lines = human_by_label[label]
        synthetic_lines = synthetic_by_label[label]

        assert len(human_lines) == n_h, f"{label}: human count mismatch"
        assert len(synthetic_lines) == n_s, f"{label}: synthetic count mismatch"

        combined = human_lines + synthetic_lines
        rng.shuffle(combined)

        out_path = f"{mode_name}-{label}_sum.jsonl"
        with open(out_path, "w") as out:
            for line in combined:
                row = json.loads(line)
                out.write(json.dumps(row) + "\n")

        print(
            f"[{mode_name} {label}] human={len(human_lines)} "
            f"synthetic={len(synthetic_lines)} total={len(combined)} -> {out_path}"
        )


def main():
    generate_mode("sft", SFT_SPLITS, SFT_SEED, SFT_HUMAN_FILE, SFT_SYNTHETIC_FILE)
    print()
    generate_mode("pt", DPO_SPLITS, DPO_SEED, DPO_HUMAN_FILE, DPO_SYNTHETIC_FILE)


if __name__ == "__main__":
    main()