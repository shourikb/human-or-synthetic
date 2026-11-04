import argparse
import json
import os
import random

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


def generate_mode(mode_name, splits, seed, human_file, synthetic_file, output_dir, output_suffix):
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

        out_path = os.path.join(output_dir, f"{mode_name}-{label}{output_suffix}.jsonl")
        with open(out_path, "w") as out:
            for line in combined:
                row = json.loads(line)
                out.write(json.dumps(row) + "\n")

        print(
            f"[{mode_name} {label}] human={len(human_lines)} "
            f"synthetic={len(synthetic_lines)} total={len(combined)} -> {out_path}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_human_file", type=str, required=True)
    parser.add_argument("--sft_synthetic_file", type=str, required=True)
    parser.add_argument("--pt_human_file", type=str, required=True)
    parser.add_argument("--pt_synthetic_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_suffix", type=str, default="")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    generate_mode("sft", SFT_SPLITS, SFT_SEED, args.sft_human_file, args.sft_synthetic_file,
                  args.output_dir, args.output_suffix)
    print()
    generate_mode("pt", DPO_SPLITS, DPO_SEED, args.pt_human_file, args.pt_synthetic_file,
                  args.output_dir, args.output_suffix)


if __name__ == "__main__":
    main()
