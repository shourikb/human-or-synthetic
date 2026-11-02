"""
Randomly samples 50 unique lines from a JSONL file and writes them to a
new JSONL file.

Usage:
    python sample_jsonl.py
"""
import json
import random

INPUT_FILE = "reddit_test.jsonl"
OUTPUT_FILE = "summarization_validation.jsonl"
N = 50
SEED = 55

def main():
    if SEED is not None:
        random.seed(SEED)

    with open(INPUT_FILE, "r") as f:
        lines = [line for line in f if line.strip()]

    if len(lines) < N:
        raise ValueError(f"Requested {N} lines, but {INPUT_FILE} only has {len(lines)}.")

    # Sample indices rather than lines directly, so we can report which
    # original (1-indexed) line numbers were picked.
    sampled_indices = random.sample(range(len(lines)), N)
    sampled_indices.sort()  # report in ascending order rather than random order

    print(f"Sampled line numbers (1-indexed): {[i + 1 for i in sampled_indices]}")

    with open(OUTPUT_FILE, "w") as f:
        for idx in sampled_indices:
            row = json.loads(lines[idx])
            f.write(json.dumps(row) + "\n")

    print(f"Sampled {len(sampled_indices)} lines from {INPUT_FILE} -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()