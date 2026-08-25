"""
Convert reddit_comparisons.jsonl into pt-reddit.jsonl, validating that each
row has 'chosen' and 'rejected' fields.
"""
import json

INPUT_FILE = "reddit_comparisons_full.jsonl"
OUTPUT_FILE = "pt-reddit_full.jsonl"


def main():
    written = 0
    skipped = 0

    with open(INPUT_FILE, "r") as fin, open(OUTPUT_FILE, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            if "chosen" not in row or "rejected" not in row:
                skipped += 1
                continue

            fout.write(json.dumps({"chosen": row["chosen"], "rejected": row["rejected"]}) + "\n")
            written += 1

    print(f"{INPUT_FILE} -> {OUTPUT_FILE}: written={written}, skipped={skipped}")


if __name__ == "__main__":
    main()