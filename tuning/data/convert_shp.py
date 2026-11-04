import argparse
import json
from collections import defaultdict

MAX_PAIRS_PER_POST = 5


def convert(input_path, output_path, max_pairs_per_post=MAX_PAIRS_PER_POST):
    written = 0
    skipped_weak = 0
    skipped_bad_label = 0
    skipped_cap = 0

    pairs_by_post = defaultdict(list)

    with open(input_path) as fin:
        for line in fin:
            row = json.loads(line)

            score_ratio = row.get("score_ratio", 0)

            if score_ratio < 2.0:
                skipped_weak += 1
                continue

            label = row["labels"]

            ref_a = row["human_ref_A"]
            ref_b = row["human_ref_B"]
            prompt = row["history"]
            post_id = row["post_id"]

            if label == 1:
                chosen_resp, rejected_resp = ref_a, ref_b
            elif label == 0:
                chosen_resp, rejected_resp = ref_b, ref_a
            else:
                skipped_bad_label += 1
                continue

            chosen = f"Human: {prompt}\n\nAssistant: {chosen_resp}"
            rejected = f"Human: {prompt}\n\nAssistant: {rejected_resp}"

            pairs_by_post[post_id].append((score_ratio, chosen, rejected))

    with open(output_path, "w") as fout:
        for post_id, pairs in pairs_by_post.items():
            pairs.sort(key=lambda p: p[0], reverse=True)

            kept = pairs[:max_pairs_per_post]
            skipped_cap += len(pairs) - len(kept)

            for _, chosen, rejected in kept:
                fout.write(json.dumps({"chosen": chosen, "rejected": rejected}) + "\n")
                written += 1

    total_skipped = skipped_weak + skipped_bad_label + skipped_cap
    print(
        f"{input_path} -> {output_path}: "
        f"written={written}, skipped_weak={skipped_weak}, "
        f"skipped_bad_label={skipped_bad_label}, skipped_cap={skipped_cap}, "
        f"total_skipped={total_skipped}, unique_posts={len(pairs_by_post)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="SHP train split as JSONL")
    parser.add_argument("--output_file", type=str, required=True, help="e.g. pt-shp.jsonl")
    parser.add_argument("--max_pairs_per_post", type=int, default=MAX_PAIRS_PER_POST)
    args = parser.parse_args()
    convert(args.input_file, args.output_file, args.max_pairs_per_post)


if __name__ == "__main__":
    main()
