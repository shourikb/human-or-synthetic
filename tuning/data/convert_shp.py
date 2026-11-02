import json
from collections import defaultdict

MAX_PAIRS_PER_POST = 5  # SHP README recommendation, to avoid over-fitting to a few viral posts


def convert(input_path, output_path, max_pairs_per_post=MAX_PAIRS_PER_POST):
    written = 0
    skipped_weak = 0
    skipped_bad_label = 0
    skipped_cap = 0

    # First pass: filter + build (post_id -> list of candidate pairs), sorted by score_ratio
    pairs_by_post = defaultdict(list)

    with open(input_path) as fin:
        for line in fin:
            row = json.loads(line)

            score_ratio = row.get("score_ratio", 0)

            # Drop weak preference pairs, analogous to helpsteer2 strength==0
            if score_ratio < 2.0:
                skipped_weak += 1
                continue

            label = row["labels"]

            ref_a = row["human_ref_A"]
            ref_b = row["human_ref_B"]
            prompt = row["history"]
            post_id = row["post_id"]

            # labels == 1 means A is preferred, labels == 0 means B is preferred
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

    # Second pass: cap pairs per post, keeping the highest-score_ratio (strongest signal) ones
    with open(output_path, "w") as fout:
        for post_id, pairs in pairs_by_post.items():
            pairs.sort(key=lambda p: p[0], reverse=True)  # strongest preference first

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


if __name__ == "__main__":
    convert("shp.jsonl", "pt-shp.jsonl")