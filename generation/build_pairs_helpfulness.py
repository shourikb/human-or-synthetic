import argparse
from collections import Counter

from generation.utils import read_jsonl, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Turn majority-vote judgments into chosen/rejected conversation pairs")
    parser.add_argument("--items_file", type=str, required=True, help="conversation/response_A/response_B records that were labeled")
    parser.add_argument("--votes_file", type=str, required=True, help="Merged judgments keyed by global_conv_idx")
    parser.add_argument("--output_file", type=str, required=True)
    return parser.parse_args()


def majority_answer(judgments):
    votes = Counter(j["answer"].strip().lower() for j in judgments if j.get("answer"))
    if votes["a"] == votes["b"]:
        return None
    return "a" if votes["a"] > votes["b"] else "b"


def to_transcript(text):
    return text.replace("AI Assistant:", "Assistant:")


def main():
    args = parse_args()
    votes = {r["global_conv_idx"]: r for r in read_jsonl(args.votes_file)}

    pairs = []
    missing = skipped = 0
    for idx, item in enumerate(read_jsonl(args.items_file)):
        if idx not in votes:
            missing += 1
            continue
        answer = majority_answer(votes[idx].get("judgments", []))
        if answer is None:
            skipped += 1
            continue
        chosen, rejected = ("response_A", "response_B") if answer == "a" else ("response_B", "response_A")
        conversation = item.get("conversation", "")
        pairs.append({
            "chosen": to_transcript(conversation + " " + item.get(chosen, "")),
            "rejected": to_transcript(conversation + " " + item.get(rejected, "")),
        })

    write_jsonl(args.output_file, pairs)
    print(f"Wrote {len(pairs)} pairs -> {args.output_file}")
    print(f"Skipped {skipped} ties/unparseable, {missing} without judgments")


if __name__ == "__main__":
    main()
