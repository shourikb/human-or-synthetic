import argparse
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from generation.utils import read_jsonl, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Select synthetic conversations whose token-length distribution matches a human reference set")
    parser.add_argument("--input_file", type=str, required=True, help="Filtered synthetic conversations")
    parser.add_argument("--reference_file", type=str, required=True, help="Human preference data with a 'chosen' field")
    parser.add_argument("--existing_file", type=str, default=None,
                        help="Previously selected conversations to keep and extend")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="allenai/Llama-3.1-Tulu-3-8B-SFT")
    parser.add_argument("--bin_size", type=int, default=50)
    parser.add_argument("--histogram_file", type=str, default=None)
    return parser.parse_args()


def synthetic_text(item):
    return item["conversation"] + " " + item["response_A"]


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    def length(text):
        return len(tokenizer.encode(text.lower()))

    def bin_of(text):
        return (length(text) // args.bin_size) * args.bin_size

    reference_bins = Counter(bin_of(item["chosen"]) for item in read_jsonl(args.reference_file))
    reference_total = sum(reference_bins.values())
    reference_probs = {b: c / reference_total for b, c in reference_bins.items()}
    print(f"Reference bin counts: {reference_bins}")

    existing = read_jsonl(args.existing_file) if args.existing_file else []
    existing_bins = Counter(bin_of(synthetic_text(item)) for item in existing)
    print(f"Loaded {len(existing)} existing conversations")

    candidates = [(bin_of(synthetic_text(item)), item) for item in read_jsonl(args.input_file)]
    available = Counter(b for b, _ in candidates)
    print(f"Candidates: {len(candidates)}")

    targets = {
        b: min(int(reference_probs[b] * len(candidates)), max(0, available[b] - existing_bins[b]))
        for b in reference_bins
    }
    selected, selected_bins, leftovers = [], Counter(), defaultdict(list)
    for b, item in candidates:
        if selected_bins[b] < targets.get(b, 0):
            selected_bins[b] += 1
            selected.append(item)
        else:
            leftovers[b].append(item)
    print(f"Phase 1 selected {len(selected)} conversations")

    current_bins = existing_bins + selected_bins
    current_total = sum(current_bins.values())
    current_probs = {b: c / current_total for b, c in current_bins.items()} if current_total else {}
    max_extra = int(min((len(leftovers[b]) / p for b, p in current_probs.items() if p > 0 and leftovers.get(b)),
                        default=0))
    extra_targets = {b: int(p * max_extra) for b, p in current_probs.items()}
    extra_bins = Counter()
    for b, items in leftovers.items():
        for item in items:
            if extra_bins[b] < extra_targets.get(b, 0):
                extra_bins[b] += 1
                selected.append(item)
    print(f"Phase 2 added {sum(extra_bins.values())} conversations")

    output = existing + selected
    write_jsonl(args.output_file, output)
    print(f"Wrote {len(output)} conversations ({len(existing)} existing, {len(selected)} new) -> {args.output_file}")

    final_bins = existing_bins + selected_bins + extra_bins
    final_total = sum(final_bins.values())
    print(f"{'Bin':>6}  {'Reference%':>10}  {'Final%':>8}  {'Count':>8}")
    for b in sorted(reference_probs):
        print(f"{b:>6}  {reference_probs[b] * 100:>9.2f}%  {final_bins[b] / final_total * 100:>7.2f}%  {final_bins[b]:>8}")

    if args.histogram_file:
        lengths = [length(synthetic_text(item)) for item in output]
        plt.rcParams["pdf.fonttype"] = 42
        plt.figure(figsize=(8, 5))
        plt.hist(lengths, bins=range(0, max(lengths) + args.bin_size, args.bin_size), color="green", alpha=0.7)
        plt.xlabel("Conversation Length (tokens)")
        plt.ylabel("Number of Conversations")
        plt.title(f"Histogram of Conversation Token Lengths\n{args.output_file}")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(args.histogram_file)
        plt.close()


if __name__ == "__main__":
    main()
