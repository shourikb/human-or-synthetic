import argparse
from collections import Counter

import nltk
from nltk.util import ngrams

from generation.utils import read_jsonl, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Drop posts containing corpus-common or self-repeated 5-grams")
    parser.add_argument("--reference_files", type=str, nargs="+", required=True,
                        help="Files used to compute corpus-wide 5-gram frequencies")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--threshold_frac", type=float, default=0.05,
                        help="A 5-gram is common if it appears more than this fraction of reference posts")
    parser.add_argument("--max_repeats", type=int, default=2,
                        help="Drop a post if any 5-gram repeats more than this many times within it")
    return parser.parse_args()


def get_5grams(text):
    return list(ngrams(nltk.word_tokenize(text.lower()), 5))


def main():
    args = parse_args()
    nltk.download("punkt", quiet=True)

    freq = Counter()
    num_reference = 0
    for path in args.reference_files:
        for entry in read_jsonl(path):
            freq.update(get_5grams(entry.get("post", "")))
            num_reference += 1

    threshold = int(num_reference * args.threshold_frac)
    print(f"5-gram frequency threshold: {threshold}")
    common = {gram for gram, count in freq.items() if count > threshold}
    print(f"Common 5-grams: {common}")

    entries = read_jsonl(args.input_file)
    kept = []
    for entry in entries:
        grams = get_5grams(entry.get("post", ""))
        has_common = any(gram in common for gram in grams)
        has_repeat = any(c > args.max_repeats for c in Counter(grams).values())
        if not has_common and not has_repeat:
            kept.append(entry)

    write_jsonl(args.output_file, kept)
    print(f"Kept {len(kept)} / {len(entries)} posts -> {args.output_file}")


if __name__ == "__main__":
    main()
