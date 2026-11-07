import argparse
import re
from collections import Counter

import nltk
from nltk.util import ngrams

from generation.utils import read_jsonl, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Drop conversations containing corpus-common or self-repeated 5-grams")
    parser.add_argument("--reference_files", type=str, nargs="+", required=True,
                        help="Files used to compute corpus-wide 5-gram frequencies")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--threshold_frac", type=float, default=0.05,
                        help="A 5-gram is common if it appears more than this fraction of reference conversations")
    parser.add_argument("--max_repeats", type=int, default=2,
                        help="Drop a conversation if any 5-gram repeats more than this many times within it")
    return parser.parse_args()


def get_5grams(text):
    cleaned = re.sub(r"(Human:|AI Assistant:)", "", text)
    return list(ngrams(nltk.word_tokenize(cleaned.lower()), 5))


def entry_text(entry):
    return entry.get("conversation", "") + " " + entry.get("response_A", "")


def is_exempt(gram):
    joined = " ".join(gram)
    return "human" in joined or "assistant" in joined or joined == "i 'd be happy to help"


def main():
    args = parse_args()
    nltk.download("punkt", quiet=True)

    freq = Counter()
    num_reference = 0
    for path in args.reference_files:
        for entry in read_jsonl(path):
            freq.update(get_5grams(entry_text(entry)))
            num_reference += 1

    threshold = int(num_reference * args.threshold_frac)
    print(f"5-gram frequency threshold: {threshold}")
    common = {gram for gram, count in freq.items() if count > threshold and not is_exempt(gram)}

    entries = read_jsonl(args.input_file)
    kept = []
    for entry in entries:
        grams = get_5grams(entry_text(entry))
        has_common = any(gram in common for gram in grams)
        has_repeat = any(c > args.max_repeats for c in Counter(grams).values())
        if not has_common and not has_repeat:
            kept.append(entry)

    write_jsonl(args.output_file, kept)
    print(f"Kept {len(kept)} / {len(entries)} conversations -> {args.output_file}")


if __name__ == "__main__":
    main()
