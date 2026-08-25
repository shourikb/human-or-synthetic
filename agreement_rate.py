#!/usr/bin/env python3
"""
Compute per-line agreement rate and the overall average agreement rate
for a judging .jsonl file.

Each line is expected to be a JSON object containing a "judgments" field:
a list of judgment objects, each with an "answer" field (e.g. "A" or "B").

"Agreement rate" for a line = (number of judgments matching the majority
answer) / (total number of judgments on that line).

The script prints the per-line agreement rate and the overall average
across all lines.

Usage:
    python agreement_rate.py path/to/file.jsonl
"""

import argparse
import json
import sys
from collections import Counter


def compute_line_agreement(judgments):
    """
    Given a list of judgment dicts (each with an 'answer' key),
    return the fraction that agree with the majority answer.
    Returns None if there are no judgments.
    """
    answers = [j.get("answer") for j in judgments if j.get("answer") is not None]
    if not answers:
        return None

    counts = Counter(answers)
    majority_count = counts.most_common(1)[0][1]
    return majority_count / len(answers)


def main():
    parser = argparse.ArgumentParser(description="Compute agreement rate from a judging jsonl file.")
    parser.add_argument("jsonl_path", help="Path to the .jsonl file")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print per-line agreement rates as they are computed"
    )
    args = parser.parse_args()

    line_agreements = []

    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as e:
                print(f"Warning: could not parse line {line_num}: {e}", file=sys.stderr)
                continue

            judgments = record.get("judgments", [])
            agreement = compute_line_agreement(judgments)

            if agreement is None:
                print(f"Warning: line {line_num} has no valid judgments; skipping.", file=sys.stderr)
                continue

            line_agreements.append(agreement)

            if args.verbose:
                idx = record.get("global_conv_idx", line_num)
                print(f"Line {line_num} (idx={idx}): agreement rate = {agreement:.4f} "
                      f"({len(judgments)} judgments)")

    if not line_agreements:
        print("No valid lines with judgments found.", file=sys.stderr)
        sys.exit(1)

    average_agreement = sum(line_agreements) / len(line_agreements)

    print(f"\nProcessed {len(line_agreements)} lines.")
    print(f"Average agreement rate: {average_agreement:.4f}")


if __name__ == "__main__":
    main()