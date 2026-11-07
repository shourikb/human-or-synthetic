import argparse
import glob
import json

from generation.utils import write_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Merge per-run preference judgment shards into one file sorted by item index")
    parser.add_argument("--input_glob", type=str, required=True,
                        help="Glob matching shard files, e.g. 'majority_votes_32_synthetic_run*_offset0.jsonl'")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--global_offset", type=int, default=0)
    parser.add_argument("--expected_count", type=int, default=None,
                        help="If given, report indices in [global_offset, global_offset + expected_count) that are missing")
    return parser.parse_args()


def main():
    args = parse_args()

    records = []
    for path in sorted(glob.glob(args.input_glob)):
        print(f"Reading {path}")
        with open(path) as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  Skipping bad line: {e}")

    records.sort(key=lambda r: r.get("global_conv_idx", 0))
    write_jsonl(args.output_file, records)
    indices = {r.get("global_conv_idx") for r in records}
    print(f"Merged {len(records)} records ({len(indices)} unique indices) -> {args.output_file}")

    if args.expected_count is not None:
        expected = set(range(args.global_offset, args.global_offset + args.expected_count))
        missing = sorted(expected - indices)
        if missing:
            print(f"WARNING: {len(missing)} missing indices, first few: {missing[:20]}")
        else:
            print(f"All {args.expected_count} items accounted for")


if __name__ == "__main__":
    main()
