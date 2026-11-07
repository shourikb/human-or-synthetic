import argparse
import json


def convert(input_path, output_path):
    written = 0
    skipped = 0

    with open(input_path, "r") as fin, open(output_path, "w") as fout:
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

    print(f"{input_path} -> {output_path}: written={written}, skipped={skipped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="TL;DR comparisons as JSONL")
    parser.add_argument("--output_file", type=str, required=True, help="e.g. pt-reddit.jsonl")
    args = parser.parse_args()
    convert(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
