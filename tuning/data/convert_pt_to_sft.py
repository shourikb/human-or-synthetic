import argparse
import json

from tuning.utils.conversation import split_conversation


def convert(input_path, output_path):
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            example = json.loads(line)
            messages = split_conversation(example["chosen"])
            fout.write(json.dumps({"messages": messages}) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="e.g. pt-synsum.jsonl")
    parser.add_argument("--output_file", type=str, required=True, help="e.g. sft-synsum.jsonl")
    args = parser.parse_args()
    convert(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
