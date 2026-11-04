import argparse
import json
import re


def parse_conversation(text):
    messages = []
    turns = re.split(r'\s+(Human|AI Assistant|Assistant):\s*', text.strip())
    if turns[0]:
        match = re.match(r'^(Human|AI Assistant|Assistant): (.+)$', turns[0], re.DOTALL)
        if match:
            role = "user" if match.group(1) == "Human" else "assistant"
            messages.append({"role": role, "content": match.group(2).strip()})
    for i in range(1, len(turns), 2):
        role = "user" if turns[i] == "Human" else "assistant"
        content = turns[i + 1].strip()
        messages.append({"role": role, "content": content})
    return messages


def convert(input_path, output_path):
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            example = json.loads(line)
            messages = parse_conversation(example["chosen"])
            fout.write(json.dumps({"messages": messages}) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="e.g. pt-synsum.jsonl")
    parser.add_argument("--output_file", type=str, required=True, help="e.g. sft-synsum.jsonl")
    args = parser.parse_args()
    convert(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
