import argparse
import json


def format_context(context, final_response):
    parts = []
    for turn in context:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            parts.append("Human: " + content)
        elif role == "assistant":
            parts.append("Assistant: " + content)
    parts.append("Assistant: " + final_response)
    return "\n\n".join(parts)


def convert(input_path, output_path):
    written = 0
    skipped = 0

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            row = json.loads(line)

            if row.get("domain") == "multilingual" or row.get("domain") == "code" or row.get("domain") == "stem":
                skipped += 1
                continue

            pref = row["overall_preference"]
            if pref == 0:
                skipped += 1
                continue

            r1 = row["response1"]
            r2 = row["response2"]

            if pref > 0:
                chosen_resp, rejected_resp = r2, r1
            else:
                chosen_resp, rejected_resp = r1, r2

            context = row["context"]
            chosen = format_context(context, chosen_resp)
            rejected = format_context(context, rejected_resp)

            fout.write(json.dumps({"chosen": chosen, "rejected": rejected}) + "\n")
            written += 1

    print(f"{input_path} -> {output_path}: written={written}, skipped={skipped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="HelpSteer3 preference split as JSONL")
    parser.add_argument("--output_file", type=str, required=True, help="e.g. pt-helpsteer3-g.jsonl")
    args = parser.parse_args()
    convert(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
