import argparse
import json


def convert(input_path, output_path):
    written = 0
    skipped = 0

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            row = json.loads(line)

            strength = row["preference_strength"]
            if strength == 0:
                skipped += 1
                continue

            r1 = row["response_1"]
            r2 = row["response_2"]
            prompt = row["prompt"]

            if strength > 0:
                chosen_resp, rejected_resp = r2, r1
            else:
                chosen_resp, rejected_resp = r1, r2

            chosen = f"Human: {prompt}\n\nAssistant: {chosen_resp}"
            rejected = f"Human: {prompt}\n\nAssistant: {rejected_resp}"

            fout.write(json.dumps({"chosen": chosen, "rejected": rejected}) + "\n")
            written += 1

    print(f"{input_path} -> {output_path}: written={written}, skipped={skipped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="HelpSteer2 preference split as JSONL")
    parser.add_argument("--output_file", type=str, required=True, help="e.g. pt-helpsteer2.jsonl")
    args = parser.parse_args()
    convert(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
