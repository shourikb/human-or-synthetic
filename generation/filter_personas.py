import argparse
import json

import fasttext


def parse_args():
    parser = argparse.ArgumentParser(description="Keep only English personas from a persona jsonl file")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--lid_model", type=str, default="lid.176.bin",
                        help="Path to the fastText language-ID model (lid.176.bin)")
    return parser.parse_args()


def main():
    args = parse_args()
    model = fasttext.load_model(args.lid_model)

    kept = 0
    with open(args.input_file, "r", encoding="utf-8") as fin, \
            open(args.output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            persona = next(iter(json.loads(line).values()))
            language = model.predict(persona, k=1)[0][0].replace("__label__", "")
            if language == "en":
                fout.write(json.dumps({"persona": persona}) + "\n")
                kept += 1

    print(f"Kept {kept} English personas -> {args.output_file}")


if __name__ == "__main__":
    main()
