import argparse
import json
import re


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_files", nargs="+", required=True)
    parser.add_argument("--preference_file", required=True)
    parser.add_argument("--num_samples", type=int, default=32, help="Judgments sampled per preference row")
    parser.add_argument("--generation_input_price", type=float, default=0.05)
    parser.add_argument("--generation_output_price", type=float, default=0.08)
    parser.add_argument("--labelling_input_price", type=float, default=0.29)
    parser.add_argument("--labelling_output_price", type=float, default=0.59)
    return parser.parse_args()


def main():
    args = parse_args()

    gen_input = gen_output = gen_lines = 0
    for path in args.generation_files:
        with open(path, "r") as f:
            for line in f:
                info = json.loads(line).get("info", "")
                input_match = re.search(r"Total Input Tokens:\s*(\d+)", info)
                output_match = re.search(r"Total Output Tokens:\s*(\d+)", info)
                if input_match:
                    gen_input += int(input_match.group(1))
                if output_match:
                    gen_output += int(output_match.group(1))
                gen_lines += 1

    pref_input = pref_output = pref_lines = 0
    with open(args.preference_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if "input_tokens" in data:
                pref_input += data["input_tokens"] * args.num_samples
            for judgment in data.get("judgments", []):
                pref_output += judgment.get("output_tokens", 0)
            pref_lines += 1

    gen_input = gen_input / gen_lines * pref_lines
    gen_output = gen_output / gen_lines * pref_lines

    gen_input_cost = gen_input / 1e6 * args.generation_input_price
    gen_output_cost = gen_output / 1e6 * args.generation_output_price
    pref_input_cost = pref_input / 1e6 * args.labelling_input_price
    pref_output_cost = pref_output / 1e6 * args.labelling_output_price
    total_cost = gen_input_cost + gen_output_cost + pref_input_cost + pref_output_cost

    print(f"Total input tokens: {gen_input / 1e6}M")
    print(f"Total output tokens: {gen_output / 1e6}M")
    print(f"Average input tokens per line: {gen_input / pref_lines}")
    print(f"Average output tokens per line: {gen_output / pref_lines}")
    print(f"Total input tokens preference: {pref_input / 1e6}M")
    print(f"Total output tokens preference: {pref_output / 1e6}M")
    print(f"Average input preference tokens per line: {pref_input / (pref_lines * args.num_samples)}")
    print(f"Average output preference tokens per line: {pref_output / (pref_lines * args.num_samples)}")
    print(f"Input cost: ${gen_input_cost:.4f}")
    print(f"Output cost: ${gen_output_cost:.4f}")
    print(f"Input preference cost: ${pref_input_cost:.4f}")
    print(f"Output preference cost: ${pref_output_cost:.4f}")
    print(f"Total input cost: ${gen_input_cost + pref_input_cost:.4f}")
    print(f"Total output cost: ${gen_output_cost + pref_output_cost:.4f}")
    print(f"Estimated cost: ${total_cost:.4f}")
    print(f"Lines: {gen_lines}")
    print(f"Lines preference: {pref_lines}")


if __name__ == "__main__":
    main()
