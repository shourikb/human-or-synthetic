import json
import re

# adjust this to your actual file path
file_path = "sample5.jsonl"
file_path_preference = "majority_votes_20000_synthetic_merged_offset0.jsonl"

SCALING_CONSTANT = 57856 / 45793
NUM_SAMPLES = 32

total_input = 0
total_output = 0
lines = 0
lines_preference = 0

total_input_preference = 0
total_output_preference = 0

with open(file_path, "r") as f:
    for line in f:
        data = json.loads(line)
        info = data.get("info", "")

        # find the token counts using regex
        input_match = re.search(r"Total Input Tokens:\s*(\d+)", info)
        output_match = re.search(r"Total Output Tokens:\s*(\d+)", info)

        if input_match:
            total_input += int(input_match.group(1))
        if output_match:
            total_output += int(output_match.group(1))
        
        lines += 1

with open(file_path_preference, "r") as f:
    for line in f:
        if not line.strip():
            continue 
        data = json.loads(line)

        # Adjust these keys if they’re nested (e.g., data["stats"]["input tokens"])
        if "input_tokens" in data:
            total_input_preference += data["input_tokens"] * NUM_SAMPLES

        judgements = data.get("judgments", [])
        for judgment in judgements:
            total_output_preference += judgment.get("output_tokens", 0)
        
        lines_preference += 1

total_input = (total_input / lines) * (lines_preference * SCALING_CONSTANT)
total_output = (total_output / lines) * (lines_preference * SCALING_CONSTANT)

# example prices (change as needed)
price_per_1M_input = 0.05  # dollars per 1M input tokens
price_per_1M_output = 0.08  # dollars per 1M output tokens

price_per_1M_input_preference = 0.29 # dollars per 1M input tokens for preference model
price_per_1M_output_preference = 0.59 # dollars per 1M output tokens for preference model
# https://www.alibabacloud.com/help/en/model-studio/model-pricing?utm_source=chatgpt.com#9d4f9a71e1prj

total_input_cost = (total_input / 1000000) * price_per_1M_input
total_output_cost = (total_output / 1000000) * price_per_1M_output

total_input_preference_cost = (total_input_preference / 1000000) * price_per_1M_input_preference
total_output_preference_cost = (total_output_preference / 1000000) * price_per_1M_output_preference

total_cost = total_input_cost + total_output_cost + total_input_preference_cost + total_output_preference_cost


print(f"Total input tokens: {total_input / 1000000}M")
print(f"Total output tokens: {total_output / 1000000}M")
print(f"Average input tokens per line: {total_input / (lines_preference * SCALING_CONSTANT) if lines else 0}")
print(f"Average output tokens per line: {total_output / (lines_preference * SCALING_CONSTANT) if lines else 0}")
print(f"Total input tokens preference: {total_input_preference / 1000000}M")
print(f"Total output tokens preference: {total_output_preference / 1000000}M")
print(f"Average input preference tokens per line: {total_input_preference / (lines_preference * NUM_SAMPLES) if lines else 0}")
print(f"Average output preference tokens per line: {total_output_preference / (lines_preference * NUM_SAMPLES)if lines else 0}")
print(f"Input cost: ${total_input_cost:.4f}")
print(f"Output cost: ${total_output_cost:.4f}")
print(f"Input preference cost: ${total_input_preference_cost:.4f}")
print(f"Output preference cost: ${total_output_preference_cost:.4f}")
print(f"Total input cost: ${total_input_cost + total_input_preference_cost:.4f}")
print(f"Total output cost: ${total_output_cost + total_output_preference_cost:.4f}")
print(f"Estimated cost: ${total_cost:.4f}")
