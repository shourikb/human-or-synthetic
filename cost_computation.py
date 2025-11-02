import json
import re

# adjust this to your actual file path
file_path = "sample5.jsonl"
file_path_preference = "choices_qwen.jsonl"

total_input = 0
total_output = 0
lines = 0

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
            continue  # skip empty lines
        data = json.loads(line)

        # Adjust these keys if they’re nested (e.g., data["stats"]["input tokens"])
        if "input tokens" in data:
            total_input_preference += data["input tokens"]
        if "normal tokens" in data:
            total_output_preference += data["normal tokens"]

# example prices (change as needed)
price_per_1M_input = 0.05  # dollars per 1M input tokens
price_per_1M_output = 0.08  # dollars per 1M output tokens

price_per_1M_input_preference = 0.7 # dollars per 1M input tokens for preference model
price_per_1M_output_preference = 8.4 # dollars per 1M output tokens for preference model

total_input_cost = (total_input / 1000000) * price_per_1M_input
total_output_cost = (total_output / 1000000) * price_per_1M_output

total_input_preference_cost = (total_input_preference / 1000000) * price_per_1M_input_preference
total_output_preference_cost = (total_output_preference / 1000000) * price_per_1M_output_preference

total_cost = total_input_cost + total_output_cost + total_input_preference_cost + total_output_preference_cost


print(f"Total input tokens: {total_input / 1000000}M")
print(f"Total output tokens: {total_output / 1000000}M")
print(f"Average input tokens per line: {total_input / lines if lines else 0}")
print(f"Average output tokens per line: {total_output / lines if lines else 0}")
print(f"Total input tokens preference: {total_input_preference / 1000000}M")
print(f"Total output tokens preference: {total_output_preference / 1000000}M")
print(f"Average input preference tokens per line: {total_input_preference / lines if lines else 0}")
print(f"Average output preference tokens per line: {total_output_preference / lines if lines else 0}")
print(f"Input cost: ${total_input_cost:.4f}")
print(f"Output cost: ${total_output_cost:.4f}")
print(f"Input preference cost: ${total_input_preference_cost:.4f}")
print(f"Output preference cost: ${total_output_preference_cost:.4f}")
print(f"Total input cost: ${total_input_cost + total_input_preference_cost:.4f}")
print(f"Total output cost: ${total_output_cost + total_output_preference_cost:.4f}")
print(f"Estimated cost: ${total_cost:.4f}")
