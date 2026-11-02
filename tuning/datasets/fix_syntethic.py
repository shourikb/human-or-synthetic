import json

input_file = "pt-synthetic.jsonl"
output_file = "pt-synthetic-n.jsonl"

with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
    for line in f_in:
        record = json.loads(line)
        record["chosen"] = record["chosen"].replace("AI Assistant:", "Assistant:")
        record["rejected"] = record["rejected"].replace("AI Assistant:", "Assistant:")
        f_out.write(json.dumps(record) + "\n")

print("Done")