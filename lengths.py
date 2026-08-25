from transformers import AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B")

def build_text(post, summary):
    return f"Human: Summarize the following text\n\n{post}\n\nAssistant: {summary}"

lengths = []
with open("dpo-training/datasets/pt-reddit.jsonl") as f:
    for line in f:
        row = json.loads(line)
        lengths.append(len(tokenizer.encode(row["chosen"])))
lengths.sort()
n = len(lengths)
print(f"pt-reddit.jsonl: n={n}, max={lengths[-1]}, p99={lengths[int(n*0.99)]}, p95={lengths[int(n*0.95)]}, p50={lengths[n//2]}")

lengths = []
with open("finalposts.jsonl") as f:
    for line in f:
        row = json.loads(line)
        len_a = len(tokenizer.encode(build_text(row["post"], row["summary_A"])))
        len_b = len(tokenizer.encode(build_text(row["post"], row["summary_B"])))
        lengths.append(max(len_a, len_b))  # worst case, since chosen/rejected assignment isn't known yet

lengths.sort()
n = len(lengths)
print(f"finalposts.jsonl: n={n}, max={lengths[-1]}, p99={lengths[int(n*0.99)]}, p95={lengths[int(n*0.95)]}, p50={lengths[n//2]}")