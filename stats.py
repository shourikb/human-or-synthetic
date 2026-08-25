from datasets import load_dataset
import tiktoken
import statistics

enc = tiktoken.get_encoding("cl100k_base")

ds = load_dataset("nvidia/HelpSteer3", split="train")
ds = ds.filter(lambda x: x["domain"] != "multilingual" and x["domain"] != "code" and x["domain"] != "stem")

def count_tokens(example):
    context_text = " ".join(m["content"] for m in example["context"])
    full_text = context_text + " " + example["response1"] + " " + example["response2"]
    return len(enc.encode(full_text, disallowed_special=()))

token_counts = [count_tokens(ex) for ex in ds]

print(f"Num conversations : {len(token_counts)}")
print(f"Avg tokens/conv   : {statistics.mean(token_counts):.1f}")
print(f"Median            : {statistics.median(token_counts):.1f}")
print(f"Stdev             : {statistics.stdev(token_counts):.1f}")
print(f"Min / Max         : {min(token_counts)} / {max(token_counts)}")
print(f"Total tokens      : {sum(token_counts):,}")