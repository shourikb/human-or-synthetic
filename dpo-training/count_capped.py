"""
Count how many responses ending mid-sentence reached the 2,048-token generation limit,
using each model's own tokenizer.

Usage:
    python count_capped.py responses_models_*.jsonl

Note: generate_responses.py cuts a response at any line starting with a role tag
(e.g. "Human:"), so a generation that hit the cap and was then cut will tokenize to
fewer than 2,048 tokens and is not counted here. Saving vLLM's finish_reason at
generation time is the only fully exact check.
"""
import json
import re
import sys

from transformers import AutoTokenizer

MAX_TOKENS = 2048
SLACK = 8  # re-tokenizing decoded text can differ from the generated ids by a few tokens

MODEL_TOKENIZERS = [
    ("olmo", "allenai/Olmo-3-1025-7B"),
    ("llama", "unsloth/Meta-Llama-3.1-8B"),
    ("qwen", "unsloth/Qwen2.5-7B"),
]

MID = re.compile(r"[A-Za-z,;:-]$")
_cache = {}


def tokenizer_for(path):
    for key, tok_id in MODEL_TOKENIZERS:
        if key in path.lower():
            if tok_id not in _cache:
                _cache[tok_id] = AutoTokenizer.from_pretrained(tok_id)
            return _cache[tok_id]
    raise ValueError(f"no tokenizer configured for {path}")


print(f"{'file':70} {'mid':>5} {'capped':>7} {'PTR excl. capped':>17}")
for path in sys.argv[1:]:
    tok = tokenizer_for(path)
    responses = [json.loads(line)["response"] for line in open(path) if line.strip()]
    n_tokens = [len(x) for x in tok(responses, add_special_tokens=False)["input_ids"]]
    capped = [n >= MAX_TOKENS - SLACK for n in n_tokens]
    mid = [bool(MID.search(r.strip())) for r in responses]

    mid_total = sum(mid)
    mid_capped = sum(m and c for m, c in zip(mid, capped))
    uncapped = [m for m, c in zip(mid, capped) if not c]
    ptr = sum(uncapped) / len(uncapped)
    print(f"{path[-70:]:70} {mid_total:5} {mid_capped:7} {ptr:16.1%}")