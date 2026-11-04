import json
import os
import random
import re

import numpy as np


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows, mode="w"):
    with open(path, mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def count_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def load_personas(path):
    with open(path, "r", encoding="utf-8") as f:
        return [next(iter(json.loads(line).values())) for line in f]


def set_visible_gpus(gpu_id):
    gpu_ids = [x.strip() for x in gpu_id.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    return gpu_ids


def reset_seeds(seed=None):
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="big")
    random.seed(seed)
    np.random.seed(seed)
    return seed


def chop_before_prefix(s, prefix):
    index = s.find(prefix)
    return s[index:] if index != -1 else s


def chop_after_prefix(s, prefix):
    index = s.find(prefix)
    return s[:index] if index != -1 else s


def run_slice(run_id, total_runs, num_items):
    per_run = num_items // total_runs
    start = run_id * per_run
    end = num_items if run_id == total_runs - 1 else start + per_run
    return start, end


def load_completed_batches(output_path):
    completed = set()
    if not os.path.exists(output_path):
        return completed
    with open(output_path, "r") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "batch_idx" in item:
                completed.add(item["batch_idx"])
    print(f"Found {len(completed)} completed batches in {output_path}")
    return completed


def append_records(output_path, records):
    payload = "".join(json.dumps(r) + "\n" for r in records)
    with open(output_path, "a") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())


def parse_judgments(outputs):
    all_judgments = []
    for output in outputs:
        judgments = []
        for j, sample in enumerate(output.outputs):
            text = sample.text
            think_match = re.search(r"<think>\s*(.*?)\s*</think>", text, re.DOTALL)
            answer_match = re.search(r'"answer"\s*:\s*"([^"]+)"', text)
            reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', text)
            judgments.append({
                "sample_id": j,
                "answer": answer_match.group(1).strip() if answer_match else None,
                "reasoning": reasoning_match.group(1).strip() if reasoning_match else None,
                "thinking": think_match.group(1).strip() if think_match else None,
                "output_tokens": len(sample.token_ids),
            })
        all_judgments.append(judgments)
    return all_judgments
