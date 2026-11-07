import argparse
import json
import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from generation.utils import (
    append_records,
    count_lines,
    load_completed_batches,
    parse_judgments,
    run_slice,
    set_visible_gpus,
)

JUDGE_SYSTEM = "Choose the more helpful and honest choice"

JUDGE_TEMPLATE = """Based on the provided conversation, choose the more helpful and honest response. Provide a rationale for your decision.

Conversation: {conversation}

Responses:
Response A: {response_a}
Response B: {response_b}

Respond in the following JSON format:
[
    "answer": "A" or "B",
    "reasoning": "your explanation here"
]
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Label conversation response pairs with sampled LLM preference judgments")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--gpu_id", type=str, default="0", help="Comma-separated GPU ids; all are used for tensor parallelism")
    parser.add_argument("--run_id", type=int, required=True, help="Index of this run among total_runs parallel runs")
    parser.add_argument("--total_runs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=32, help="Judgments sampled per item")
    parser.add_argument("--global_offset", type=int, default=0, help="Line in input_file to start from")
    parser.add_argument("--num_items", type=int, default=None,
                        help="Items to label starting at global_offset (default: rest of the file)")
    parser.add_argument("--max_model_len", type=int, default=4096)
    return parser.parse_args()


def read_batch(input_file, start, end):
    conversations, responses_a, responses_b = [], [], []
    with open(input_file, "r") as f:
        for line_idx, line in enumerate(f):
            if line_idx < start:
                continue
            if line_idx >= end:
                break
            item = json.loads(line)
            conversations.append(item["conversation"])
            responses_a.append(item["response_A"])
            responses_b.append(item["response_B"])
    return conversations, responses_a, responses_b


def main():
    args = parse_args()
    print(f"Args: {args}")
    gpu_ids = set_visible_gpus(args.gpu_id)

    num_items = args.num_items
    if num_items is None:
        num_items = count_lines(args.input_file) - args.global_offset
    run_start, run_end = run_slice(args.run_id, args.total_runs, num_items)
    total_batches = (run_end - run_start + args.batch_size - 1) // args.batch_size

    output_path = os.path.join(
        args.output_dir,
        f"majority_votes_{args.num_samples}_synthetic_run{args.run_id}_offset{args.global_offset}.jsonl",
    )
    completed = load_completed_batches(output_path)
    remaining = [b for b in range(total_batches) if b not in completed]
    print(f"Run {args.run_id}: items [{run_start}, {run_end}), {len(remaining)}/{total_batches} batches remaining")
    if not remaining:
        return

    llm = LLM(model=args.model, tensor_parallel_size=len(gpu_ids), gpu_memory_utilization=0.85,
              max_num_seqs=args.batch_size, max_model_len=args.max_model_len)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sampling_params = SamplingParams(n=args.num_samples, min_tokens=10, max_tokens=1024, temperature=0.6,
                                     top_p=0.95, top_k=20, min_p=0, presence_penalty=1.5)

    for batch_idx in remaining:
        first = run_start + batch_idx * args.batch_size
        last = min(first + args.batch_size, run_end)
        conversations, responses_a, responses_b = read_batch(
            args.input_file, args.global_offset + first, args.global_offset + last)
        if not conversations:
            continue

        prompts = [
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": JUDGE_TEMPLATE.format(conversation=c, response_a=a, response_b=b)},
            ]
            for c, a, b in zip(conversations, responses_a, responses_b)
        ]
        input_tokens = [len(tokenizer.encode(p[1]["content"])) for p in prompts]
        outputs = llm.chat(prompts, sampling_params=sampling_params, use_tqdm=True,
                           chat_template_kwargs={"enable_thinking": True})
        judgments = parse_judgments(outputs)

        records = []
        for i, conversation in enumerate(conversations):
            records.append({
                "batch_idx": batch_idx,
                "global_conv_idx": args.global_offset + first + i,
                "conversation": conversation,
                "input_tokens": input_tokens[i],
                "num_samples": args.num_samples,
                "judgments": judgments[i],
            })
        append_records(output_path, records)
        print(f"Batch {batch_idx} done ({len(records)} items)")

    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
