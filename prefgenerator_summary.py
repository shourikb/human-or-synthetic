"""
Generates majority-vote preference judgments for summarization comparisons.

Input format (jsonl), one object per line:
    {"info": "...", "post": "...", "summary_A": "...", "summary_B": "..."}

Reads from INPUT_FILE (hardcoded below).

Usage:
    python generate_summary_judgments.py \
        --run_id 0 \
        --total_runs 6 \
        --batch_size 64 \
        --num_samples 32 \
        --gpu_id 0
"""
from vllm import LLM
from vllm import SamplingParams
from transformers import AutoTokenizer
import json
import re
import argparse
import os
import sys
import tempfile
import shutil


TOTAL_CONVERSATIONS = 110448
INPUT_FILE = "finalposts.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(description='Run summarization preference judgment in batches')
    parser.add_argument('--gpu_id', type=str, default="0", help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--run_id', type=int, required=True, help='Which of the total_runs parallel runs this is (0-indexed)')
    parser.add_argument('--total_runs', type=int, default=6, help='Total number of parallel runs')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of items per batch')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory for output files')
    parser.add_argument('--num_samples', type=int, default=32, help='Number of preference judgments per post/summary pair')
    parser.add_argument('--global_offset', type=int, default=0, help='Global line index to start from')
    return parser.parse_args()


def count_lines(path):
    count = 0
    with open(path, 'r') as f:
        for _ in f:
            count += 1
    return count


def get_output_path(output_dir, run_id, num_samples, global_offset):
    return os.path.join(output_dir, f"majority_votes_{num_samples}_summarization_run{run_id}_offset{global_offset}.jsonl")


def load_completed_batches(output_path):
    """Return a set of batch_idx values already written to the output file."""
    completed = set()
    if not os.path.exists(output_path):
        return completed
    with open(output_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                if 'batch_idx' in item:
                    completed.add(item['batch_idx'])
            except json.JSONDecodeError:
                continue
    print(f"[Checkpoint] Found {len(completed)} already-completed batches: {sorted(completed)}")
    return completed


def atomic_append_batch(output_path, records):
    """Append a list of JSON records to output_path atomically using a temp file + copy trick.

    Since each run_id writes to its own output file, there's no cross-process conflict.
    """
    dir_name = os.path.dirname(output_path) or '.'
    with tempfile.NamedTemporaryFile('w', dir=dir_name, delete=False, suffix='.tmp') as tmp:
        for record in records:
            tmp.write(json.dumps(record) + "\n")
        tmp_path = tmp.name

    with open(output_path, 'a') as out_f, open(tmp_path, 'r') as tmp_f:
        shutil.copyfileobj(tmp_f, out_f)
        out_f.flush()
        os.fsync(out_f.fileno())

    os.remove(tmp_path)


def read_conversations_for_run(input_file, run_id, total_runs, total_lines, batch_size, batch_idx, global_offset):
    """Read a single batch worth of post/summary_A/summary_B records for this run_id.

    Contiguous split: run 0 gets global indices [0, total/N), run 1 gets [total/N, 2*total/N), etc.
    """
    lines_per_run = total_lines // total_runs
    global_start = run_id * lines_per_run
    if run_id == total_runs - 1:
        global_end = total_lines
    else:
        global_end = global_start + lines_per_run

    local_start = batch_idx * batch_size
    local_end = min(local_start + batch_size, global_end - global_start)

    if local_start >= (global_end - global_start):
        return None, None, None, None  # No more work

    file_start = global_offset + global_start + local_start
    file_end = global_offset + global_start + local_end

    posts, summaries_a, summaries_b, infos = [], [], [], []

    with open(input_file, 'r') as f:
        for line_idx, line in enumerate(f):
            if line_idx < file_start:
                continue
            if line_idx >= file_end:
                break
            item = json.loads(line)
            posts.append(item['post'])
            summaries_a.append(item['summary_A'])
            summaries_b.append(item['summary_B'])
            infos.append(item.get('info'))

    return posts, summaries_a, summaries_b, infos


def build_prompts(posts, summaries_a, summaries_b):
    prompts = []
    for i in range(len(posts)):
        prompt = [
            {
                "role": "system",
                "content": "You are comparing the quality of two summaries of a given text."
            },
            {
                "role": "user",
                "content": f"""Which of the following summaries does a better job of summarizing the most important points in the given forum post, without including unimportant or irrelevant details? A good summary is both precise and concise.

Judge each summary along these dimensions, the way a careful reader would:
- Coverage: how much important information from the original post is covered.
- Accuracy: to what degree the statements in the summary are actually stated or implied in the post (a summary that is coherent but inaccurate should score lower on accuracy).
- Coherence: how easy the summary is to read on its own, independent of the post.

Place less weight on polished sentence fluidity for its own sake, and less weight on whether the summary captures the poster's tone or intent beyond the literal content — focus primarily on whether it faithfully and efficiently conveys the important information in the post.

Post: {posts[i]}

Summaries:
Summary A: {summaries_a[i]}
Summary B: {summaries_b[i]}

First, weigh the two summaries against each other along coverage, accuracy, and coherence. Then decide which is better overall.

Respond in the following JSON format:
[
    "answer": "A" or "B",
    "reasoning": "your explanation here"
]
""",
            },
        ]
        prompts.append(prompt)
    return prompts


def process_outputs(outputs, num_samples):
    all_texts = []
    all_tokens = []
    for output in outputs:
        texts = [o.text for o in output.outputs]
        tokens = [len(o.token_ids) for o in output.outputs]
        all_texts.append(texts)
        all_tokens.append(tokens)
    return all_texts, all_tokens


def parse_judgments(all_texts, all_tokens):
    all_judgments = []
    for i, texts in enumerate(all_texts):
        judgments = []
        for j, output_text in enumerate(texts):
            think_match = re.search(r"<think>\s*(.*?)\s*</think>", output_text, re.DOTALL)
            thinking = think_match.group(1).strip() if think_match else None
            answer_match = re.search(r'"answer"\s*:\s*"([^"]+)"', output_text)
            reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', output_text)
            judgments.append({
                "sample_id": j,
                "answer": answer_match.group(1).strip() if answer_match else None,
                "reasoning": reasoning_match.group(1).strip() if reasoning_match else None,
                "thinking": thinking,
                "output_tokens": all_tokens[i][j],
            })
        all_judgments.append(judgments)
    return all_judgments


def main():
    print("=" * 80)
    print("DIAGNOSTIC INFO:")
    print("CWD:", os.getcwd())
    print("Python executable:", sys.executable)
    print("Script location:", os.path.abspath(__file__))
    print("=" * 80)

    args = parse_args()
    print("Args:", args)

    gpu_ids = [x.strip() for x in args.gpu_id.split(',')]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    print(f"Using GPUs: {gpu_ids}")

    run_id = args.run_id
    total_runs = args.total_runs
    batch_size = args.batch_size
    num_samples = args.num_samples
    global_offset = args.global_offset
    input_file = INPUT_FILE

    print(f"Input file: {input_file} ({TOTAL_CONVERSATIONS} lines, hardcoded).")

    output_path = get_output_path(args.output_dir, run_id, num_samples, global_offset)
    print(f"Output file: {output_path}")

    # --- Checkpointing: find already-completed batches ---
    completed_batches = load_completed_batches(output_path)

    # Compute total batches for this run
    lines_per_run = TOTAL_CONVERSATIONS // total_runs
    if run_id == total_runs - 1:
        lines_this_run = TOTAL_CONVERSATIONS - run_id * lines_per_run
    else:
        lines_this_run = lines_per_run
    total_batches = (lines_this_run + batch_size - 1) // batch_size

    remaining_batches = [b for b in range(total_batches) if b not in completed_batches]
    print(f"Run {run_id}: {lines_this_run} lines, {total_batches} total batches, "
          f"{len(remaining_batches)} remaining.")

    if not remaining_batches:
        print("All batches already completed. Exiting.")
        return

    # --- Load model once ---
    model_name = "Qwen/Qwen3-32B"
    llm = LLM(
        model=model_name,
        tensor_parallel_size=len(gpu_ids),
        gpu_memory_utilization=0.85,
        max_num_seqs=batch_size,
        max_model_len=8192,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sampling_params = SamplingParams(
        n=num_samples,
        min_tokens=10,
        max_tokens=1024,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0,
        presence_penalty=1.5,
    )

    total_processed = 0
    for batch_idx in remaining_batches:
        print(f"\n--- Batch {batch_idx} / {total_batches - 1} ---")

        posts, summaries_a, summaries_b, infos = read_conversations_for_run(
            input_file, run_id, total_runs, TOTAL_CONVERSATIONS, batch_size, batch_idx, global_offset
        )

        if not posts:
            print(f"No data for batch {batch_idx}, skipping.")
            continue

        prompts = build_prompts(posts, summaries_a, summaries_b)
        input_token_counts = [len(tokenizer.encode(p[1]['content'])) for p in prompts]

        print(f"  Running inference on {len(prompts)} posts x {num_samples} samples...")
        outputs = llm.chat(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=True,
            chat_template_kwargs={"enable_thinking": True},
        )

        all_texts, all_tokens = process_outputs(outputs, num_samples)
        all_judgments = parse_judgments(all_texts, all_tokens)

        # Build records to write
        records = []
        for i in range(len(posts)):
            record = {
                "batch_idx": batch_idx,
                "global_conv_idx": global_offset + run_id * (TOTAL_CONVERSATIONS // total_runs) + batch_idx * batch_size + i,
                "post": posts[i],
                "summary_A": summaries_a[i],
                "summary_B": summaries_b[i],
                "info": infos[i],
                "input_tokens": input_token_counts[i],
                "num_samples": num_samples,
                "judgments": all_judgments[i],
            }
            records.append(record)

        atomic_append_batch(output_path, records)

        flat_tokens = [t for convo in all_tokens for t in convo]
        print(f"  Batch {batch_idx} done. Avg input tokens: {sum(input_token_counts)/len(input_token_counts):.0f}, "
              f"Avg output tokens: {sum(flat_tokens)/len(flat_tokens):.0f}")
        total_processed += len(posts)

    print(f"\nRun {run_id} complete. Total posts processed this session: {total_processed}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()