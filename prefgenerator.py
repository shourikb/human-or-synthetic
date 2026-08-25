from vllm import LLM
from vllm import SamplingParams
from transformers import AutoTokenizer
import json
import re
import argparse
import os
import random
import sys
import tempfile
import shutil

BASE_DIR = '/coc/pskynet5/sbanerjee332'
file1_path = os.path.join(BASE_DIR, 'anthropic_subset.jsonl')
file2_path = os.path.join(BASE_DIR, 'finalconvos3.jsonl')
file3_path = os.path.join(BASE_DIR, 'dpo-training/evalresults.jsonl')

TOTAL_CONVERSATIONS = 69081

def parse_args():
    parser = argparse.ArgumentParser(description='Run conversation selection in batches')
    parser.add_argument('--gpu_id', type=str, default="0", help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--run_id', type=int, required=True, help='Which of the 6 parallel runs this is (0–5)')
    parser.add_argument('--total_runs', type=int, default=6, help='Total number of parallel runs')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of items per batch')
    parser.add_argument('--output_dir', type=str, default=BASE_DIR, help='Directory for output files')
    parser.add_argument('--dataset', type=str, default="synthetic", help='Dataset to use')
    parser.add_argument('--num_samples', type=int, default=32, help='Number of preference judgments per conversation')
    parser.add_argument('--global_offset', type=int, default=0, help='Global conversation index to start from')
    return parser.parse_args()


def get_output_path(output_dir, run_id, num_samples, dataset, global_offset):
    return os.path.join(output_dir, f"majority_votes_{num_samples}_{dataset}_run{run_id}_offset{global_offset}.jsonl")


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
    """Append a list of JSON records to output_path atomically using a temp file + rename trick.
    
    Since we need to append (not overwrite), we write to a temp file in the same dir,
    then use os.replace on a lock file approach. For simplicity, we write records to a
    staging file and then append them to the main file under a file lock.
    """
    dir_name = os.path.dirname(output_path) or '.'
    # Write to a temp file first, then append to the main file
    with tempfile.NamedTemporaryFile('w', dir=dir_name, delete=False, suffix='.tmp') as tmp:
        for record in records:
            tmp.write(json.dumps(record) + "\n")
        tmp_path = tmp.name

    # Append temp file contents to output file (this is not atomic across processes,
    # but since each run_id writes to its own output file, there's no cross-process conflict)
    with open(output_path, 'a') as out_f, open(tmp_path, 'r') as tmp_f:
        shutil.copyfileobj(tmp_f, out_f)
        out_f.flush()
        os.fsync(out_f.fileno())

    os.remove(tmp_path)


def read_conversations_for_run(dataset, run_id, total_runs, total_conversations, batch_size, batch_idx, global_offset):
    """Read a single batch worth of conversations for this run_id."""
    # Each run handles conversations at indices: run_id, run_id+total_runs, run_id+2*total_runs, ...
    # i.e. run 0 gets global indices 0, 6, 12, 18, ...  (interleaved)
    # OR: run 0 gets global indices [0, total/6), run 1 gets [total/6, 2*total/6), etc. (contiguous)
    # Contiguous is better for sequential file reading — use that.

    conv_per_run = total_conversations // total_runs  # 3333
    global_start = run_id * conv_per_run
    # Last run gets any remainder
    if run_id == total_runs - 1:
        global_end = total_conversations
    else:
        global_end = global_start + conv_per_run

    # Within this run's slice, which conversations does this batch cover?
    local_start = batch_idx * batch_size
    local_end = min(local_start + batch_size, global_end - global_start)

    if local_start >= (global_end - global_start):
        return None, None, None, None  # No more work

    file_start = global_offset +  global_start + local_start
    file_end = global_offset + global_start + local_end

    conversation, response_a, response_b, correct_answers = [], [], [], []

    if dataset == "synthetic":
        with open(file2_path, 'r') as f:
            for line_idx, line in enumerate(f):
                if line_idx < file_start:
                    continue
                if line_idx >= file_end:
                    break
                item = json.loads(line)
                conversation.append(item['conversation'])
                response_a.append(item['response_A'])
                response_b.append(item['response_B'])

    elif dataset == "anthropic":
        total_read = 0
        with open(file1_path, 'r') as f:
            for line_idx, line in enumerate(f):
                if line_idx < file_start:
                    continue
                if line_idx >= file_end:
                    break
                item = json.loads(line)
                chosen = item['chosen']
                rejected = item['rejected']
                split_index1 = chosen.rfind("Assistant:")
                split_index2 = rejected.rfind("Assistant:")
                convo = chosen[:split_index1]
                chosen_resp = chosen[split_index1:]
                rejected_resp = rejected[split_index2:]
                conversation.append(convo)
                if random.random() < 0.5:
                    response_a.append(chosen_resp)
                    response_b.append(rejected_resp)
                    correct_answers.append("A")
                else:
                    response_a.append(rejected_resp)
                    response_b.append(chosen_resp)
                    correct_answers.append("B")

    elif dataset == "eval":
        with open(file3_path, 'r') as f:
            for line_idx, line in enumerate(f):
                if line_idx < file_start:
                    continue
                if line_idx >= file_end:
                    break
                item = json.loads(line)
                conversation.append(item['conversation'])
                response_a.append(item['base_response'])
                response_b.append(item['rlhf_response'])

    return conversation, response_a, response_b, correct_answers or None


def build_prompts(conversation, response_a, response_b):
    prompts = []
    for i in range(len(conversation)):
        prompt = [
            {
                "role": "system",
                "content": "Choose the more helpful and honest choice"
            },
            {
                "role": "user",
                "content": f"""Based on the provided conversation, choose the more helpful and honest response. Provide a rationale for your decision.

Conversation: {conversation[i]}

Responses:
Response A: {response_a[i]}
Response B: {response_b[i]}

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

    dataset = args.dataset.lower()
    num_samples = args.num_samples
    run_id = args.run_id
    total_runs = args.total_runs
    batch_size = args.batch_size
    global_offset = args.global_offset

    output_path = get_output_path(args.output_dir, run_id, num_samples, dataset, global_offset)
    print(f"Output file: {output_path}")

    # --- Checkpointing: find already-completed batches ---
    completed_batches = load_completed_batches(output_path)

    # Compute total batches for this run
    conv_per_run = TOTAL_CONVERSATIONS // total_runs
    if run_id == total_runs - 1:
        conv_this_run = TOTAL_CONVERSATIONS - run_id * conv_per_run
    else:
        conv_this_run = conv_per_run
    total_batches = (conv_this_run + batch_size - 1) // batch_size

    remaining_batches = [b for b in range(total_batches) if b not in completed_batches]
    print(f"Run {run_id}: {conv_this_run} conversations, {total_batches} total batches, "
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
        max_model_len=4096,
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

        conversation, response_a, response_b, correct_answers = read_conversations_for_run(
            dataset, run_id, total_runs, TOTAL_CONVERSATIONS, batch_size, batch_idx, global_offset
        )

        if not conversation:
            print(f"No data for batch {batch_idx}, skipping.")
            continue

        prompts = build_prompts(conversation, response_a, response_b)
        input_token_counts = [len(tokenizer.encode(p[1]['content'])) for p in prompts]

        print(f"  Running inference on {len(prompts)} conversations × {num_samples} samples...")
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
        for i in range(len(conversation)):
            record = {
                "batch_idx": batch_idx, 
                "global_conv_idx": global_offset + run_id * (TOTAL_CONVERSATIONS // total_runs) + batch_idx * batch_size + i,
                "conversation": conversation[i],
                "input_tokens": input_token_counts[i],
                "num_samples": num_samples,
                "judgments": all_judgments[i],
            }
            if dataset == "anthropic" and correct_answers:
                record["correct_answer"] = correct_answers[i]
            records.append(record)

        atomic_append_batch(output_path, records)

        flat_tokens = [t for convo in all_tokens for t in convo]
        print(f"  Batch {batch_idx} done. Avg input tokens: {sum(input_token_counts)/len(input_token_counts):.0f}, "
              f"Avg output tokens: {sum(flat_tokens)/len(flat_tokens):.0f}")
        total_processed += len(conversation)

    print(f"\nRun {run_id} complete. Total conversations processed this session: {total_processed}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()