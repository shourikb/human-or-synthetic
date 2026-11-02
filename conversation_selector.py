# DEPRECATED - prefgnerator.py is the new version of this with more features and better structure

from vllm import LLM
from vllm import SamplingParams
from transformers import AutoTokenizer
import json
import re
import argparse
import os
import random
import sys

BASE_DIR = '/coc/pskynet6/sbanerjee332'
file1_path = os.path.join(BASE_DIR, 'anthropic_subset.jsonl')
file2_path = os.path.join(BASE_DIR, 'finalconvos.jsonl')
file3_path = os.path.join(BASE_DIR, 'dpo-training/evalresults.jsonl')


def parse_args():
    parser = argparse.ArgumentParser(description='Run conversation selection in batches')
    parser.add_argument('--gpu_id', type=str, default="0", help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--batch_start', type=int, default=0, help='Starting batch index')
    parser.add_argument('--batch_count', type=int, default=1, help='Number of batches to process')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of items per batch')
    parser.add_argument('--output_suffix', type=str, default="", help='Suffix for output file')
    parser.add_argument('--dataset', type=str, default="anthropic", help='Dataset to use')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of preference judgments per conversation')
    return parser.parse_args()

def main():
    print("=" * 80)
    print("DIAGNOSTIC INFO:")
    print("CWD:", os.getcwd())
    print("Python executable:", sys.executable)
    print("Script location:", os.path.abspath(__file__))
    print("=" * 80)
    
    args = parse_args()
    print("Args:", args)
    print("Dataset:", args.dataset)
    
    # Print all three file paths and their existence
    print(f"file1_path: {file1_path}, exists: {os.path.exists(file1_path)}")
    print(f"file2_path: {file2_path}, exists: {os.path.exists(file2_path)}")
    print(f"file3_path: {file3_path}, exists: {os.path.exists(file3_path)}")
    
    if os.path.exists(file2_path):
        print("file2 size (bytes):", os.path.getsize(file2_path))

    args = parse_args()
    gpu_ids = [x.strip() for x in args.gpu_id.split(',')]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    print(f"Using GPU ID: {args.gpu_id}")

    dataset = args.dataset.lower()
    num_samples = args.num_samples

    batch_start = args.batch_start
    batch_end = batch_start + args.batch_count
    batch_size = args.batch_size

    
    if dataset == "eval":
        model_name = "Qwen/Qwen3-32B"  # Use a larger model for eval
        llm = LLM(model=model_name, tensor_parallel_size=2, gpu_memory_utilization=0.85, max_num_seqs=64)
    else:
        model_name = "Qwen/Qwen3-32B" # "Qwen/Qwen3-14B" # "Qwen/Qwen2.5-14B-Instruct"  "Qwen/Qwen3-30B-A3B-Instruct-2507"  "Unbabel/M-Prometheus-14B"
        llm = LLM(model=model_name, tensor_parallel_size=4, gpu_memory_utilization=0.85, max_num_seqs=batch_size, max_model_len=4096)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
   
    total = 0
    for batch_idx in range(batch_start, batch_end):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        print(f"\nProcessing batch {batch_idx} (lines {start_idx}–{end_idx})")

        conversation = []
        response_a = []
        response_b = []
        correct_answers = []

        total_read = 0
        if dataset == "anthropic":
            with open(file1_path, 'r') as f:
                for line in f:
                    if total_read >= 2000:
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
                        # chosen → A
                        response_a.append(chosen_resp)
                        response_b.append(rejected_resp)
                        correct_answer = "A"
                    else:
                        # chosen → B
                        response_a.append(rejected_resp)
                        response_b.append(chosen_resp)
                        correct_answer = "B"

                    # store label
                    correct_answers.append(correct_answer)
                    total_read += 1
        elif dataset == "synthetic":
            with open(file2_path, 'r') as f:
                for line_idx, line in enumerate(f):
                    if line_idx < start_idx:
                        continue
                    if line_idx >= end_idx:
                        break
                    
                    item = json.loads(line)
                    conversation.append(item['conversation'])
                    response_a.append(item['response_A'])
                    response_b.append(item['response_B'])
                    total_read += 1
        elif dataset == "eval":
            # handle eval
            with open(file3_path, 'r') as f:
                for line_idx, line in enumerate(f):
                    if line_idx < start_idx:
                        continue
                    if line_idx >= end_idx:
                        break
                    item = json.loads(line)
                    chosen = item['conversation']
                    response_A = item['base_response']
                    response_B = item['rlhf_response']
                    conversation.append(chosen)
                    response_a.append(response_A)
                    response_b.append(response_B)
                    total_read += 1

        print(f"Total conversations read for this batch: {total_read}") 

        input_tokens = []
        normal_prompts = []
        for i in range(len(conversation)):
            prompt_normal = [
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
            normal_prompts.append(prompt_normal)

            input_tokens.append(len(tokenizer.encode(prompt_normal[1]['content'])))

        all_normal_outputs = [] # outputs[i][j] = jth judgement for ith conversation
        all_normal_tokens = []

        sampling_params = SamplingParams(n=num_samples, min_tokens=10, max_tokens=1024, temperature=0.6, top_p=0.95, top_k=20, min_p=0, presence_penalty=1.5)
        print("Running normal prompts...")
        normal_outputs = llm.chat(normal_prompts, sampling_params=sampling_params, use_tqdm=True, chat_template_kwargs={"enable_thinking": True})

        for i in range(len(normal_outputs)):
            outputs_i = normal_outputs[i].outputs  # length = num_samples

            texts = []
            tokens = []
            for out in outputs_i:
                texts.append(out.text)
                tokens.append(len(out.token_ids))

            all_normal_outputs.append(texts)
            all_normal_tokens.append(tokens)

        output_file = f"majority_votes_{num_samples}_synthetic.jsonl"
        with open(output_file, "a", encoding="utf-8") as f:
            for i in range(len(all_normal_outputs)):

                judgments = []

                for j, output_text in enumerate(all_normal_outputs[i]):
                    think_match = re.search(r"<think>\s*(.*?)\s*</think>", output_text, re.DOTALL)
                    thinking = think_match.group(1).strip() if think_match else None

                    answer_match = re.search(r'"answer"\s*:\s*"([^"]+)"', output_text)
                    reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', output_text)

                    answer = answer_match.group(1).strip() if answer_match else None
                    reasoning = reasoning_match.group(1).strip() if reasoning_match else None

                    judgments.append({
                        "sample_id": j,
                        "answer": answer,
                        "reasoning": reasoning,
                        "thinking": thinking,
                        "output_tokens": all_normal_tokens[i][j],
                    })

                    total += 1

                json_entry = {
                    "conversation": conversation[i],
                    "input_tokens": input_tokens[i],
                    "num_samples": num_samples,
                    "judgments": judgments,
                }

                if dataset == "anthropic":
                    json_entry["correct_answer"] = correct_answers[i]

                f.write(json.dumps(json_entry) + "\n")


        print("Average input tokens ", sum(input_tokens) / len(input_tokens) if input_tokens else 0)
        flat_tokens = [t for convo in all_normal_tokens for t in convo]
        print("Average output tokens ", sum(flat_tokens) / len(flat_tokens) if flat_tokens else 0)
        print("Total input tokens ", sum(input_tokens) if input_tokens else 0)
        print("Total output tokens ", sum(flat_tokens) if flat_tokens else 0)

if __name__ == "__main__":
    main()