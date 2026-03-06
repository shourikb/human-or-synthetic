"""
Usage:
    python best-chckpt.py \
        --model_dir models/tulu3-8B_pt-anthropic-769_13532 \
        --dataset_dir datasets/pt-anthropic-769/test/data-00000-of-00001.arrow \
        --checkpoints 18 35 52 70 87 104 122 139 156 174 \
        --base_model allenai/Llama-3.1-Tulu-3-8B-SFT \
        --total_eval 76 \
        --output_file best_checkpoints.json
"""

import argparse
import json
import os
import time

import torch
import gc
from datasets import Dataset, load_from_disk
from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported
from trl import DPOConfig, DPOTrainer
from unsloth.chat_templates import get_chat_template
from utils import chat_template_func

PatchDPOTrainer()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--checkpoints", type=int, nargs="+", required=True)
    parser.add_argument("--total_eval", type=int, default=76)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_file", type=str, default="best_checkpoints.json")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=1024)
    return parser.parse_args()


def get_checkpoint_path(model_dir, step):
    path = os.path.join(model_dir, f"checkpoint-{step}")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path

def evaluate_checkpoint(
    checkpoint_path,
    ref_model,
    dataset,
    n_samples,
    batch_size,
    beta,
    max_length,
):

    print(f"\nLoading policy model from {checkpoint_path}")

    # Load policy (Unsloth style)
    policy_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = checkpoint_path,
        max_seq_length = max_length,
        dtype = None,
        load_in_4bit = False,
    )   

    # Inference mode (important for eval)
    FastLanguageModel.for_inference(policy_model)
    
    policy_model.eval()
    ref_model.eval()

    subset = dataset.select(range(n_samples))

    trainer = DPOTrainer(
        model = policy_model,
        ref_model = ref_model,
        tokenizer = chat_template_func(tokenizer),
        beta = beta,
        train_dataset = subset,   # unused but required
        eval_dataset = subset,
        max_length = max_length,
        args = DPOConfig(
            per_device_eval_batch_size = batch_size,
            output_dir = "/tmp/dpo_eval_tmp",
            report_to = "none",
            remove_unused_columns = False,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
        ),
    )

    print("Running evaluation...")
    metrics = trainer.evaluate()
    loss = metrics["eval_loss"]

    print(f"Eval loss: {loss:.6f}")

    # Cleanup like your training script
    del policy_model
    del trainer
    torch.cuda.empty_cache()
    gc.collect()

    return loss


def main():
    args = parse_args()

    checkpoint_paths = []
    for step in args.checkpoints:
        path = get_checkpoint_path(args.model_dir, step)
        checkpoint_paths.append(path)
        print(f"  Found: {path}")

    n_tiers = len(args.checkpoints)
    total_steps = max(args.checkpoints)

    eval_counts = [
        max(1, int(args.total_eval * step / total_steps))
        for step in args.checkpoints
    ]
    print(f"\nEval sample counts per tier: {eval_counts}")

    print(f"\nLoading dataset from {args.dataset_dir} ...")
    if args.dataset_dir.endswith(".arrow"):
        dataset = Dataset.from_file(args.dataset_dir)
    else:
        dataset = load_from_disk(args.dataset_dir)
    print(f"Dataset size: {len(dataset)} samples")

    if len(dataset) < args.total_eval:
        raise ValueError(f"Dataset has {len(dataset)} samples but --total_eval={args.total_eval}.")

    results = {}
    total_evals = sum(range(1, n_tiers + 1))
    evals_done = 0
    run_start = time.time()

    print(f"Loading reference model from {args.base_model} ...")
    ref_model, _ = FastLanguageModel.from_pretrained(
        model_name = args.base_model,
        max_seq_length = args.max_length,
        dtype = None,
        load_in_4bit = False,
    )
    FastLanguageModel.for_inference(ref_model)

    for tier_idx in range(n_tiers):
        tier_num = tier_idx + 1
        n_eval   = eval_counts[tier_idx]
        budget   = (tier_idx + 1) * 100

        print(f"\n{'='*60}")
        print(f"Tier ${budget} ({tier_num}/{n_tiers}) | eval samples: {n_eval} | "
              f"checkpoints: {args.checkpoints[:tier_num]}")
        print(f"Overall progress: {evals_done}/{total_evals} evaluations done")
        print(f"{'='*60}")

        tier_results = {}

        for ckpt_idx in range(tier_num):
            step       = args.checkpoints[ckpt_idx]
            ckpt_path  = checkpoint_paths[ckpt_idx]
            ckpt_start = time.time()

            print(f"\n  [{ckpt_idx+1}/{tier_num}] checkpoint-{step} on {n_eval} samples")
            loss = evaluate_checkpoint(
                checkpoint_path = ckpt_path,
                ref_model = ref_model,
                dataset = dataset,
                n_samples = n_eval,
                batch_size = args.batch_size,
                beta = args.beta,
                max_length = args.max_length,
            )
            elapsed    = time.time() - ckpt_start
            evals_done += 1
            tier_results[step] = loss
            print(f"  checkpoint-{step}: loss={loss:.6f}  ({elapsed:.0f}s)")

            avg_per_eval = (time.time() - run_start) / evals_done
            remaining    = total_evals - evals_done
            eta_secs     = avg_per_eval * remaining
            h, rem       = divmod(int(eta_secs), 3600)
            m, s         = divmod(rem, 60)
            print(f"  ETA: {h}h {m}m {s}s  ({evals_done}/{total_evals} done)")

        best_step = min(tier_results, key=tier_results.get)
        best_loss = tier_results[best_step]
        best_path = get_checkpoint_path(args.model_dir, best_step)

        print(f"\n  ✓ Best for ${budget}: checkpoint-{best_step} (loss={best_loss:.6f})")

        results[f"${budget}"] = {
            "tier": tier_num,
            "n_eval_samples": n_eval,
            "best_checkpoint_step": best_step,
            "best_checkpoint_path": best_path,
            "best_loss": best_loss,
            "all_losses": {f"checkpoint-{k}": v for k, v in tier_results.items()},
        }

        # Save incrementally so progress isn't lost on crash
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)

    total_time = time.time() - run_start
    print(f"\n\nCompleted in {total_time/3600:.1f}h")
    print(f"Results saved to {args.output_file}")
    print("\n" + "="*70)
    print(f"{'Budget':<10} {'Best Checkpoint':<20} {'Loss':<14} {'N Eval'}")
    print("-"*70)
    for budget_key, info in results.items():
        print(f"{budget_key:<10} checkpoint-{info['best_checkpoint_step']:<14} "
              f"{info['best_loss']:<14.6f} {info['n_eval_samples']}")
    print("="*70)


if __name__ == "__main__":
    main()