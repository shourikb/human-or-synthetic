import csv
import fcntl
import json
import os

from unsloth import FastLanguageModel


def load_lora_model(model_path, model_load_config, lora_config):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=model_load_config.max_seq_length,
        dtype=model_load_config.dtype,
        load_in_4bit=model_load_config.load_in_4bit,
        attn_implementation="sdpa",
    )
    model = FastLanguageModel.get_peft_model(model, **lora_config.model_dump())
    return model, tokenizer


def save_run_results(trainer, trainer_stats, run_config, training_args, results_csv=None):
    with open(os.path.join(run_config.output_dir, "training_config.json"), "w") as f:
        json.dump(trainer.args.to_dict(), f, indent=4)

    summary = {
        "run_name": run_config.run_name,
        "output_dir": run_config.output_dir,
        "learning_rate": training_args.learning_rate,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "effective_batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
        "beta": getattr(training_args, "beta", None),
        "num_train_epochs": training_args.num_train_epochs,
        "train_loss": trainer_stats.metrics.get("train_loss"),
        "train_runtime": trainer_stats.metrics.get("train_runtime"),
        "train_samples_per_second": trainer_stats.metrics.get("train_samples_per_second"),
    }
    with open(os.path.join(run_config.output_dir, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    if results_csv:
        with open(results_csv, "a", newline="") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.seek(0, os.SEEK_END)
            writer = csv.DictWriter(f, fieldnames=list(summary))
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(summary)
        print(f"Appended run summary to {results_csv}")
