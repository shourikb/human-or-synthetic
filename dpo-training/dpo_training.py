import json
import wandb
from unsloth import FastLanguageModel, is_bfloat16_supported, PatchDPOTrainer
from config import MODELS_DIR
from train_dataset import get_train_dataset
from config_training import PTRunConfig, LoraConfig, ModelLoadConfig, DatasetConfig, DPOTrainingConfig, dpo_batch_size, effective_batch_size
from utils import apply_chat_template_pt, chat_template_func
from trl import DPOTrainer, DPOConfig
import pprint
import os
from transformers import TrainerCallback
import math

PatchDPOTrainer()

import torch
import gc

GRID_SEARCH_DIR = "/coc/pskynet6/sbanerjee332/dpo-training"

class ProgressiveEvalCallback(TrainerCallback):
    def __init__(self, eval_steps, total_steps, total_eval_samples, output_dir):
        self.eval_steps = set(eval_steps)
        self.total_steps = total_steps
        self.total_eval_samples = total_eval_samples
        self.output_dir = output_dir
        self._full_eval_dataset = None
        self.trainer = None
    
    def on_init_end(self, args, state, control, **kwargs):
        # store trainer reference here if available
        self.trainer = kwargs.get("model", None)  # fallback

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.eval_steps:
            fraction = state.global_step / self.total_steps
            n_samples = max(1, int(self.total_eval_samples * fraction))
            print(f"step={state.global_step}, total_steps={self.total_steps}, fraction={fraction}, n_samples={n_samples}")

            self._full_eval_dataset = self.trainer.eval_dataset
            self.trainer.eval_dataset = self.trainer.eval_dataset.select(range(n_samples))

            control.should_evaluate = True
            control.should_save = True

        return control

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if self._full_eval_dataset is not None:
            self.trainer.eval_dataset = self._full_eval_dataset
            self._full_eval_dataset = None
        
        eval_loss = metrics.get('eval_loss', float('inf'))
        step = state.global_step
        save_path = f"{self.output_dir}/checkpoint-{step}"
        self.trainer.save_model(save_path)
        
        print(f"Step {step} | eval_loss: {eval_loss:.4f}")

def train_model_dpo(
        run_config: PTRunConfig = None,
        lora_config: LoraConfig = None,
        model_load_config: ModelLoadConfig = None,
        training_args: DPOTrainingConfig = None,
):

    train_batch_size = training_args.per_device_train_batch_size  # dpo_batch_size(run_config.dataset_config.train_size)
    gradient_accumulation_steps = training_args.gradient_accumulation_steps # effective_batch_size(run_config.dataset_config.train_size) // train_batch_size

    print(f"Per device train batch size: {train_batch_size}")

    dataset = get_train_dataset(run_config)
    
    if run_config.sft_run_config:
        model_path = f"{MODELS_DIR}/{run_config.sft_run_config.run_name}"
    else:
        model_path = run_config.model_name_hf

    print(f"Loading model from {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = model_load_config.max_seq_length,
        dtype = model_load_config.dtype,
        load_in_4bit = model_load_config.load_in_4bit,
        attn_implementation = "sdpa",
    )

    dataset = apply_chat_template_pt(tokenizer, dataset, run_config.model_name)

    print("=== SAMPLE DPO EXAMPLE AFTER TEMPLATING ===")
    print("PROMPT:", dataset["train"][0]["prompt"])
    print("CHOSEN:", dataset["train"][0]["chosen"])
    print("REJECTED:", dataset["train"][0]["rejected"])
    print("=== END SAMPLE ===")

    pprint.pprint(dataset["train"][0])
    pprint.pprint(dataset)

    if run_config.model_name == "qwen2-7B":
        lora_config.target_modules.extend(["embed_tokens", "lm_head"])

    print(f"Using LORA with config: {lora_config.target_modules}")
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_config.r,
        target_modules = lora_config.target_modules,
        lora_alpha = lora_config.lora_alpha, 
        lora_dropout = lora_config.lora_dropout,
        bias = lora_config.bias,
        use_gradient_checkpointing = lora_config.use_gradient_checkpointing,
        random_state = lora_config.random_state, 
        use_rslora = lora_config.use_rslora,
        loftq_config = lora_config.loftq_config,
    )

    print(f"Model loaded - {type(model)}")

    # Print memory usage
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    torch.cuda.empty_cache()
    gc.collect()

    total_steps = math.ceil(len(dataset["train"]) / (train_batch_size * gradient_accumulation_steps)) * training_args.num_train_epochs
    # eval_steps = [2145, 4290, 6435, 8580, 10725, 12870, 15015, 17160, 19305, 21450] # [18, 35, 52, 70, 87, 104, 122, 139, 156, 174]
    # callback = ProgressiveEvalCallback(eval_steps=eval_steps, total_steps=total_steps, total_eval_samples=len(dataset["test"]), output_dir=run_config.output_dir)
    trainer = DPOTrainer(
        model = model,
        ref_model = None,
        tokenizer = chat_template_func(tokenizer, run_config.model_name),
        beta = training_args.beta,
        train_dataset = dataset["train"],
        # eval_dataset = dataset["test"],
        max_length = model_load_config.max_seq_length,
        # callbacks = [callback],        
        args = DPOConfig(
            per_device_train_batch_size = train_batch_size,
            gradient_accumulation_steps = gradient_accumulation_steps,

            per_device_eval_batch_size = training_args.per_device_eval_batch_size,
            warmup_ratio = training_args.warmup_ratio,
            num_train_epochs = training_args.num_train_epochs,
            learning_rate = training_args.learning_rate,
            do_eval = training_args.do_eval,
            eval_strategy = training_args.eval_strategy,
            # eval_steps = training_args.eval_steps,
            optim = training_args.optim,
            weight_decay = training_args.weight_decay,
            lr_scheduler_type = training_args.lr_scheduler_type,
            report_to = training_args.report_to,
            logging_steps = training_args.logging_steps,

            output_dir = run_config.output_dir,
            save_strategy = "steps",
            save_steps = 1000,
            # load_best_model_at_end=True,
            # metric_for_best_model="eval_loss",

            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            seed = 42,
        )
    )
    # callback.trainer = trainer

    trainer_stats = trainer.train()

    model.save_pretrained_merged(run_config.output_dir, tokenizer, save_method = "merged_16bit")

    args = trainer.args.to_dict()
    with open(f"{run_config.output_dir}/training_config.json", "w") as f:
        json.dump(args, f, indent=4)   

    csv_filename = os.path.join(GRID_SEARCH_DIR, f"grid_search_results_{run_config.dataset_config.dataset_full_name}.csv")
    log_training_results(
        trainer=trainer,
        trainer_stats=trainer_stats,
        run_config=run_config,
        training_args=training_args,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        csv_filename=csv_filename
    ) 

def log_training_results(
    trainer,
    trainer_stats,
    run_config,
    training_args,
    train_batch_size,
    gradient_accumulation_steps,
    csv_filename="grid_search_results.csv"
):
    """
    Log training results to a JSON file and append to a CSV for grid search comparison.
    
    Args:
        trainer: The DPOTrainer instance
        trainer_stats: Training statistics returned by trainer.train()
        run_config: PTRunConfig object
        training_args: DPOTrainingConfig object
        train_batch_size: Per-device training batch size
        gradient_accumulation_steps: Gradient accumulation steps
        csv_filename: Name of the CSV file to append results to
    """
    import json
    import fcntl
    from pathlib import Path
    
    # Extract best eval loss from trainer state
    best_eval_loss = None
    if hasattr(trainer.state, 'log_history'):
        eval_losses = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]
        if eval_losses:
            best_eval_loss = min(eval_losses)
    
    # Create results summary
    results_summary = {
        "hyperparameters": {
            "learning_rate": training_args.learning_rate,
            "per_device_train_batch_size": train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": train_batch_size * gradient_accumulation_steps,
            "beta": training_args.beta,
            "num_train_epochs": training_args.num_train_epochs,
        },
        "metrics": {
            "best_eval_loss": best_eval_loss,
            "final_train_loss": trainer_stats.metrics.get('train_loss'),
            "train_runtime": trainer_stats.metrics.get('train_runtime'),
            "train_samples_per_second": trainer_stats.metrics.get('train_samples_per_second'),
        },
        "output_dir": run_config.output_dir,
        "run_name": getattr(run_config, 'run_name', None),
    }
    
    # Save individual run results to JSON
    with open(f"{run_config.output_dir}/results_summary.json", "w") as f:
        json.dump(results_summary, f, indent=4)
    
    # Append to global CSV with file locking
    csv_path = Path(csv_filename)
    
    with open(csv_path, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Acquire exclusive lock
        
        # Write header if file is empty/new
        if f.tell() == 0:
            f.write("run_name,output_dir,lr,batch_size,grad_accum,effective_bs,beta,epochs,"
                   "best_eval_loss,final_train_loss,train_runtime,samples_per_sec\n")
        
        # Append this run's results
        f.write(f"{results_summary['run_name']},"
                f"{results_summary['output_dir']},"
                f"{results_summary['hyperparameters']['learning_rate']},"
                f"{results_summary['hyperparameters']['per_device_train_batch_size']},"
                f"{results_summary['hyperparameters']['gradient_accumulation_steps']},"
                f"{results_summary['hyperparameters']['effective_batch_size']},"
                f"{results_summary['hyperparameters']['beta']},"
                f"{results_summary['hyperparameters']['num_train_epochs']},"
                f"{results_summary['metrics']['best_eval_loss']},"
                f"{results_summary['metrics']['final_train_loss']},"
                f"{results_summary['metrics']['train_runtime']},"
                f"{results_summary['metrics']['train_samples_per_second']}\n")
        
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Release lock
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best eval loss: {best_eval_loss}")
    print(f"Results saved to: {run_config.output_dir}/results_summary.json")
    print(f"Summary appended to: {csv_path.absolute()}")
    print(f"{'='*60}\n")
    
    return results_summary

if __name__ == "__main__":
    dataset_config = DatasetConfig(
        dataset = "anthropic",
        dataset_type = "pt",
        train_size = 141
    )

    print(dataset_config)

    lora_config = LoraConfig()
    model_load_config = ModelLoadConfig()
    training_args = DPOTrainingConfig()

    run_config = PTRunConfig(
        dataset_config = dataset_config,
        model_name_hf = "allenai/Llama-3.1-Tulu-3-8B-SFT",
        model_name = "tulu3-8B",
        do_training=True,
        do_inference=False,
        do_evaluation=False,
    )
    run = wandb.init(name=run_config.run_name, project="tuning", reinit=True)

    print(run_config)
    with run:
        train_model_dpo(
            run_config = run_config,
            lora_config = lora_config,
            model_load_config = model_load_config,
            training_args = training_args,
        )