from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
import torch
import os
from datetime import datetime
import wandb
import argparse

torch.set_default_dtype(torch.bfloat16)

def parse_args():
    parser = argparse.ArgumentParser(description="DPO Training Script")

    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate for the optimizer.")

    parser.add_argument("--per_device_batch_size", type=int, default=1,
                        help="Batch size per device during training.")

    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--output_dir", type=str, default="dpo_model_output",
                        help="Directory to save the trained model and checkpoints.")
    
    # Add more hyperparameters as needed
    return parser.parse_args()

def load_model_and_tokenizer(model_name="allenai/Llama-3.1-Tulu-3-8B-SFT", custom_adapter=None):
    bfloat16_supported = torch.cuda.is_bf16_supported()
    torch_dtype = torch.bfloat16 if bfloat16_supported else torch.float16
    
    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        # attn_implementation="flash_attention_2",
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    model.config.use_cache = False
    
    if custom_adapter:
        print(f"Loading and applying local adapter: {custom_adapter}")
        model = PeftModel.from_pretrained(model, custom_adapter, is_trainable=True)
    else:
        print("No custom adapter provided, initializing new LoRA config.")
        peft_config = LoraConfig(
            r=32,
            lora_alpha=32,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        # need this line to enable gradients for gradient checkpointing
        # model.enable_input_require_grads()
        
        model = get_peft_model(model, peft_config)

    print("Final LoRA model info:")
    model.print_trainable_parameters()

    return model, tokenizer

def train_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    checkpoint_dir,
    logging_dir,
    save_dir,
    project_name,
    batch_size=1,
    gradient_accumulation_steps=4,
    max_seq_length=16384,
    num_epochs=1,
    learning_rate=2e-4,
    resume_checkpoint_path=None,
):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{current_time}_{project_name}_{max_seq_length}"

    if os.environ.get("RANK", "0") == "0":
        wandb.init(project="dpo_gridsearch", name=run_name)

    EVAL_STEPS = 1
    SAVE_STEPS = 100
    NUM_WORKERS = 16

    is_bf16 = torch.cuda.is_bf16_supported()

    if eval_dataset is None:
        eval_strategy = "no"
        eval_steps = None
    else:
        eval_strategy = "steps"
        eval_steps = EVAL_STEPS

    training_args = DPOConfig(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=5,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        fp16=not is_bf16,
        bf16=is_bf16,
        logging_steps=1,
        output_dir=checkpoint_dir,
        logging_dir=logging_dir,
        optim="paged_adamw_8bit",
        # seed=3407,
        report_to="wandb",
        run_name=run_name,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=10,
        dataloader_num_workers=NUM_WORKERS,
        dataloader_prefetch_factor=2,
        remove_unused_columns=False,
        max_length=max_seq_length,
        ddp_timeout=10800,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
    )

    # model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "args": training_args,
        "processing_class": tokenizer,
    }
    
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    trainer = DPOTrainer(**trainer_kwargs)
    trainer.train(resume_from_checkpoint=resume_checkpoint_path)

    if trainer.is_world_process_zero():
        adapter_save_dir = f"{save_dir}_adapters"
        print(f"Saving adapter to {adapter_save_dir}...")
        
        # This is fast, just saves the adapter weights
        trainer.save_model(adapter_save_dir) 
        tokenizer.save_pretrained(adapter_save_dir)
        print(f"Adapter saved to {adapter_save_dir}")

        # This is also fast, as the model is not sharded
        print("Merging model...")
        merged = model.merge_and_unload()
        merged.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Merged model saved to {save_dir}")

    if os.environ.get("RANK", "0") == "0":
        wandb.finish()

    return model, tokenizer


def main():
    args = parse_args()

    TRAIN_DATASET = "../anthropic_subset.jsonl"
    CHECKPOINT_PATH = None # "tulu-3-8b-anthropic-dpo-10k/checkpoints/checkpoint-4900"
    TRAIN_DATASET_SIZE = 141
    EVAL_PERCENT = 0.03
    eval_size  = 15 if TRAIN_DATASET_SIZE < 1000 else int(TRAIN_DATASET_SIZE * EVAL_PERCENT)

    base_save_dir = args.output_dir

    model_name = "allenai/Llama-3.1-Tulu-3-8B-SFT"

    model, tokenizer = load_model_and_tokenizer(model_name=model_name, custom_adapter=None)


    raw = load_dataset("json", data_files={"train": TRAIN_DATASET})
    raw["train"] = raw["train"].select(range(TRAIN_DATASET_SIZE))
    raw_split = raw["train"].train_test_split(
        test_size=eval_size,
        shuffle=True,
        seed=42,
    )

    train_dataset = raw_split["train"]
    eval_dataset = raw_split["test"]
    save_dir = f"{base_save_dir}/model"
    checkpoint_dir = f"{base_save_dir}/checkpoints"
    logging_dir = f"{base_save_dir}/logging"


    model, tokenizer = model, tokenizer = train_model(
                model,
                tokenizer,
                train_dataset,
                eval_dataset,
                checkpoint_dir=checkpoint_dir,
                logging_dir=logging_dir,
                save_dir=save_dir,
                project_name=base_save_dir,
                batch_size=args.per_device_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                num_epochs=1,
                learning_rate=args.learning_rate,
                resume_checkpoint_path=CHECKPOINT_PATH,
            )

if __name__ == "__main__":
    main()