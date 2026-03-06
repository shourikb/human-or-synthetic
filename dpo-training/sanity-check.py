from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import wandb

def main():
    if os.environ.get("RANK", "0") == "0":
        wandb.init(project="sanity-check2", name="default")

    model_name = "Qwen/Qwen2-0.5B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto", 
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

    training_args = DPOConfig(
        per_device_train_batch_size=8,        
        output_dir="sanity-check",
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
    )

    trainer.train()

    if os.environ.get("RANK", "0") == "0":
        wandb.finish()

if __name__ == "__main__":
    main()
