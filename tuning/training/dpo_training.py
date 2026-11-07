from unsloth import PatchDPOTrainer, is_bfloat16_supported

from trl import DPOConfig, DPOTrainer

from tuning.data.train_dataset import get_train_dataset
from tuning.training.common import load_lora_model, save_run_results
from tuning.training.config_training import DPOTrainingConfig, LoraConfig, ModelLoadConfig, PTRunConfig
from tuning.utils.utils import apply_chat_template_pt, chat_template_func

PatchDPOTrainer()


def train_model_dpo(
    run_config: PTRunConfig,
    lora_config: LoraConfig,
    model_load_config: ModelLoadConfig,
    training_args: DPOTrainingConfig,
    results_csv: str = None,
):
    dataset = get_train_dataset(run_config.dataset_config)

    print(f"Loading model from {run_config.model_path}")
    model, tokenizer = load_lora_model(run_config.model_path, model_load_config, lora_config)
    dataset = apply_chat_template_pt(tokenizer, dataset, run_config.model_name)

    example = dataset["train"][0]
    print("=== SAMPLE DPO EXAMPLE AFTER TEMPLATING ===")
    print("PROMPT:", example["prompt"])
    print("CHOSEN:", example["chosen"])
    print("REJECTED:", example["rejected"])
    print("=== END SAMPLE ===")

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        tokenizer=chat_template_func(tokenizer, run_config.model_name),
        beta=training_args.beta,
        train_dataset=dataset["train"],
        max_length=model_load_config.max_seq_length,
        args=DPOConfig(
            output_dir=run_config.output_dir,
            save_strategy="steps",
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            **training_args.trainer_kwargs(),
        ),
    )
    trainer_stats = trainer.train()

    model.save_pretrained_merged(run_config.output_dir, tokenizer, save_method="merged_16bit")
    save_run_results(trainer, trainer_stats, run_config, training_args, results_csv)
