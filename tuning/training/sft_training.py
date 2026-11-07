from unsloth import is_bfloat16_supported

from transformers import TrainingArguments
from trl import SFTTrainer

from tuning.data.train_dataset import get_train_dataset
from tuning.training.common import load_lora_model, save_run_results
from tuning.training.config_training import LoraConfig, ModelLoadConfig, SFTRunConfig, TrainingArgumentsConfig
from tuning.utils.utils import apply_chat_template, chat_template_func


def train_model_sft(
    run_config: SFTRunConfig,
    lora_config: LoraConfig,
    model_load_config: ModelLoadConfig,
    training_args: TrainingArgumentsConfig,
    results_csv: str = None,
):
    dataset = get_train_dataset(run_config.dataset_config)

    model, tokenizer = load_lora_model(run_config.model_name_hf, model_load_config, lora_config)
    dataset = apply_chat_template(tokenizer, dataset, run_config.model_name)
    print(dataset)
    print(dataset["train"][0]["text"])

    trainer = SFTTrainer(
        model=model,
        tokenizer=chat_template_func(tokenizer, run_config.model_name),
        train_dataset=dataset["train"],
        dataset_text_field="text",
        max_seq_length=model_load_config.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            output_dir=run_config.output_dir,
            save_strategy="steps",
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            average_tokens_across_devices=False,
            **training_args.trainer_kwargs(),
        ),
    )
    trainer_stats = trainer.train()

    model.save_pretrained_merged(run_config.output_dir, tokenizer, save_method="merged_16bit")
    save_run_results(trainer, trainer_stats, run_config, training_args, results_csv)
