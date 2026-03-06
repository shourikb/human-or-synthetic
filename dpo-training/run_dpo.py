import wandb
import argparse
from config_training import ModelLoadConfig, LoraConfig, PTRunConfig, DPOTrainingConfig, DatasetConfig, SFTRunConfig
from config import HF_MODEL_MAP
from dpo_training import train_model_dpo
import random

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SFT training")

    parser.add_argument("--model", type=str, help="llama3-8B", required=True)
    parser.add_argument("--dataset", type=str, help="Dataset", required=True)
    parser.add_argument("--train_size", type=int, help="Train size", required=True)
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Per device train batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    # parser.add_argument("--do_training", action="store_true", help="Do training")
    # parser.add_argument("--do_inference", action="store_true", help="Do inference")
    # parser.add_argument("--do_evaluation", action="store_true", help="Do evaluation")

    args = parser.parse_args()

    print(args.__dict__)
    lora_config = LoraConfig()
    model_load_config = ModelLoadConfig()
    training_args = DPOTrainingConfig()

    dataset_config = DatasetConfig(
        dataset = args.dataset,
        dataset_type = "pt",
        train_size = args.train_size,
    )

    print(dataset_config)

    run_config = PTRunConfig(
        dataset_config = dataset_config,
        model_name_hf = HF_MODEL_MAP[args.model],
        model_name = args.model,
        do_training=True, # args.do_training,
        do_inference=False, # args.do_inference,
        do_evaluation=False, # args.do_evaluation,
        hashs = random.randint(1,99999)
    )

    print(run_config)

    training_args.learning_rate = args.learning_rate
    training_args.per_device_train_batch_size = args.per_device_train_batch_size
    training_args.gradient_accumulation_steps = args.gradient_accumulation_steps


    if run_config.do_training:
        run = wandb.init(name=run_config.run_name, project="tuning", reinit=True)

        with run:
            if run_config.pft_method == "dpo":
                train_model_dpo(
                    run_config = run_config,
                    lora_config = lora_config,
                    model_load_config = model_load_config,
                    training_args = training_args,
                )

    print(f"********" * 20) 