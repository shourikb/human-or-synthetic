import wandb
from config_training import ModelLoadConfig, LoraConfig, SFTRunConfig, TrainingArgumentsConfig, DatasetConfig
from config import HF_MODEL_MAP
from sft_training import train_model_sft

import argparse

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
    training_args = TrainingArgumentsConfig()

    train_size = args.train_size 
    dataset_config = DatasetConfig(
        dataset = args.dataset,
        dataset_type = "sft",
        train_size = args.train_size,
    ) 

    print(dataset_config)

    run_config = SFTRunConfig(
        dataset_config = dataset_config,
        model_name_hf = HF_MODEL_MAP[args.model],
        model_name = args.model,
        do_training=True,
        do_inference=False,
        do_evaluation=False,
    )

    print(run_config)

    # training_args.learning_rate = args.learning_rate
    # training_args.per_device_train_batch_size = args.per_device_train_batch_size
    # training_args.gradient_accumulation_steps = args.gradient_accumulation_steps


    if run_config.do_training:
        run = wandb.init(name=run_config.run_name, project="tuning", reinit=True)

        with run:
            train_model_sft(
                run_config = run_config,
                lora_config = lora_config,
                model_load_config = model_load_config,
                training_args = training_args,
            )


    print(f"********" * 20) 