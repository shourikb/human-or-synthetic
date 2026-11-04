import argparse

from tuning.training.sft_training import train_model_sft

import wandb

from tuning.config import DATASETS_DIR, HF_MODEL_MAP, MODELS_DIR
from tuning.training.config_training import (
    DatasetConfig,
    LoraConfig,
    ModelLoadConfig,
    SFTRunConfig,
    TrainingArgumentsConfig,
    add_hparam_args,
    hparam_overrides,
)


def parse_args():
    parser = argparse.ArgumentParser(description="SFT training")
    parser.add_argument("--model", type=str, required=True, choices=sorted(HF_MODEL_MAP), help="Base model")
    parser.add_argument("--dataset", type=str, required=True, help="Reads <data_dir>/sft-<dataset>.jsonl")
    parser.add_argument("--train_size", type=int, required=True, help="Number of training examples to sample")
    parser.add_argument("--data_dir", type=str, default=DATASETS_DIR)
    parser.add_argument("--models_dir", type=str, default=MODELS_DIR, help="Where the trained model is saved")
    parser.add_argument("--results_csv", type=str, default=None,
                        help="Optional CSV to append this run's summary to (e.g. for hyperparameter sweeps)")
    parser.add_argument("--wandb_project", type=str, default="tuning")
    add_hparam_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    print(vars(args))

    run_config = SFTRunConfig(
        model_name=args.model,
        model_name_hf=HF_MODEL_MAP[args.model],
        dataset_config=DatasetConfig(
            dataset=args.dataset,
            dataset_type="sft",
            train_size=args.train_size,
            data_dir=args.data_dir,
        ),
        models_dir=args.models_dir,
    )
    training_args = TrainingArgumentsConfig(**hparam_overrides(args))
    print(run_config)
    print(training_args)

    with wandb.init(name=run_config.run_name, project=args.wandb_project):
        train_model_sft(
            run_config=run_config,
            lora_config=LoraConfig(),
            model_load_config=ModelLoadConfig(),
            training_args=training_args,
            results_csv=args.results_csv,
        )


if __name__ == "__main__":
    main()
