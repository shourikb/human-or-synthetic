import argparse
import random

from tuning.training.dpo_training import train_model_dpo

import wandb

from tuning.config import DATASETS_DIR, HF_MODEL_MAP, MODELS_DIR
from tuning.training.config_training import (
    DatasetConfig,
    DPOTrainingConfig,
    LoraConfig,
    ModelLoadConfig,
    PTRunConfig,
    add_hparam_args,
    hparam_overrides,
)


def parse_args():
    parser = argparse.ArgumentParser(description="DPO training")
    parser.add_argument("--model", type=str, required=True, choices=sorted(HF_MODEL_MAP),
                        help="Base model family; selects the chat template")
    parser.add_argument("--init_model", type=str, default=None,
                        help="Checkpoint to start from, e.g. an SFT model from run_sft.py "
                             "(default: the base model on Hugging Face)")
    parser.add_argument("--dataset", type=str, required=True, help="Reads <data_dir>/pt-<dataset>.jsonl")
    parser.add_argument("--train_size", type=int, required=True, help="Number of preference pairs to sample")
    parser.add_argument("--data_dir", type=str, default=DATASETS_DIR)
    parser.add_argument("--models_dir", type=str, default=MODELS_DIR, help="Where the trained model is saved")
    parser.add_argument("--beta", type=float, default=None, help="Override the default DPO beta")
    parser.add_argument("--results_csv", type=str, default=None,
                        help="Optional CSV to append this run's summary to (e.g. for hyperparameter sweeps)")
    parser.add_argument("--wandb_project", type=str, default="tuning")
    add_hparam_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    print(vars(args))

    run_config = PTRunConfig(
        model_name=args.model,
        model_path=args.init_model or HF_MODEL_MAP[args.model],
        dataset_config=DatasetConfig(
            dataset=args.dataset,
            dataset_type="pt",
            train_size=args.train_size,
            data_dir=args.data_dir,
        ),
        models_dir=args.models_dir,
        run_id=random.randint(1, 99999),
    )
    overrides = hparam_overrides(args)
    if args.beta is not None:
        overrides["beta"] = args.beta
    training_args = DPOTrainingConfig(**overrides)
    print(run_config)
    print(training_args)

    with wandb.init(name=run_config.run_name, project=args.wandb_project):
        train_model_dpo(
            run_config=run_config,
            lora_config=LoraConfig(),
            model_load_config=ModelLoadConfig(),
            training_args=training_args,
            results_csv=args.results_csv,
        )


if __name__ == "__main__":
    main()
