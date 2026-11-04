import os
import random

from datasets import DatasetDict, load_dataset, load_from_disk

from tuning.training.config_training import DatasetConfig

SUBSET_SEED = 555


def get_train_dataset(dataset_config: DatasetConfig) -> DatasetDict:
    stub = f"{dataset_config.dataset_type}-{dataset_config.dataset}"
    subset_path = os.path.join(dataset_config.data_dir, f"{stub}-{dataset_config.train_size}")
    if os.path.exists(subset_path):
        print(f"Loading saved subset from {subset_path}")
        return load_from_disk(subset_path)

    full_path = os.path.join(dataset_config.data_dir, f"{stub}.jsonl")
    print(f"Sampling {dataset_config.train_size} rows from {full_path}")
    full = load_dataset("json", data_files=full_path)["train"]
    if dataset_config.train_size > len(full):
        print(f"WARNING: requested {dataset_config.train_size} rows but {full_path} only has {len(full)}")

    indices = list(range(len(full)))
    random.Random(SUBSET_SEED).shuffle(indices)
    subset = DatasetDict({"train": full.select(indices[:dataset_config.train_size])})

    subset.save_to_disk(subset_path)
    return subset
