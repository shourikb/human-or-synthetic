from typing import Union
from config_training import DatasetConfig, PTRunConfig, SFTRunConfig
from datasets import load_from_disk, DatasetDict, load_dataset
from config import DATASETS_DIR
from pathlib import Path
import random

def get_train_dataset(run_config: Union[PTRunConfig, SFTRunConfig]) -> DatasetDict:

    print(f"Getting train dataset for run config: {run_config}")

    if run_config.run_type == "pt":

        # if run_config.sft_run_config is None:
        #     dataset_stub = f"pt-{run_config.dataset_config.dataset}"
        # else:
        #     sft_run_name = run_config.sft_run_config.run_name
        #     sft_run_name = sft_run_name.replace("_", "|").replace("-", "|")
        #     dataset_stub = f"pt-{sft_run_name}"

        dataset_stub = f"pt-{run_config.dataset_config.dataset}"
    elif run_config.run_type == "sft":
        dataset_stub = f"sft-{run_config.dataset_config.dataset}"
    
    save_name = f"{dataset_stub}-{run_config.dataset_config.train_size}"
    check_path = f"{DATASETS_DIR}/{save_name}"

    print(f"Checking for dataset at {check_path}")
    if Path(check_path).exists():
        print(f"Dataset already exists at {check_path}")
        return load_from_disk(check_path)

    train_size = run_config.dataset_config.train_size
    
    full_dataset_path = f"{DATASETS_DIR}/{dataset_stub}.jsonl"
    print(f"Loading dataset from {full_dataset_path}")
    full_dataset = load_dataset('json', data_files=full_dataset_path) # load_from_disk(full_dataset_path) 
    print(f"Full dataset: {full_dataset}")

    sampled_dataset = get_random_train_subset(full_dataset, train_size, 123)
    print(f"Sampled dataset: {sampled_dataset}")
    print(f"Example row: {sampled_dataset['train'][0]}")

    sampled_dataset.save_to_disk(f"{DATASETS_DIR}/{dataset_stub}-{train_size}")

    return sampled_dataset

def get_random_train_subset(dataset: DatasetDict, total_size: int, seed: int) -> DatasetDict:
    random.seed(seed)

    train_split = dataset["train"]
    if "test" in dataset:
        test_split = dataset["test"]
    else:
        # Split the train data into train and test
        # Take a small portion for test (e.g., 10% or a fixed number like 100)
        test_size = (total_size * 10) // 100  # 10%
        train_size = total_size - test_size
        
        # Create indices for test split
        all_indices = list(range(len(train_split)))
        random.shuffle(all_indices)
        
        test_indices = all_indices[:test_size]
        train_indices = all_indices[test_size:test_size + train_size]
        
        test_split = train_split.select(test_indices)
        train_split = train_split.select(train_indices)

    sampled_dataset = DatasetDict()
    sampled_dataset["train"] = train_split
    sampled_dataset["test"] = test_split

    return sampled_dataset

if __name__ == "__main__":

    run_config = SFTRunConfig(
            model_name="llama3-8B",
            task_name="math",
            dataset_config=DatasetConfig(
                dataset="gsm8k",
                dataset_type="sft",
                train_size=10000,
            ),
            do_training=True,
        )

    dataset = get_train_dataset(run_config=run_config)

    print(f"Dataset: {dataset}")


