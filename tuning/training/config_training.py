import os
from typing import Optional

from pydantic import BaseModel

from tuning.config import DATASETS_DIR, MODELS_DIR

BaseModel.model_config["protected_namespaces"] = ()


class ModelLoadConfig(BaseModel):
    max_seq_length: int = 1024
    dtype: Optional[str] = None
    load_in_4bit: bool = False


class LoraConfig(BaseModel):
    r: int = 16
    target_modules: list[str] = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    lora_alpha: int = 32
    lora_dropout: float = 0
    bias: str = "none"
    use_gradient_checkpointing: bool = False
    random_state: int = 42
    use_rslora: bool = False
    loftq_config: Optional[dict] = None


class TrainingArgumentsConfig(BaseModel):
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_train_epochs: int = 2
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 1
    save_steps: int = 1000
    seed: int = 42
    report_to: list[str] = ["wandb"]

    def trainer_kwargs(self):
        return self.model_dump(exclude={"beta"})


class DPOTrainingConfig(TrainingArgumentsConfig):
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-6
    beta: float = 1.0


OVERRIDABLE_HPARAMS = {
    "learning_rate": float,
    "per_device_train_batch_size": int,
    "gradient_accumulation_steps": int,
    "num_train_epochs": int,
}


def add_hparam_args(parser):
    for name, type_ in OVERRIDABLE_HPARAMS.items():
        parser.add_argument(f"--{name}", type=type_, default=None,
                            help=f"Override the default {name}")


def hparam_overrides(args):
    return {name: getattr(args, name) for name in OVERRIDABLE_HPARAMS if getattr(args, name) is not None}


class DatasetConfig(BaseModel):
    dataset: str
    dataset_type: str
    train_size: int
    data_dir: str = DATASETS_DIR

    @property
    def dataset_full_name(self):
        return f"{self.dataset_type}-{self.dataset}-{self.train_size}"

    def __str__(self):
        return self.dataset_full_name


class SFTRunConfig(BaseModel):
    model_name: str
    model_name_hf: str
    dataset_config: DatasetConfig
    models_dir: str = MODELS_DIR

    @property
    def run_name(self):
        return f"{self.model_name}_{self.dataset_config.dataset_full_name}"

    @property
    def output_dir(self):
        return os.path.join(self.models_dir, self.run_name)

    def __str__(self):
        return self.run_name


class PTRunConfig(BaseModel):
    model_name: str
    model_path: str
    dataset_config: DatasetConfig
    run_id: int
    models_dir: str = MODELS_DIR

    @property
    def run_name(self):
        if os.path.isdir(self.model_path):
            prefix = os.path.basename(os.path.normpath(self.model_path))
        else:
            prefix = self.model_name
        return f"{prefix}_{self.dataset_config.dataset_full_name}_{self.run_id}"

    @property
    def output_dir(self):
        return os.path.join(self.models_dir, self.run_name)

    def __str__(self):
        return self.run_name
