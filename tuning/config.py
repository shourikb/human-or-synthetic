import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

HF_MODEL_MAP = {
    "llama3-8B": "unsloth/Meta-Llama-3.1-8B",
    "olmo3-7B": "allenai/Olmo-3-1025-7B",
    "qwen2.5-7B": "unsloth/Qwen2.5-7B",
}
