"""
Prints each model's actual chat-template output, straight from its own tokenizer,
so you can visually confirm the CHAT_TEMPLATE_MAP presets in utils.py actually
match what the model natively expects (special tokens, turn structure, EOS).

Usage:
    python verify_templates.py
"""
from transformers import AutoTokenizer
from utils import chat_template_func

# model_name here must match a key in utils.CHAT_TEMPLATE_MAP;
# model_id is the actual HF repo to load the tokenizer from.
MODELS_TO_CHECK = [
    {"model_name": "llama3-8B", "model_id": "unsloth/Meta-Llama-3.1-8B"},
    {"model_name": "olmo3-7B", "model_id": "allenai/Olmo-3-1025-7B"},
    {"model_name": "qwen2.5-7B", "model_id": "unsloth/Qwen2.5-7B"}
]

SAMPLE_MESSAGES = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello"},
]

for entry in MODELS_TO_CHECK:
    model_name, model_id = entry["model_name"], entry["model_id"]
    print(f"\n{'=' * 60}\n{model_name}  ({model_id})\n{'=' * 60}")

    tok = AutoTokenizer.from_pretrained(model_id)

    print("--- native tokenizer template (ground truth) ---")
    try:
        print(tok.apply_chat_template(SAMPLE_MESSAGES, tokenize=False))
    except Exception as e:
        print(f"  [no native chat_template on this tokenizer: {e}]")

    print(f"\n--- utils.chat_template_func output for model_name={model_name!r} ---")
    mapped_tok = chat_template_func(tok, model_name)
    print(mapped_tok.apply_chat_template(SAMPLE_MESSAGES, tokenize=False))

    print(f"\neos_token: {tok.eos_token!r}  |  after mapping: {mapped_tok.eos_token!r}")