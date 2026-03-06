import json
import os
from unsloth import FastLanguageModel

BASE_MODEL = "allenai/Llama-3.1-Tulu-3-8B-SFT"
MODEL_ROOT = "models/tulu3-8B_pt-anthropic-769_13532"
OUTPUT_ROOT = "rlhf"
TIER_METADATA = "best_checkpoints.json"


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    with open(TIER_METADATA) as f:
        metadata = json.load(f)

    for tier_key, info in metadata.items():
        budget = tier_key.replace("$", "")
        checkpoint_path = info["best_checkpoint_path"]

        print(f"\n==== Building tier {budget} ====")
        print(f"Source checkpoint: {checkpoint_path}")

        output_dir = os.path.join(OUTPUT_ROOT, budget)

        # Skip if already built
        if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
            print(f"Skipping {output_dir} (already exists)")
            continue

        os.makedirs(output_dir, exist_ok=True)

        # Load model + merge LoRA weights
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_path,
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=False,
        )

        print("Saving merged inference model...")

        model.save_pretrained_merged(
            output_dir,
            tokenizer,
            save_method="merged_16bit",
        )

        print(f"✓ Tier {budget} saved → {output_dir}")

    print("\nAll tiers built inside rlhf/")


if __name__ == "__main__":
    main()