from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import random

trained_model_dir = "tulu-3-8b-anthropic-dpo-10k/model"
base_model_name = "allenai/Llama-3.1-Tulu-3-8B-SFT"

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)


def load_pipeline(model_path_or_name):
    print(f"Loading model: {model_path_or_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_path_or_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    return pipe, tokenizer

trained_pipe, trained_tok = load_pipeline(trained_model_dir)

# Initialize but don't load base model yet (lazy load on switch)
base_pipe, base_tok = None, None

# Separate conversation histories
trained_msgs = [{"role": "system", "content": "You are a helpful and concise assistant."}]
base_msgs = [{"role": "system", "content": "You are a helpful and concise assistant."}]

# Start chatting
use_trained = True
print("\nChat started! Type 'switch' to swap between models, 'exit' to quit.\n")
print(f"Currently chatting with: {'DPO-trained model'}")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chat.")
        break

    if user_input.lower() == "switch":
        use_trained = not use_trained
        if not use_trained and base_pipe is None:
            base_pipe, base_tok = load_pipeline(base_model_name)
        print(f"\nSwitched to: {'DPO-trained model' if use_trained else 'Base Tulu model'}\n")
        continue

    # Pick the active model + history
    if use_trained:
        pipe, tok, msgs = trained_pipe, trained_tok, trained_msgs
    else:
        pipe, tok, msgs = base_pipe, base_tok, base_msgs

    # Update conversation history
    msgs.append({"role": "user", "content": user_input})
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # Generate reply
    output = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)
    response = output[0]["generated_text"][len(prompt):].strip()

    print(f"Model ({'DPO' if use_trained else 'Base'}): {response}\n")

    msgs.append({"role": "assistant", "content": response})
