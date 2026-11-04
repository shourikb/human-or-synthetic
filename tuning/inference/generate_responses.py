import argparse
import json
import os
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams

from tuning.utils.utils import chat_template_func, split_conversation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path or name of the model to generate responses with")
    parser.add_argument("--model_name", type=str, required=True,
                         help="Key into CHAT_TEMPLATE_MAP, e.g. llama3-8B or olmo3-7B — "
                              "determines which chat template gets applied")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--num_samples", type=int, required=True, help="Use the first N conversations of input_file")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=55, help="Sampling seed")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    return parser.parse_args()


def extract_prompt(chosen_text):
    split_tag = "Assistant:"
    idx = chosen_text.rfind(split_tag)
    if idx == -1:
        raise ValueError("No 'Assistant:' tag found.")
    return chosen_text[: idx + len(split_tag)]


def extract_first_response(text):
    text = re.sub(r'^Assistant:\s*', '', text.strip())
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    role_tag = r'(?im)^[ \t]*(human|user|assistant|ai assistant)\s*:'
    match = re.search(role_tag, text)
    if match:
        return text[:match.start()].strip()

    return text.strip()


def build_prompt(tokenizer, conv_text):
    messages = split_conversation(conv_text)
    messages = [m for m in messages if m["content"]]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_outputs(model_path, model_name, conversations, batch_size, seed, tp=1):
    print(f"\nLoading model: {model_path}")
    llm = LLM(model=model_path, tensor_parallel_size=tp)
    tokenizer = llm.get_tokenizer()
    tokenizer = chat_template_func(tokenizer, model_name)
    max_model_len = llm.llm_engine.model_config.max_model_len

    sampling_params = SamplingParams(
        n=1,
        min_tokens=10,
        max_tokens=2048,
        temperature=1,
        repetition_penalty=1.1,
        top_p=0.9,
        top_k=50,
        seed=seed,
        stop=["<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "<|endoftext|>"],
    )

    outputs = []

    for start in tqdm(range(0, len(conversations), batch_size)):
        batch = conversations[start : start + batch_size]
        batch_prompts = []
        for conv in batch:
            prompt = build_prompt(tokenizer, conv)
            tokens = tokenizer.encode(prompt)
            if len(tokens) > max_model_len - 200:
                tokens = tokens[-(max_model_len - 200):]
                prompt = tokenizer.decode(tokens)
            batch_prompts.append(prompt)

        if start == 0:
            print("=== SAMPLE PROMPT SENT TO MODEL ===")
            print(batch_prompts[0])
            print("=== END SAMPLE ===")

        batch_outputs = llm.generate(batch_prompts, sampling_params=sampling_params)
        outputs.extend([extract_first_response(o.outputs[0].text) for o in batch_outputs])

    return outputs


def main():
    args = parse_args()

    with open(args.input_file, "r") as f:
        lines = [json.loads(l) for l in f]

    lines = lines[: args.num_samples]
    conversations = [extract_prompt(l["chosen"]) for l in lines]

    responses = generate_outputs(args.model, args.model_name, conversations, args.batch_size,
                                 args.seed, args.tensor_parallel_size)

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, "w") as f:
        for conv, response in zip(conversations, responses):
            f.write(json.dumps({"conversation": conv, "response": response}) + "\n")

    print(f"\nSaved {len(responses)} responses to {args.output_file}")


if __name__ == "__main__":
    main()
