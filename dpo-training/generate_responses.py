"""
Usage:
    python generate_responses.py \
        --model rlhf/1000 \
        --input_file ../anthropic_test.jsonl \
        --num_samples 64 \
        --batch_size 64
"""
import argparse
import json
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path or name of the model to generate responses with")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    return parser.parse_args()


def extract_prompt(chosen_text):
    split_tag = "Assistant:"
    idx = chosen_text.rfind(split_tag)
    if idx == -1:
        raise ValueError("No 'Assistant:' tag found.")
    return chosen_text[: idx + len(split_tag)]

def extract_first_response(text):
    text = re.sub(r'^Assistant:\s*', '', text.strip())
    match = re.search(r'\n\s*(Human:|User:|AI Assistant:|Assistant:)', text)
    if match:
        return text[:match.start()].strip()
    return text.strip()

def generate_outputs(model_path, conversations, batch_size, tp=1):
    print(f"\nLoading model: {model_path}")
    llm = LLM(model=model_path, tensor_parallel_size=tp)

    sampling_params = SamplingParams(
        n=1,
        min_tokens=10,
        max_tokens=2048,
        temperature=1,
        repetition_penalty=1.1,
        top_p=0.9,
    )
    tokenizer = llm.get_tokenizer()
    max_model_len = llm.llm_engine.model_config.max_model_len

    outputs = []

    for start in tqdm(range(0, len(conversations), batch_size)):
        batch = conversations[start : start + batch_size]
        batch_prompts = []
        for conv in batch:
            prompt = f"Continue the conversation. Provide only the assistant response.\n\n{conv}\n"
            tokens = tokenizer.encode(prompt)
            if len(tokens) > max_model_len - 200:
                tokens = tokens[-(max_model_len - 200):]
                prompt = tokenizer.decode(tokens)
            batch_prompts.append(prompt)

        batch_outputs = llm.generate(batch_prompts, sampling_params=sampling_params)
        outputs.extend([extract_first_response(o.outputs[0].text) for o in batch_outputs])

    return outputs


def main():
    args = parse_args()

    with open(args.input_file, "r") as f:
        lines = [json.loads(l) for l in f]

    lines = lines[: args.num_samples]
    conversations = [extract_prompt(l["chosen"]) for l in lines]

    responses = generate_outputs(args.model, conversations, args.batch_size)

    model_name = args.model.replace("/", "_")
    output_file = f"responses_{model_name}.jsonl"

    with open(output_file, "w") as f:
        for conv, response in zip(conversations, responses):
            f.write(json.dumps({"conversation": conv, "response": response}) + "\n")

    print(f"\nSaved {len(responses)} responses to {output_file}")


if __name__ == "__main__":
    main()