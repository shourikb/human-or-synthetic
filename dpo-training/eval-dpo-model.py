import json
import re
import os
from vllm import LLM, SamplingParams

input_file = "../anthropic_eval.jsonl"
output_file = "evalresults.jsonl"

base_model_name = "allenai/Llama-3.1-Tulu-3-8B-SFT"
rlhf_model_name = "tulu-3-8b-anthropic-dpo-10k/model"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
base_model = LLM(model=base_model_name, gpu_memory_utilization=0.8, tensor_parallel_size=1)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
rlhf_model = LLM(model=rlhf_model_name, gpu_memory_utilization=0.8, tensor_parallel_size=1)

sampling_params = SamplingParams(n=1, min_tokens=10, max_tokens=2048, temperature=1, repetition_penalty=1.1, top_p=0.9)

def parse_conversation(text):
    conversation = [
        {
            "role": "system",
            "content": "Provide the most helpful and honest response based on the conversation."
        }
    ]

    pattern = re.compile(r'(Human|Assistant):\s*(.*?)\s*(?=(Human|Assistant|$))', re.DOTALL)

    for match in pattern.finditer(text):
        role, content = match.group(1), match.group(2).strip()
        if role == "Human":
            conversation.append({
                "role": "user",
                "content": content
            })
        elif role == "Assistant":
            conversation.append({
                "role": "assistant",
                "content": content
            })

    return conversation

def generate_response(model, conversation):
    outputs = model.chat(conversation, sampling_params=sampling_params, use_tqdm=False)
    return outputs[0].outputs[0].text

def remove_last_assistant_block(text):
    pattern = re.compile(r'(Assistant:\s*[\s\S]*?)(?=(Human:|$))', re.DOTALL)
    matches = list(pattern.finditer(text))
    if matches:
        last = matches[-1]
        text = text[:last.start()].rstrip()
    return text

def main():
    linesread = 0
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            if linesread >= 10:
                break

            data = json.loads(line)
            if "chosen" not in data:
                continue

            conversation_text = data["chosen"]
            conversation_text = remove_last_assistant_block(conversation_text)
            parsed = parse_conversation(conversation_text)

            base_response = generate_response(base_model, parsed)
            rlhf_response = generate_response(rlhf_model, parsed)

            output_obj = {
                "conversation": conversation_text,
                "base_response": base_response,
                "rlhf_response": rlhf_response
            }

            outfile.write(json.dumps(output_obj) + "\n")
            linesread += 1


if __name__ == "__main__":
    main()
