import argparse

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from generation.utils import (
    chop_after_prefix,
    chop_before_prefix,
    load_personas,
    reset_seeds,
    set_visible_gpus,
    write_jsonl,
)

HUMAN = "Human:"
ASSISTANT = "AI Assistant:"

SAMPLE_CONVERSATION = """Human: Do you know the reasons as to why people love coffee so much?

    AI Assistant: That's an interesting question!  I think it's partly because of the emotional effect of the caffeine.  Is that the main reason you're asking?

    Human: The emotional effect? Such as?\x20

    AI Assistant: I think you'll notice that coffee can have different effects on people, depending on what they were doing beforehand.  In general, coffee is stimulating, which is important to many people when they are feeling sluggish or apathetic.  But it can also be really relaxing for people who are anxious, nervous, or stressed.

    Human: I feel like how energized it makes one feel it would do the opposite of feeling relaxed.

    AI Assistant: I understand! But what I meant is, coffee can make people more nervous or relaxed, depending on how they are feeling beforehand.  If they're feeling stressed and needing to relax, coffee can have a soothing effect, but if they're feeling anxious or nervous, coffee can make them feel more agitated.

    Human: Hm, I see. Where did you read this?

    AI Assistant: I'm just recalling a couple of things I've read.  One of them was a psychology paper, but I'm also just drawing from my own experiences.

    Human: I'm a little skeptical about what you're saying but you certainly got some stuff right such as coffee being a stimulant and helping those feeling sluggish

    AI Assistant: Glad I was able to answer your questions!"

    """

FIRST_TURN_SYSTEM = "Generate a conversation starter as if you were the persona given"

FIRST_TURN_TEMPLATE = """Decide on a task you'd like an AI assistant to help you with and try to converse with this assistant. You are a {persona}. The conversation should be open ended, asking for help, advice, or for the model to accomplish a task. This should simulate having a conversation. Below is a sample conversation for style and formatting reference:

            === SAMPLE CONVERSATION ===
            {sample}
            === END SAMPLE ===

            Now, write a **completely new** conversation on a different topic. Do NOT discuss or continue the sample conversation topic. Instead, focus on the real conversation.

            Generate the first human turn of the conversation as if you were a {persona} starting a conversation in English as the Human speaking to an AI assistant.
            """

HUMAN_TURN_SYSTEM = "Given the above conversation, generate the just the next human part of the conversation for the human by trying to converse with the assistant. Follow the topic provided in the conversation. Try to generate of similar length to a single response in the sample conversation."

ASSISTANT_TURN_SYSTEM = "Continue the conversation, provide just the AI assistant response to the last human message. Try to generate of similar length to a single response in the sample conversation."

NEXT_TURN_TEMPLATE = (
    "\n"
    "        You are generating the next line in a conversation between a human and an assistant.\n"
    "        Below is a sample conversation for style:\n"
    "        === SAMPLE CONVERSATION === \n"
    "        {sample}\n"
    "        === END SAMPLE ===\n"
    "        \n"
    "        Now continue the real conversation below.\n"
    "        Only add one new line in the correct speaker's voice.\n"
    "        \n"
    "        === REAL CONVERSATION SO FAR===  \n"
    "        {convo} \n"
    "        === CONTINUE ON THE NEXT LINE===\n"
    "        {prefix}\n"
    "        "
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate persona-conditioned multi-turn conversations with two candidate final responses")
    parser.add_argument("--personas_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--first_turns_file", type=str, default=None,
                        help="Optional text file to append each generated first human turn to")
    parser.add_argument("--model", type=str, default="allenai/Llama-3.1-Tulu-3-8B-SFT")
    parser.add_argument("--gpu_id", type=str, default="0", help="Comma-separated GPU ids")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--batch_start", type=int, default=0)
    parser.add_argument("--batch_count", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=None,
                        help="Base seed; batch b uses seed + b. Random per batch if omitted")
    return parser.parse_args()


def make_sampling_params(seed, max_tokens):
    return SamplingParams(n=1, min_tokens=10, max_tokens=max_tokens, temperature=1, seed=seed,
                          repetition_penalty=1.1, top_p=0.9)


def generate_turn(llm, tokenizer, convos, sampling_params, is_human, input_tokens, output_tokens):
    if is_human:
        system, prefix, other_prefix = HUMAN_TURN_SYSTEM, HUMAN, ASSISTANT
    else:
        system, prefix, other_prefix = ASSISTANT_TURN_SYSTEM, ASSISTANT, HUMAN

    prompts = [
        [
            {"role": "system", "content": system},
            {"role": "user", "content": NEXT_TURN_TEMPLATE.format(sample=SAMPLE_CONVERSATION, convo=convo, prefix=prefix)},
        ]
        for convo in convos
    ]
    outputs = llm.chat(prompts, continue_final_message=True, add_generation_prompt=False,
                       sampling_params=sampling_params, use_tqdm=True)

    turns = []
    for i, output in enumerate(outputs):
        input_tokens[i] += len(tokenizer.encode(prompts[i][1]["content"]))
        output_tokens[i] += len(output.outputs[0].token_ids)

        text = output.outputs[0].text.strip()
        if not text:
            print(f"Empty generation at index {i}")
            text = prefix + " [GENERATION FAILED]"
        text = chop_after_prefix(chop_before_prefix(text, prefix), other_prefix)
        if not text.startswith(prefix):
            text = prefix + " " + text
        turns.append(text)
    return turns


def generate_first_turns(llm, tokenizer, personas, sampling_params, input_tokens, output_tokens):
    prompts = [
        [
            {"role": "system", "content": FIRST_TURN_SYSTEM},
            {"role": "user", "content": FIRST_TURN_TEMPLATE.format(persona=persona, sample=SAMPLE_CONVERSATION)},
        ]
        for persona in personas
    ]
    outputs = llm.chat(prompts, sampling_params=sampling_params, use_tqdm=True)

    turns = []
    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        if len(text) < 10:
            text = "ALERT ALERT FAILURE ALERT ALERT"
        text = chop_after_prefix(chop_before_prefix(text, HUMAN), ASSISTANT)
        if not text.startswith(HUMAN):
            text = HUMAN + " " + text
        turns.append(text)
        output_tokens[i] += len(output.outputs[0].token_ids)
        input_tokens[i] += len(tokenizer.encode(prompts[i][1]["content"]))
    return turns


def main():
    args = parse_args()
    set_visible_gpus(args.gpu_id)

    all_personas = load_personas(args.personas_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_memory_utilization)

    for batch in range(args.batch_start, args.batch_start + args.batch_count):
        reset_seeds(None if args.seed is None else args.seed + batch)
        n = args.batch_size
        input_tokens = [0] * n
        output_tokens = [0] * n
        personas = list(np.random.choice(all_personas, n, replace=False))

        params = make_sampling_params(np.random.randint(1, 1000), args.max_tokens)
        convos = generate_first_turns(llm, tokenizer, personas, params, input_tokens, output_tokens)
        first_turns = list(convos)

        num_turns = np.random.randint(0, 3)
        for _ in range(num_turns):
            for is_human in (False, True):
                turns = generate_turn(llm, tokenizer, convos, params, is_human, input_tokens, output_tokens)
                convos = [c + "\n\n" + t for c, t in zip(convos, turns)]

        responses_a = generate_turn(llm, tokenizer, convos, params, False, input_tokens, output_tokens)
        params_b = make_sampling_params(np.random.randint(1001, 2000), args.max_tokens)
        responses_b = generate_turn(llm, tokenizer, convos, params_b, False, input_tokens, output_tokens)

        records = []
        for i in range(n):
            info = (f"\nPersona: {personas[i]}\nTurns: {num_turns}"
                    f"\nTotal Input Tokens: {input_tokens[i]}\nTotal Output Tokens: {output_tokens[i]}\n")
            records.append({
                "info": info,
                "conversation": convos[i],
                "response_A": responses_a[i] + "\n",
                "response_B": responses_b[i] + "\n",
            })
        write_jsonl(args.output_file, records, mode="a")

        if args.first_turns_file:
            with open(args.first_turns_file, "a", encoding="utf-8") as f:
                for turn in first_turns:
                    f.write(turn + "\nNEXT RESPONSE\n")

        print(f"Finished batch {batch}")


if __name__ == "__main__":
    main()
