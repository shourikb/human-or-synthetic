from vllm import LLM
from vllm import SamplingParams
from transformers import AutoTokenizer
import random
import json
from collections import Counter
import time
import numpy as np
import os
import argparse


def chop_before_prefix(s, prefix):
    index = s.find(prefix)
    if index != -1:
        return s[index:]
    return s

def chop_after_prefix(s, prefix):
    index = s.find(prefix)
    if index != -1:
        return s[:index]
    return s

def parse_args():
    parser = argparse.ArgumentParser(description='Run batch processing on specific GPU')
    parser.add_argument('--gpu_id', type=str, default="0", help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--batch_start', type=int, default=0, help='Starting batch index')
    parser.add_argument('--batch_count', type=int, default=1, help='Number of batches to process')
    parser.add_argument('--output_suffix', type=str, default="", help='Suffix for output files')
    return parser.parse_args()

def reset_all_seeds():
    seed = int.from_bytes(os.urandom(4), byteorder="big")
    random.seed(seed)
    np.random.seed(seed)
    return seed

test_tracker = ["" for _ in range(64)]
def eachturn(convos, model, sp, is_human, sample, tokens, input_tokens, tokenizer, convo_tracker=[], update=True):
    time.sleep(1)
    if is_human:
        prompt = "Given the above conversation, generate the just the next human part of the conversation for the human by trying to converse with the assistant. Follow the topic provided in the conversation. Try to generate of similar length to a single response in the sample conversation."
        prefix = "Human:"
        alt_prefix = "AI Assistant:"
    else:
        prompt = "Continue the conversation, provide just the AI assistant response to the last human message. Try to generate of similar length to a single response in the sample conversation."
        prefix = "AI Assistant:"
        alt_prefix = "Human:"
        
    all_conversations = []
    for convo in convos:
        context_convo = f"""
        You are generating the next line in a conversation between a human and an assistant.
        Below is a sample conversation for style:
        === SAMPLE CONVERSATION === 
        {sample}
        === END SAMPLE ===
        
        Now continue the real conversation below.
        Only add one new line in the correct speaker's voice.
        
        === REAL CONVERSATION SO FAR===  
        {convo} 
        === CONTINUE ON THE NEXT LINE===
        {prefix}
        """
        
        full_conversation = [
             {
                "role": "system",
                "content": prompt
            },
            {
                "role":"user",
                "content":context_convo
            }
        ]
        all_conversations.append(full_conversation)
        # if not update:
        #     print(context_convo)
    
    outputs = model.chat(all_conversations, continue_final_message=True, add_generation_prompt=False, sampling_params=sp, use_tqdm=True)
    for i in range(len(outputs)):
        final_result = outputs[i].outputs[0].text.strip()

        input_tokens[i] += len(tokenizer.encode(all_conversations[i][1]["content"]))

        test_tracker[i] += "\n\nSWITCH\n\n" + final_result
        if not final_result:
            print(f"PANIC on index {i}")
            final_result = prefix + " [GENERATION FAILED]"
        
        final_result = chop_before_prefix(final_result, prefix)
        final_result = chop_after_prefix(final_result, alt_prefix)
        if not final_result.startswith(prefix):
            final_result = prefix + " " + final_result
        convo_tracker.append(final_result + "\n")
        if update:
            convos[i] += "\n\n" + final_result
        tokens[i] += len(outputs[i].outputs[0].token_ids)

    return convos

def main():
    args = parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # print(f"Running on GPU {args.gpu_id}")
    gpu_ids = [x.strip() for x in args.gpu_id.split(',')]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    print(f"Using GPU ID: {args.gpu_id}")

    with open("filtered_personas.jsonl", "r", encoding="utf-8") as file:
        personas = [next(iter(json.loads(line).values())) for line in file]

    model_name = "allenai/Llama-3.1-Tulu-3-8B-SFT" 
    # model_name = "unsloth/Meta-Llama-3.1-8b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.save_pretrained("tokenizer_tmp")

    # llm = LLM(model="unsloth/Meta-Llama-3.1-8b-instruct")
    llm = LLM(model=model_name, tokenizer="tokenizer_tmp", gpu_memory_utilization=0.8)

    batch_end = args.batch_start + args.batch_count
    for c in range(args.batch_start, batch_end):
        reset_all_seeds()
        information = []
        all_outputs = ["" for _ in range(64)]
        tokens = [0 for _ in range(len(all_outputs))]
        input_tokens = [0 for _ in range(len(all_outputs))]
        a_responses = []
        b_responses = []
        first_turn = []
        random_personas = list(np.random.choice(personas, len(all_outputs), replace=False))
        stopwords = [] # ["Human", "AI Assistant"]

        conversations = []
        sample_convo = """Human: Do you know the reasons as to why people love coffee so much?

    AI Assistant: That's an interesting question!  I think it's partly because of the emotional effect of the caffeine.  Is that the main reason you're asking?

    Human: The emotional effect? Such as? 

    AI Assistant: I think you'll notice that coffee can have different effects on people, depending on what they were doing beforehand.  In general, coffee is stimulating, which is important to many people when they are feeling sluggish or apathetic.  But it can also be really relaxing for people who are anxious, nervous, or stressed.

    Human: I feel like how energized it makes one feel it would do the opposite of feeling relaxed.

    AI Assistant: I understand! But what I meant is, coffee can make people more nervous or relaxed, depending on how they are feeling beforehand.  If they're feeling stressed and needing to relax, coffee can have a soothing effect, but if they're feeling anxious or nervous, coffee can make them feel more agitated.

    Human: Hm, I see. Where did you read this?

    AI Assistant: I'm just recalling a couple of things I've read.  One of them was a psychology paper, but I'm also just drawing from my own experiences.

    Human: I'm a little skeptical about what you're saying but you certainly got some stuff right such as coffee being a stimulant and helping those feeling sluggish

    AI Assistant: Glad I was able to answer your questions!"

    """
        for j in range(len(random_personas)):
            conversation = [
                {
                    "role": "system",
                    "content": "Generate a conversation starter as if you were the persona given"
                },
                {
                    "role": "user",
                    "content": f"""Decide on a task you'd like an AI assistant to help you with and try to converse with this assistant. You are a {random_personas[j]}. The conversation should be open ended, asking for help, advice, or for the model to accomplish a task. This should simulate having a conversation. Below is a sample conversation for style and formatting reference:

            === SAMPLE CONVERSATION ===
            {sample_convo}
            === END SAMPLE ===

            Now, write a **completely new** conversation on a different topic. Do NOT discuss or continue the sample conversation topic. Instead, focus on the real conversation.

            Generate the first human turn of the conversation as if you were a {random_personas[j]} starting a conversation in English as the Human speaking to an AI assistant.
            """,
                },
            ]
            conversations.append(conversation)
            information.append("\nPersona: " + random_personas[j] + "\nTurns: ")

        rand_seed = np.random.randint(1, 1000)
        sampling_params = SamplingParams(n=1, min_tokens=10, max_tokens=2048, temperature=1, seed=rand_seed, stop=stopwords, repetition_penalty=1.1, top_p=0.9)
        outputs = llm.chat(conversations, sampling_params=sampling_params, use_tqdm=True)
        time.sleep(1)
        for i in range(len(all_outputs)):
            final_result = outputs[i].outputs[0].text
            if len(final_result) < 10:
                final_result = "ALERT ALERT FAILURE ALERT ALERT"
            test_tracker[i] = final_result
            final_result = chop_before_prefix(final_result, "Human:")
            final_result = chop_after_prefix(final_result, "AI Assistant:")
            if not final_result.startswith("Human:"):
                final_result = "Human: " + final_result
            all_outputs[i] += final_result
            tokens[i] += len(outputs[i].outputs[0].token_ids)
            input_tokens[i] += len(tokenizer.encode(conversations[i][1]["content"]))
            first_turn.append(final_result + "\nNEXT RESPONSE\n")

        additional_turns = np.random.randint(0, 3)
        for i in range(len(information)):
            information[i] += str(additional_turns)
        
        if additional_turns == 0:
            eachturn(all_outputs, llm, sampling_params, False, sample_convo, tokens, input_tokens, tokenizer, convo_tracker=a_responses, update=False)
            rand_seed = np.random.randint(1001, 2000)
            smp2 = SamplingParams(n=1, min_tokens=10, max_tokens = 2048, temperature=1, seed=rand_seed, stop=stopwords, repetition_penalty=1.1, top_p=0.9)           
            eachturn(all_outputs, llm, smp2, False, sample_convo, tokens, input_tokens, tokenizer, convo_tracker=b_responses, update=False)
        else:
            all_outputs = eachturn(all_outputs, llm, sampling_params, False, sample_convo, tokens, input_tokens, tokenizer)

        for k in range(additional_turns):
            all_outputs = eachturn(all_outputs, llm, sampling_params, True, sample_convo, tokens, input_tokens, tokenizer)
            if k != additional_turns - 1:
                all_outputs = eachturn(all_outputs, llm, sampling_params, False, sample_convo, tokens, input_tokens, tokenizer)
            else:
                eachturn(all_outputs, llm, sampling_params, False, sample_convo, tokens, input_tokens, tokenizer, convo_tracker=a_responses, update=False)
                time.sleep(1)
                rand_seed = np.random.randint(1001, 2000)
                smp2 = SamplingParams(n=1, min_tokens = 10, max_tokens = 2048, temperature=1, seed=rand_seed, stop=stopwords, repetition_penalty=1.1, top_p=0.9)           
                eachturn(all_outputs, llm, smp2, False, sample_convo, tokens, input_tokens, tokenizer, convo_tracker=b_responses, update=False)

        for i in range(len(information)):
            information[i] += "\nTotal Input Tokens: " + str(input_tokens[i]) + "\nTotal Output Tokens: " + str(tokens[i]) + "\n"

        with open("sample7.jsonl", "a", encoding="utf-8") as f:
            for i in range(len(all_outputs)):
                json_entry = {
                    "info": information[i] + f"\nGPU id: {args.gpu_id}",
                    "test_convo": "".join(test_tracker[i]),
                    "conversation": "".join(all_outputs[i]),
                    "response_A": a_responses[i],
                    "response_B": b_responses[i]
                }
                f.write(json.dumps(json_entry) + "\n")

            
        with open("batch_human_prompts.txt", "a", encoding="utf-8") as f:
            for response in first_turn:
                f.write(response)

    print("Done with all batches.")

if __name__ == "__main__":
    main()