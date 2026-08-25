from vllm import LLM
from vllm import SamplingParams
from transformers import AutoTokenizer
import random
import json
import re
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

PREAMBLE_PATTERNS = [
    r"^certainly!?\s*",
    r"^sure!?\s*",
    r"^of course!?\s*",
    r"^here'?s? (a |an |another )?(fresh |unique |new )?reddit[\s-]style post.*?:?\s*",
    r"^here'?s? a post.*?:?\s*",
    r"^title:\s*.*?\n",
    r"^---+\s*",
]

def strip_generation_preamble(text):
    """Strip common chat-model preamble/meta-commentary that precedes the
    actual generated content, e.g. 'Certainly! Here's a Reddit-style post...'
    Applies patterns iteratively since models sometimes stack several
    (e.g. 'Certainly! Here's a post:\n\n---\n\n' all at once)."""
    text = text.strip()
    changed = True
    while changed:
        changed = False
        for pattern in PREAMBLE_PATTERNS:
            new_text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
            if new_text != text:
                text = new_text.strip()
                changed = True
    # Also strip a trailing "---" or meta-commentary sometimes appended after the post
    text = re.sub(r"\n*---+\s*$", "", text).strip()
    text = re.sub(r"\n*(feel free to .*|this post (is|was|embraces).*)$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    return text

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


def main():
    args = parse_args()

    gpu_ids = [x.strip() for x in args.gpu_id.split(',')]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    print(f"Using GPU ID: {args.gpu_id}")

    with open("filtered_personas.jsonl", "r", encoding="utf-8") as file:
        personas = [next(iter(json.loads(line).values())) for line in file]

    model_name = "allenai/Llama-3.1-Tulu-3-8B-SFT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained("tokenizer_tmp")

    llm = LLM(model=model_name, tokenizer="tokenizer_tmp", gpu_memory_utilization=0.8)

    batch_end = args.batch_start + args.batch_count
    for c in range(args.batch_start, batch_end):
        reset_all_seeds()
        information = []
        batch_size = 64
        all_posts = ["" for _ in range(batch_size)]
        post_tokens = [0 for _ in range(batch_size)]
        post_input_tokens = [0 for _ in range(batch_size)]
        summary_tokens = [0 for _ in range(batch_size)]
        summary_input_tokens = [0 for _ in range(batch_size)]
        random_personas = list(np.random.choice(personas, batch_size, replace=False))
        stopwords = []

        sample_post = """A little background first. Like most engineers I'm taking differential equations and also solid mechanics. For solid mechanics we have to do a lab in addition to taking the class. Pretty standard stuff.

Anyway, the class had us sign up for labs in groups by providing sign up sheets where we could sign up for which lab we wanted by providing our name and student ID numbers. The lab assignments were posted online today with a column for our names and student ID numbers (which had been blacked out). Somehow, my name wasn't on the list and in frustration I began dragging the scroll bar up and down frantically hoping that I'd magically see my name; that's when I saw it.

When dragging the scroll bar fast enough, I found that the blacked out student ID numbers would temporarily become visible. At first I thought, "No big deal, what could someone do with a person's ID number." Then it hit me. My university uses a web based homework website (WebWork) that initially allows students to access their account by inputting their user ID number as their user name and password. What's worse is that this site NEVER PROMPTS THE STUDENT TO CREATE A NEW PASSWORD. As such, most students never change their password from their student ID number.

Realizing that someone who happened upon this discovery could in turn use it to access other student's homework assignments and tamper with them, I contacted the administrator in charge of assigning lab schedules. In doing so, I not only signed up for a lab group but had them completely remove the ID numbers from the list.

Best part is that nobody in my class will know. That's why I wanted to tell you reddit."""

        sample_summary = "A math class had us sign up for a lab group by providing our name and student ID numbers. Someone discovered how easy it was to steal homework assignments and nobody will know."

        # ---- Stage 1: generate a persona-conditioned Reddit-style post ----
        post_conversations = []
        for j in range(len(random_personas)):
            conversation = [
                {
                    "role": "system",
                    "content": "Generate a Reddit-style post as if you were the persona given. Output ONLY the raw post text. Never include any preamble, acknowledgment, introduction, or meta-commentary such as 'Certainly! Here's a post...' or 'Title:' or '---'. Your entire response must be the post itself and nothing else."
                },
                {
                    "role": "user",
                    "content": f"""You are a {random_personas[j]}. Write a Reddit-style post describing a situation, story, or question you'd want summarized by someone else. The post should be open ended, sharing a personal story, asking for advice, or describing an event in detail, similar to a real Reddit post. Below is a sample post for style and length reference:

                    === SAMPLE POST ===
                    {sample_post}
                    === END SAMPLE ===

                    Now, write a **completely new** post on a different topic. Do NOT discuss or continue the sample post's topic. Instead, focus on your own situation.

                    The post's SUBJECT MATTER must be clearly and specifically grounded in this persona's expertise, role, circumstances, or interests — not a generic post that could have come from anyone. A reader should be able to infer roughly who is writing (their job, background, or situation) from the content alone, without being told the persona directly. Draw on concrete details, terminology, concerns, or situations that someone in this exact position would plausibly encounter — avoid defaulting to unrelated generic topics unless they are a natural and specific expression of this particular persona.

                    Do not include a title, preamble, acknowledgment, or any commentary before or after the post. Do not write phrases like "Certainly!" or "Here's a Reddit-style post." Begin your response immediately with the first word of the post itself, as if you are the Reddit user typing directly into the text box.

                    Vary your opening. Pick whatever opening (mid-action, a direct question, an exclamation, a specific concrete detail, dialogue) best fits this specific persona and story, and make sure it differs from how you would open a post for a different persona.

                    Write your post at a moderate length — aim for roughly 3-5 paragraphs of
                    detail (around 200-400 words), enough to include real specifics (what
                    happened, when, who was involved, why it matters to you) rather than a
                    short one- or two-paragraph summary of the situation. Real Reddit posts in
                    this style usually include enough backstory and detail that a reader
                    unfamiliar with the situation could follow along without needing to ask
                    follow-up questions.

                    Do not pad the post with unnecessary repetition or filler just to make it
                    longer — every sentence should add a genuine detail, not restate something
                    already said.
                    """,
                },
            ]
            post_conversations.append(conversation)
            information.append("\nPersona: " + random_personas[j])

        rand_seed = np.random.randint(1, 1000)
        post_sampling_params = SamplingParams(n=1, min_tokens=10, max_tokens=2048, temperature=1, seed=rand_seed, stop=stopwords, repetition_penalty=1.1, top_p=0.9)
        post_outputs = llm.chat(post_conversations, sampling_params=post_sampling_params, use_tqdm=True)
        time.sleep(1)

        for i in range(len(all_posts)):
            final_result = post_outputs[i].outputs[0].text.strip()
            final_result = strip_generation_preamble(final_result)
            if len(final_result) < 10:
                final_result = "ALERT ALERT FAILURE ALERT ALERT"
            all_posts[i] = final_result
            post_tokens[i] += len(post_outputs[i].outputs[0].token_ids)
            post_input_tokens[i] += len(tokenizer.encode(post_conversations[i][1]["content"]))

        # ---- Stage 2: generate two candidate summaries per post (A and B) ----
        # System message is explicit that the task is summarization only, NOT answering
        # any question or request for advice contained within the post itself. Posts are
        # often phrased as questions/advice-requests (per the persona prompt above), and
        # without this constraint the model tends to answer the post rather than summarize it.
        summary_system_message = (
            "You are a helpful assistant who is an expert at summarizing content into TL;DR sentences. "
            "Summarize the following text. Do not answer any questions posed in the text, do not give advice, "
            "and do not respond to the author. Only produce a concise summary (2-3 sentences) of what the "
            "text says."
        )

        summary_conversations = []
        for i in range(len(all_posts)):
            summary_conversation = [
                {
                    "role": "system",
                    "content": summary_system_message
                },
                {
                    "role": "user",
                    "content": all_posts[i]
                },
            ]
            summary_conversations.append(summary_conversation)

        rand_seed_a = np.random.randint(1001, 2000)
        smp_a = SamplingParams(n=1, min_tokens=10, max_tokens=256, temperature=1, seed=rand_seed_a, stop=stopwords, repetition_penalty=1.1, top_p=0.9)
        outputs_a = llm.chat(summary_conversations, sampling_params=smp_a, use_tqdm=True)
        time.sleep(1)

        rand_seed_b = np.random.randint(2001, 3000)
        smp_b = SamplingParams(n=1, min_tokens=10, max_tokens=256, temperature=1, seed=rand_seed_b, stop=stopwords, repetition_penalty=1.1, top_p=0.9)
        outputs_b = llm.chat(summary_conversations, sampling_params=smp_b, use_tqdm=True)
        time.sleep(1)

        a_summaries = []
        b_summaries = []
        for i in range(len(all_posts)):
            summary_a = strip_generation_preamble(outputs_a[i].outputs[0].text.strip())
            summary_b = strip_generation_preamble(outputs_b[i].outputs[0].text.strip())
            a_summaries.append(summary_a)
            b_summaries.append(summary_b)
            summary_input_tokens[i] += len(tokenizer.encode(summary_conversations[i][1]["content"])) * 2  # same post encoded for both A and B calls
            summary_tokens[i] += len(outputs_a[i].outputs[0].token_ids) + len(outputs_b[i].outputs[0].token_ids)

        for i in range(len(information)):
            total_input_tokens_i = post_input_tokens[i] + summary_input_tokens[i]
            total_output_tokens_i = post_tokens[i] + summary_tokens[i]
            information[i] += (
                "\nTotal Input Tokens: " + str(total_input_tokens_i)
                + "\nTotal Output Tokens: " + str(total_output_tokens_i)
                + "\n"
            )

        with open("sample9.jsonl", "a", encoding="utf-8") as f:
            for i in range(len(all_posts)):
                json_entry = {
                    "info": information[i] + f"\nGPU id: {args.gpu_id}",
                    "post": all_posts[i],
                    "summary_A": a_summaries[i],
                    "summary_B": b_summaries[i]
                }
                f.write(json.dumps(json_entry) + "\n")

        with open("batch_posts.txt", "a", encoding="utf-8") as f:
            for post in all_posts:
                f.write(post + "\nNEXT POST\n")

    print("Done with all batches.")

if __name__ == "__main__":
    main()