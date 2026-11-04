import argparse
import re

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from generation.utils import load_personas, reset_seeds, set_visible_gpus, write_jsonl

PREAMBLE_PATTERNS = [
    r"^certainly!?\s*",
    r"^sure!?\s*",
    r"^of course!?\s*",
    r"^here'?s? (a |an |another )?(fresh |unique |new )?reddit[\s-]style post.*?:?\s*",
    r"^here'?s? a post.*?:?\s*",
    r"^title:\s*.*?\n",
    r"^---+\s*",
]

SAMPLE_POST = """A little background first. Like most engineers I'm taking differential equations and also solid mechanics. For solid mechanics we have to do a lab in addition to taking the class. Pretty standard stuff.

Anyway, the class had us sign up for labs in groups by providing sign up sheets where we could sign up for which lab we wanted by providing our name and student ID numbers. The lab assignments were posted online today with a column for our names and student ID numbers (which had been blacked out). Somehow, my name wasn't on the list and in frustration I began dragging the scroll bar up and down frantically hoping that I'd magically see my name; that's when I saw it.

When dragging the scroll bar fast enough, I found that the blacked out student ID numbers would temporarily become visible. At first I thought, "No big deal, what could someone do with a person's ID number." Then it hit me. My university uses a web based homework website (WebWork) that initially allows students to access their account by inputting their user ID number as their user name and password. What's worse is that this site NEVER PROMPTS THE STUDENT TO CREATE A NEW PASSWORD. As such, most students never change their password from their student ID number.

Realizing that someone who happened upon this discovery could in turn use it to access other student's homework assignments and tamper with them, I contacted the administrator in charge of assigning lab schedules. In doing so, I not only signed up for a lab group but had them completely remove the ID numbers from the list.

Best part is that nobody in my class will know. That's why I wanted to tell you reddit."""

POST_SYSTEM = "Generate a Reddit-style post as if you were the persona given. Output ONLY the raw post text. Never include any preamble, acknowledgment, introduction, or meta-commentary such as 'Certainly! Here's a post...' or 'Title:' or '---'. Your entire response must be the post itself and nothing else."

POST_TEMPLATE = """You are a {persona}. Write a Reddit-style post describing a situation, story, or question you'd want summarized by someone else. The post should be open ended, sharing a personal story, asking for advice, or describing an event in detail, similar to a real Reddit post. Below is a sample post for style and length reference:

                    === SAMPLE POST ===
                    {sample}
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
                    """

SUMMARY_SYSTEM = (
    "You are a helpful assistant who is an expert at summarizing content into TL;DR sentences. "
    "Summarize the following text. Do not answer any questions posed in the text, do not give advice, "
    "and do not respond to the author. Only produce a concise summary (2-3 sentences) of what the "
    "text says."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate persona-conditioned Reddit-style posts with two candidate summaries")
    parser.add_argument("--personas_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--posts_file", type=str, default=None,
                        help="Optional text file to append each generated post to")
    parser.add_argument("--model", type=str, default="allenai/Llama-3.1-Tulu-3-8B-SFT")
    parser.add_argument("--gpu_id", type=str, default="0", help="Comma-separated GPU ids")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--batch_start", type=int, default=0)
    parser.add_argument("--batch_count", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--post_max_tokens", type=int, default=2048)
    parser.add_argument("--summary_max_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=None,
                        help="Base seed; batch b uses seed + b. Random per batch if omitted")
    return parser.parse_args()


def strip_generation_preamble(text):
    text = text.strip()
    changed = True
    while changed:
        changed = False
        for pattern in PREAMBLE_PATTERNS:
            new_text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
            if new_text != text:
                text = new_text.strip()
                changed = True
    text = re.sub(r"\n*---+\s*$", "", text).strip()
    text = re.sub(r"\n*(feel free to .*|this post (is|was|embraces).*)$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    return text


def make_sampling_params(seed, max_tokens):
    return SamplingParams(n=1, min_tokens=10, max_tokens=max_tokens, temperature=1, seed=seed,
                          repetition_penalty=1.1, top_p=0.9)


def main():
    args = parse_args()
    set_visible_gpus(args.gpu_id)

    all_personas = load_personas(args.personas_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_memory_utilization)

    for batch in range(args.batch_start, args.batch_start + args.batch_count):
        reset_seeds(None if args.seed is None else args.seed + batch)
        n = args.batch_size
        personas = list(np.random.choice(all_personas, n, replace=False))

        post_prompts = [
            [
                {"role": "system", "content": POST_SYSTEM},
                {"role": "user", "content": POST_TEMPLATE.format(persona=persona, sample=SAMPLE_POST)},
            ]
            for persona in personas
        ]
        post_params = make_sampling_params(np.random.randint(1, 1000), args.post_max_tokens)
        post_outputs = llm.chat(post_prompts, sampling_params=post_params, use_tqdm=True)

        posts = []
        for output in post_outputs:
            post = strip_generation_preamble(output.outputs[0].text.strip())
            posts.append(post if len(post) >= 10 else "ALERT ALERT FAILURE ALERT ALERT")

        summary_prompts = [
            [{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": post}]
            for post in posts
        ]
        params_a = make_sampling_params(np.random.randint(1001, 2000), args.summary_max_tokens)
        outputs_a = llm.chat(summary_prompts, sampling_params=params_a, use_tqdm=True)
        params_b = make_sampling_params(np.random.randint(2001, 3000), args.summary_max_tokens)
        outputs_b = llm.chat(summary_prompts, sampling_params=params_b, use_tqdm=True)

        records = []
        for i in range(n):
            post_input = len(tokenizer.encode(post_prompts[i][1]["content"]))
            summary_input = 2 * len(tokenizer.encode(posts[i]))
            output_total = (len(post_outputs[i].outputs[0].token_ids)
                            + len(outputs_a[i].outputs[0].token_ids)
                            + len(outputs_b[i].outputs[0].token_ids))
            info = (f"\nPersona: {personas[i]}"
                    f"\nTotal Input Tokens: {post_input + summary_input}\nTotal Output Tokens: {output_total}\n")
            records.append({
                "info": info,
                "post": posts[i],
                "summary_A": strip_generation_preamble(outputs_a[i].outputs[0].text.strip()),
                "summary_B": strip_generation_preamble(outputs_b[i].outputs[0].text.strip()),
            })
        write_jsonl(args.output_file, records, mode="a")

        if args.posts_file:
            with open(args.posts_file, "a", encoding="utf-8") as f:
                for post in posts:
                    f.write(post + "\nNEXT POST\n")

        print(f"Finished batch {batch}")


if __name__ == "__main__":
    main()
