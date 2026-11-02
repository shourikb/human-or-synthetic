"""
Builds a human-annotation JSONL from a reference (test) file and a
generated-responses file, for the Summarization task. For each pair,
randomly assigns the reference summary and the model-generated summary to
"A"/"B", fills in the summarization judging prompt template, and records
which side the reference summary ended up on.

Assumes:
  - REFERENCE_FILE has one JSON object per line with a "chosen" field
    formatted as "Human: Summarize the following text\n\n{post}\n\nAssistant: {reference_summary}"
    (matching your pt-reddit.jsonl / synthetic chosen/rejected format).
  - GENERATED_FILE has one JSON object per line with "conversation" and
    "response" fields (matching generate_responses.py's output format),
    in the same order/alignment as REFERENCE_FILE.

Usage:
    python build_human_annotation_file.py
"""
import json
import random

REFERENCE_FILE = "summarization_validation.jsonl"
GENERATED_FILE = "../seed-validation/generated_summary.jsonl"
OUTPUT_FILE = "summarization_annotation_50.jsonl"
SEED = 55

PROMPT_TEMPLATE = """Which of the following summaries does a better job of summarizing the most important points in the given forum post, without including unimportant or irrelevant details? A good summary is precise, concise, and contains ONLY the summary itself.

Judge each summary along these dimensions, the way a careful reader would:

- Coverage: whether the summary conveys the post's central point(s) and any details essential to understanding the poster's situation or question. Enumerating additional true but secondary details does not by itself improve coverage — a summary that nails the core situation in one or two sentences should not lose to one that lists more true statements unless those statements are necessary to understand the post.

- Accuracy: to what degree the statements in the summary are actually stated or reasonably implied in the post. Before scoring this dimension, list every distinct factual claim each summary makes, and for each one mark whether it is explicitly stated, reasonably implied, or NOT supported by the post. Any claim not supported by the post is a hallucination and should weigh heavily against that summary's accuracy score, regardless of how plausible or well-written it sounds. A summary that is coherent but contains unsupported claims should score lower on accuracy than a summary that is less detailed but fully grounded.

- Coherence: how easy the summary is to read and understand on its own, independent of the post.

- Conciseness: whether the summary is meaningfully shorter than the post and free of padding, repetition, or restated points. This includes internal redundancy — restating the same underlying point multiple times in different phrasing (e.g. "X feels conflicted... X expresses concern... X's worry stems from...") counts against conciseness even if the summary stays shorter than the source post. A summary that is as long as or longer than the post has failed at conciseness regardless of how much it covers.

- Format adherence: whether the response contains ONLY a final summary, with no exposed reasoning, planning, meta-commentary about the task (e.g. "Okay, I need to..."), draft/false-start attempts, or duplicated passages. Any of these is a serious violation, not a minor stylistic issue — a response that leaks its own reasoning process is not a valid summary even if a summary appears somewhere within it.

Treat format adherence and conciseness as gating: a response with exposed reasoning, restarted/duplicated content, or length exceeding the source post should lose to a shorter, well-formatted response even if the shorter response covers somewhat less. Only weigh coverage/accuracy/coherence as tiebreakers between two responses that both pass the format and conciseness bar.

Place less weight on polished sentence fluidity for its own sake, and no weight on whether the summary captures the poster's tone, voice, or reads like something a human would write — focus entirely on whether it faithfully and efficiently conveys the important information in the post. Do not penalize or reward either summary for its narrative voice (first-person vs. third-person) or formality — judge informational quality only.

Post: {post}

Summaries:
{{
'model': 'model_1',
'summary': {summary_a}
}}
{{
'model': 'model_2',
'summary': {summary_b}
}}

First, for each summary, check whether it contains exposed reasoning, meta-commentary, draft/false-start attempts, or duplicated passages, and check whether it is longer than the post. Then, for each summary, list its distinct factual claims and mark each as explicitly stated, reasonably implied, or unsupported. Then weigh the two summaries against each other along coverage, accuracy, coherence, and conciseness, applying the gating rule above. Then decide which is better overall, so that the model with rank 1 has the better summary. Then return a dictionary of the form

{{
"reasoning": short justification of ranking,
"model_1": {{"rank": rank_1, "format_violation": true_or_false}},
"model_2": {{"rank": rank_2, "format_violation": true_or_false}}
}}

Do not include any reasoning or explanation outside the JSON. Do not include any text before or after the JSON. Your entire response must be parseable by json.loads(). Please provide the ranking that the majority of careful, informed readers would give if they cared only about informational quality, not writing style."""


def split_post_and_reference(chosen_text):
    """Split a 'chosen' field formatted as
    'Human: Summarize the following text\n\n{post}\n\nAssistant: {summary}'
    into (post, reference_summary)."""
    split_tag = "\n\nAssistant: "
    idx = chosen_text.rfind(split_tag)
    if idx == -1:
        raise ValueError(f"No 'Assistant:' tag found in: {chosen_text[:200]}")

    prefix = "Human: Summarize the following text\n\n"
    post = chosen_text[len(prefix): idx]
    reference_summary = chosen_text[idx + len(split_tag):].strip()
    return post, reference_summary


def main():
    random.seed(SEED)

    with open(REFERENCE_FILE, "r") as f:
        reference_rows = [json.loads(line) for line in f if line.strip()]

    with open(GENERATED_FILE, "r") as f:
        generated_rows = [json.loads(line) for line in f if line.strip()]

    if len(reference_rows) != len(generated_rows):
        raise ValueError(
            f"Row count mismatch: {REFERENCE_FILE} has {len(reference_rows)}, "
            f"{GENERATED_FILE} has {len(generated_rows)}."
        )

    written = 0
    with open(OUTPUT_FILE, "w") as out:
        for ref_row, gen_row in zip(reference_rows, generated_rows):
            post, reference_summary = split_post_and_reference(ref_row["chosen"])
            generated_summary = gen_row["response"]

            # Randomize which side the reference summary lands on.
            if random.random() < 0.5:
                summary_a, summary_b = reference_summary, generated_summary
                reference_position = "a"
            else:
                summary_a, summary_b = generated_summary, reference_summary
                reference_position = "b"

            prompt = PROMPT_TEMPLATE.format(
                post=post,
                summary_a=json.dumps(summary_a),
                summary_b=json.dumps(summary_b),
            )

            out.write(json.dumps({
                "prompt": prompt,
                "reference_position": reference_position,
            }) + "\n")
            written += 1

    print(f"Wrote {written} annotation items to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()