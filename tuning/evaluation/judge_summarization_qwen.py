import argparse
import json
import random
import re

from tqdm import tqdm
from vllm import LLM, SamplingParams

MAX_MODEL_LEN = 8192
MAX_OUTPUT_TOKENS = 4096
PROMPT_TOKEN_BUDGET = MAX_MODEL_LEN - MAX_OUTPUT_TOKENS - 64
MODEL_NAMES = ("model_1", "model_2")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen3.8-27B")
    parser.add_argument("--trained_file", type=str, required=True, help="JSONL file of trained model summaries")
    parser.add_argument("--base_file", type=str, required=True, help="JSONL file of base model summaries")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_file", type=str, required=True,
                        help="JSONL file the win-rate summary line is appended to")
    parser.add_argument("--raw_output_file", type=str, required=True,
                        help="JSONL file for the per-example judge outputs")
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--enable_thinking", action="store_true",
                        help="Enable Qwen3 thinking mode (risk: truncation before JSON at 4096 output tokens)")
    return parser.parse_args()


def parse_chosen_field(chosen):
    marker = "Assistant:"
    idx = chosen.rfind(marker)
    if idx == -1:
        raise ValueError("No 'Assistant:' found in chosen field")
    post_prompt = chosen[:idx + len(marker)]
    summary = chosen[idx + len(marker):]
    return post_prompt, summary


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(l) for l in f]


def load_generation_file(path):
    raw = load_jsonl(path)
    result = []
    for entry in raw:
        if "conversation" in entry and "response" in entry:
            result.append({"post_prompt": entry["conversation"], "summary": entry["response"]})
        else:
            post_prompt, summary = parse_chosen_field(entry["chosen"])
            result.append({"post_prompt": post_prompt, "summary": summary})
    return result


def _rank_of(value):
    if isinstance(value, dict):
        return int(value["rank"])
    return int(value)


def _format_violation_of(value):
    if isinstance(value, dict):
        return bool(value.get("format_violation", False))
    return None


def _name_as_rank(value):
    if isinstance(value, dict):
        value = value.get("rank")
    if isinstance(value, str) and value.strip() in MODEL_NAMES:
        return value.strip()
    return None


def _top_level_json_blocks(text):
    candidates = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start:i + 1])
    return candidates


def strip_thinking(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text


def extract_verdict(text):
    text = strip_thinking(text).strip()

    for json_str in reversed(_top_level_json_blocks(text)):
        try:
            data = json.loads(json_str)
        except Exception:
            continue
        if not (isinstance(data, dict) and "model_1" in data and "model_2" in data):
            continue

        fv1 = _format_violation_of(data["model_1"])
        fv2 = _format_violation_of(data["model_2"])

        named = _name_as_rank(data["model_1"])
        if named is not None:
            return ("A" if named == "model_1" else "B"), fv1, fv2

        try:
            m1_rank = _rank_of(data["model_1"])
            m2_rank = _rank_of(data["model_2"])
        except Exception:
            continue

        if m1_rank == 1:
            return "A", fv1, fv2
        elif m2_rank == 1:
            return "B", fv1, fv2

    matches = list(re.finditer(r'"(model_[12])"\s*:\s*(?:\{\s*"rank"\s*:\s*)?(\d+)', text))
    for match in reversed(matches):
        if match.group(2) == "1":
            return ("A" if match.group(1) == "model_1" else "B"), None, None

    match = re.search(r'"model_1"\s*:\s*(?:\{\s*"rank"\s*:\s*)?"(model_[12])"', text)
    if match:
        return ("A" if match.group(1) == "model_1" else "B"), None, None

    return "Invalid", None, None


def build_judge_input(post, summary_a, summary_b):
    return [
        {
            "role": "system",
            "content": "You are comparing the quality of two summaries of a given text.",
        },
        {
            "role": "user",
            "content": f"""Which of the following summaries does a better job of summarizing the most important points in the given forum post, without including unimportant or irrelevant details? A good summary is precise, concise, and contains ONLY the summary itself.

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

First, for each summary, check whether it contains exposed reasoning, meta-commentary, draft/false-start attempts, or duplicated passages, and check whether it is longer than the post. Then, for each summary, list its distinct factual claims and mark each as explicitly stated, reasonably implied, or unsupported. Then weigh the two summaries against each other along coverage, accuracy, coherence, and conciseness, applying the gating rule above. Then decide which is better overall, so that the model with rank 1 has the better summary. Then return a
dictionary of the form

{{
"reasoning": short justification of ranking,
"model_1": {{"rank": rank_1, "format_violation": true_or_false}},
"model_2": {{"rank": rank_2, "format_violation": true_or_false}}
}}

Do not include any reasoning or explanation outside the JSON. Do not include any text before or after the JSON. Your entire response must be parseable by json.loads().
Please provide the ranking that the majority of careful, informed readers would give if they cared only about informational quality, not writing style.
""",
        },
    ]


def judge_responses(judge_model, trained_data, base_data, batch_size, tensor_parallel_size, raw_output_file, enable_thinking):
    if len(trained_data) != len(base_data):
        raise ValueError(
            f"Mismatched lengths: trained_file has {len(trained_data)} rows, "
            f"base_file has {len(base_data)} rows."
        )
    for i, (t, b) in enumerate(zip(trained_data, base_data)):
        if t["post_prompt"] != b["post_prompt"]:
            raise ValueError(f"Post mismatch at row {i}. Ensure both files were generated from the same input.")

    rng = random.Random(42)
    positions = ["A" if rng.random() < 0.5 else "B" for _ in range(len(trained_data))]

    print(f"\nLoading judge model: {judge_model} (TP={tensor_parallel_size})")
    judge_llm = LLM(
        model=judge_model,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=MAX_MODEL_LEN,
    )
    tokenizer = judge_llm.get_tokenizer()
    chat_kwargs = {"enable_thinking": enable_thinking}

    judge_sampling = SamplingParams(
        n=1,
        min_tokens=10,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.6 if enable_thinking else 0.7,
        top_p=0.95 if enable_thinking else 0.8,
        top_k=20,
    )

    trained_wins = 0
    total = 0
    oversized_count = 0
    invalid_count = 0
    all_results = []

    for start in tqdm(range(0, len(trained_data), batch_size)):
        batch_trained = trained_data[start : start + batch_size]
        batch_base = base_data[start : start + batch_size]

        fit_inputs = []
        fit_meta = []
        oversized_meta = []

        for j, (t, b) in enumerate(zip(batch_trained, batch_base)):
            idx = start + j
            post_prompt = t["post_prompt"]
            t_summary = t["summary"]
            b_summary = b["summary"]

            trained_is = positions[idx]
            if trained_is == "A":
                summary_a, summary_b = t_summary, b_summary
            else:
                summary_a, summary_b = b_summary, t_summary

            messages = build_judge_input(post_prompt, summary_a, summary_b)

            token_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, **chat_kwargs
            )
            n_tokens = len(token_ids)

            meta = (idx, t, b, trained_is, post_prompt)
            if n_tokens > PROMPT_TOKEN_BUDGET:
                oversized_meta.append(meta)
            else:
                fit_inputs.append(messages)
                fit_meta.append(meta)

        for idx, t, b, trained_is, post_prompt in oversized_meta:
            oversized_count += 1
            total += 1
            all_results.append({
                "idx": idx,
                "post_prompt": post_prompt,
                "trained_summary": t["summary"],
                "base_summary": b["summary"],
                "verdict": "Oversized",
                "trained_position": trained_is,
                "trained_won": False,
                "trained_format_violation": None,
                "base_format_violation": None,
                "raw_judge_response": None,
            })
        if oversized_meta:
            print(f"  Skipped {len(oversized_meta)} oversized prompt(s) this batch "
                  f"(counted as a loss, not sent to the judge).")

        if not fit_inputs:
            continue

        batch_outputs = judge_llm.chat(
            fit_inputs, sampling_params=judge_sampling, chat_template_kwargs=chat_kwargs
        )

        for i, output in enumerate(batch_outputs):
            idx, t, b, trained_is, post_prompt = fit_meta[i]
            response_text = output.outputs[0].text
            verdict, fv1, fv2 = extract_verdict(response_text)
            won = verdict == trained_is
            if verdict == "Invalid":
                invalid_count += 1
                print(f"Invalid response at idx {idx}")

            if won:
                trained_wins += 1
            total += 1

            if trained_is == "A":
                trained_fv, base_fv = fv1, fv2
            else:
                trained_fv, base_fv = fv2, fv1

            all_results.append({
                "idx": idx,
                "post_prompt": post_prompt,
                "trained_summary": t["summary"],
                "base_summary": b["summary"],
                "verdict": verdict,
                "trained_position": trained_is,
                "trained_won": won,
                "trained_format_violation": trained_fv,
                "base_format_violation": base_fv,
                "raw_judge_response": response_text,
            })

    print(f"\nTotal oversized prompts skipped (counted as losses): {oversized_count}")
    print(f"Total invalid (unparseable) judge responses: {invalid_count}")

    all_results.sort(key=lambda r: r["idx"])

    with open(raw_output_file, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote raw outputs to {raw_output_file}")

    return trained_wins, total


def main():
    args = parse_args()

    trained_data = load_generation_file(args.trained_file)
    base_data = load_generation_file(args.base_file)

    print(args.trained_file)

    trained_wins, total = judge_responses(
        args.judge_model,
        trained_data,
        base_data,
        args.batch_size,
        args.tensor_parallel_size,
        args.raw_output_file,
        args.enable_thinking,
    )

    final_win_rate = trained_wins / total if total else 0

    print("\n==============================")
    print(f"Trained model wins: {trained_wins}")
    print(f"Total judgments:    {total}")
    print(f"Final win rate:     {final_win_rate:.4f}")
    print("==============================")

    with open(args.output_file, "a") as f:
        f.write(json.dumps({
            "judge_model": args.judge_model,
            "trained_model": args.trained_file,
            "trained_wins": trained_wins,
            "total": total,
            "win_rate": final_win_rate,
        }) + "\n")


if __name__ == "__main__":
    main()
