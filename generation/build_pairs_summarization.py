import json
from collections import Counter
from transformers import AutoTokenizer

file1_path = "finalposts.jsonl"  # contains "info", "post", "summary_A", "summary_B"
file2_path = "majority_votes_110448_summarization_merged_offset0.jsonl"  # contains "judgments" (list of {"answer": ...})
output_path = "full_summary_synthetic_data3.jsonl"

PROMPT_PREFIX = "Human: Summarize the following text\n\n"
RESPONSE_PREFIX = "\n\nAssistant: "

model_name = "allenai/Llama-3.1-Tulu-3-8B-SFT"
tokenizer = AutoTokenizer.from_pretrained(model_name)

total_read = 0
total_skipped = 0
total_ties = 0
total_chosen_tokens = 0

with open(file1_path, "r") as f1, open(file2_path, "r") as f2, open(output_path, "w") as out:
    for line1, line2 in zip(f1, f2):

        data1 = json.loads(line1)
        data2 = json.loads(line2)

        judgements = data2.get("judgments", [])
        vote_counts = Counter()

        for judgement in judgements:
            vote = judgement.get("answer", "")
            if vote:
                vote_counts[vote.strip().lower()] += 1

        votes_a = vote_counts.get("a", 0)
        votes_b = vote_counts.get("b", 0)

        if votes_a == 0 and votes_b == 0:
            print(f"Missing answer choice on line {total_read + 1}, skipping line.")
            total_skipped += 1
            continue

        if votes_a == votes_b:
            print(f"Tie on line {total_read + 1}, {votes_a} to {votes_b}, skipping.")
            total_ties += 1
            continue
        elif votes_a > votes_b:
            answer = "a"
        else:
            answer = "b"

        # Prepare new entry
        post = data1.get("post", "")
        summary_a = data1.get("summary_A", "")
        summary_b = data1.get("summary_B", "")

        prompt = PROMPT_PREFIX + post

        new_entry = {}
        if answer == "a":
            new_entry["chosen"] = prompt + RESPONSE_PREFIX + summary_a
            new_entry["rejected"] = prompt + RESPONSE_PREFIX + summary_b
        else:
            new_entry["chosen"] = prompt + RESPONSE_PREFIX + summary_b
            new_entry["rejected"] = prompt + RESPONSE_PREFIX + summary_a

        out.write(json.dumps(new_entry) + "\n")
        total_read += 1
        total_chosen_tokens += len(tokenizer.encode(new_entry["chosen"]))

print(f"  Written:  {total_read}")
print(f"  Skipped:  {total_skipped}")
print(f"  Ties:     {total_ties}")
print(f"Combined file written to {output_path}")

avg_chosen_tokens = total_chosen_tokens / total_read if total_read else 0
print(f"Average tokens per chosen: {avg_chosen_tokens:.2f}")