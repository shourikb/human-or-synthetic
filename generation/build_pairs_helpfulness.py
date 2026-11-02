import json
from collections import Counter

file1_path = "finalconvos3.jsonl"  # contains "conversation", "response_A", "response_B"
file2_path = "majority_votes_all_synthetic_merged.jsonl"  # contains "normal answer"
output_path = "full_synthetic_data.jsonl"

total_read = 0
total_skipped = 0
total_ties = 0
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

        answer = ""
        if votes_a == votes_b:
            print(f"Tie on line {total_read + 1}, {votes_a} to {votes_b}, skipping.")
            total_ties += 1
            continue
        elif votes_a > votes_b:
            answer = "a"
        else:
            answer = "b"

        # Prepare new entry
        new_entry = {}
        conversation = data1.get("conversation", "")

        if answer == "a":
            new_entry["chosen"] = conversation + " " + data1.get("response_A", "")
            new_entry["rejected"] = conversation + " " + data1.get("response_B", "")
        elif answer == "b":
            new_entry["chosen"] = conversation + " " + data1.get("response_B", "")
            new_entry["rejected"] = conversation + " " + data1.get("response_A", "")
        else:
            # Skip if answer invalid or missing
            print("Missing answer choice, skipping line.")
            continue

        out.write(json.dumps(new_entry) + "\n")
        total_read += 1

print(f"  Written:  {total_read}")
print(f"  Skipped:  {total_skipped}")
print(f"  Ties:     {total_ties}")
print(f"Combined file written to {output_path}")
