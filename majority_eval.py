import json
from collections import Counter, defaultdict

def evaluate_jsonl(path):
    total = 0
    correct = 0

    # Track agreement counts for correct/incorrect
    agreement_stats = {
        "correct": defaultdict(int),
        "incorrect": defaultdict(int)
    }

    with open(path, "r") as f:
        for line in f:
            total += 1
            data = json.loads(line)

            # Collect answers from judgments
            answers = [j["answer"] for j in data.get("judgments", [])]

            if not answers:
                continue  # skip if no judgments

            # Majority vote
            counts = Counter(answers)
            majority_answer, majority_count = counts.most_common(1)[0]

            # Compare with correct answer
            is_correct = majority_answer == data.get("correct_answer")
            if is_correct:
                correct += 1

            # Count how many agreed with majority
            num_agree = sum(1 for a in answers if a == majority_answer)

            key = "correct" if is_correct else "incorrect"
            agreement_stats[key][num_agree] += 1

    ratio = correct / total if total > 0 else 0.0

    # Sort the dictionaries by number of agreements (descending)
    sorted_correct = dict(sorted(agreement_stats["correct"].items(), key=lambda x: x[0], reverse=True))
    sorted_incorrect = dict(sorted(agreement_stats["incorrect"].items(), key=lambda x: x[0], reverse=True))

    print(f"Total lines: {total}")
    print(f"Total correct: {correct}")
    print(f"Accuracy: {ratio:.4f}")
    print("\nAgreement breakdown (Correct predictions):", sorted_correct)
    print("Agreement breakdown (Incorrect predictions):", sorted_incorrect)

    return correct, total, ratio, agreement_stats

# Example usage
evaluate_jsonl("majority_votes_32_tuned.jsonl")
