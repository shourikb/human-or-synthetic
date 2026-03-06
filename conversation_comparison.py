# DEPRECATED - see bin_filtering.py for updated version with more features and better structure


import matplotlib.pyplot as plt
import json
import re
from collections import Counter
from transformers import AutoTokenizer
from datasets import load_dataset


BIN_SIZE = 50

model_name =  "allenai/Llama-3.1-Tulu-3-8B-SFT" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained("tokenizer_tmp")

def get_bin(val, bin_size):
    return (val // bin_size) * bin_size

def tokenize(s):
    return tokenizer.encode(s.lower())

def count_distinct_tokens(s):
    tokens = tokenize(s)
    return len(set(tokens))

def bigrams(s):
    tokens = tokenize(s)
    bigrams = zip(tokens, tokens[1:])  # create consecutive pairs
    return bigrams

def compute_averages_from_jsonl(path, use_combined=False, return_selected=False):
    if use_combined:
        desired_total = 55000
        target_tulu_bins = {b: int(anthropic_probs.get(b, 0) * desired_total) for b in anthropic_bins}
    else:
        target_tulu_bins = None
    
    selected_conversations = []
    tulu_bins_used = Counter()
    total_token_counts = 0
    total_unique_bigram_counts = 0
    total_convo_length = 0
    total_turns = 0
    convo_lengths = []
    count = 0
    global_unique_words = set()
    total_bigrams = 0
    with open(path, 'r') as f:
        for line in f:
            if len(convo_lengths) > 55000 and not use_combined:
                break
            item = json.loads(line)
            if use_combined:
                text = item['conversation'] + " " + item['response_A']
            else:
                text = item['chosen']
            
            convo_len = len(tokenize(text))
            bin_id = get_bin(convo_len, BIN_SIZE)

            if use_combined and tulu_bins_used[bin_id] >= target_tulu_bins.get(bin_id, 0):
                continue
            
            tulu_bins_used[bin_id] += 1
            if return_selected:
                selected_conversations.append(item)

            convo_lengths.append(convo_len)
            total_token_counts += count_distinct_tokens(text)
            total_unique_bigram_counts += len(set(bigrams(text)))
            total_bigrams += len(list(bigrams(text)))
            total_convo_length += convo_len
            total_turns += text.count("Human:")
            global_unique_words.update(tokenize(text))
            count += 1
    avg_tokens = total_token_counts / count if count else 0
    avg_bigrams = total_unique_bigram_counts / count if count else 0
    avg_convo_length = total_convo_length / count if count else 0
    avg_turn_length = total_turns / count if count else 0
    word_to_token_ratio = len(global_unique_words) / total_convo_length if total_convo_length else 0
    unique_bigram_to_total_bigram_ratio = total_unique_bigram_counts / total_bigrams if total_bigrams else 0

    if return_selected:
        return avg_tokens, avg_bigrams, avg_convo_length, avg_turn_length, convo_lengths, word_to_token_ratio, unique_bigram_to_total_bigram_ratio, selected_conversations
    else:
        return avg_tokens, avg_bigrams, avg_convo_length, avg_turn_length, convo_lengths, word_to_token_ratio, unique_bigram_to_total_bigram_ratio

def compute_shp_stats(use_prompt_only=False):
    """
    Computes statistics for the Stanford Human Preferences (SHP) dataset.
    
    SHP format:
        {
            "history": "<prompt text>",
            "human_ref_A": "<response A>",
            "human_ref_B": "<response B>",
            "score": 1  (means A is preferred; 0 means B is preferred)
        }
    """
    dataset = load_dataset("stanfordnlp/SHP", split="train")

    convo_lengths = []
    total_token_counts = 0
    total_unique_bigram_counts = 0
    total_bigrams = 0
    total_turns = 0
    global_unique_words = set()
    count = 0

    for item in dataset:
        prompt = item["history"]
        resp_A = item["human_ref_A"]
        resp_B = item["human_ref_B"]

        # Build text depending on preference
        # Prefer "text that forms the preference pair" similar to Anthropic
        if use_prompt_only:
            text = prompt
        else:
            # Use both responses to match preference-pair structure
            text = prompt + "\n" + resp_A + "\n" + resp_B

        # Tokenize
        tokens = tokenize(text)
        convo_len = len(tokens)
        convo_lengths.append(convo_len)

        # Token/bigram stats
        total_token_counts += len(set(tokens))

        bgs = list(bigrams(text))
        total_unique_bigram_counts += len(set(bgs))
        total_bigrams += len(bgs)

        total_turns += text.count("\n")
        global_unique_words.update(tokens)
        count += 1

    avg_tokens = total_token_counts / count
    avg_bigrams = total_unique_bigram_counts / count
    avg_convo_length = sum(convo_lengths) / count
    avg_turn_length = total_turns / count
    word_to_token_ratio = len(global_unique_words) / sum(convo_lengths)
    ubtb_ratio = total_unique_bigram_counts / total_bigrams

    return {
        "avg_distinct_tokens": avg_tokens,
        "avg_distinct_bigrams": avg_bigrams,
        "avg_conversation_length": avg_convo_length,
        "avg_turn_length": avg_turn_length,
        "word_to_token_ratio": word_to_token_ratio,
        "unique_bigram_to_total_bigram_ratio": ubtb_ratio,
        "convo_lengths": convo_lengths
    }

# File paths
file1_path = 'anthropic.jsonl'  # just 'chosen'
file2_path = 'filtered_sample5.jsonl'  # 'chosen' + 'response_a'

# Compute
avg_tokens_1, avg_bigrams_1, avg_convos_1, avg_turn_1, convo_lengths_1, wtr_1, ubtb_1 = compute_averages_from_jsonl(file1_path, use_combined=False)

anthropic_bins = Counter()
for length in convo_lengths_1:
    b = get_bin(length, BIN_SIZE)
    anthropic_bins[b] += 1

total_anthropic = sum(anthropic_bins.values())
anthropic_probs = {k: v / total_anthropic for k, v in anthropic_bins.items()}

print("Anthropic bin counts:", anthropic_bins)

avg_tokens_2, avg_bigrams_2, avg_convos_2, avg_turn_2, convo_lengths_2, wtr_2, ubtb_2, selected_convos_2 = compute_averages_from_jsonl(file2_path, use_combined=True, return_selected=True)

# Print results
print("=== File 1 (Anthropic) ===")
print(f"Avg. distinct tokens:       {avg_tokens_1:.2f}")
print(f"Avg. distinct bigrams:      {avg_bigrams_1:.2f}")
print(f"Avg. conversation length:   {avg_convos_1:.2f} tokens")
print(f"Avg. turn length: {avg_turn_1:.2f}")
print(f"Unique words to total tokens: {wtr_1:.2f}")
print(f"Unique bigrams to total bigrams: {ubtb_1:.2f}")
print(f"Conversations used: {len(convo_lengths_1)}")

print("\n=== File 2 (Tulu) ===")
print(f"Avg. distinct tokens:       {avg_tokens_2:.2f}")
print(f"Avg. distinct bigrams:      {avg_bigrams_2:.2f}")
print(f"Avg. conversation length:   {avg_convos_2:.2f} tokens")
print(f"Avg. turn length: {avg_turn_2:.2f}")
print(f"Unique words to total tokens: {wtr_2:.2f}")
print(f"Unique bigrams to total bigrams: {ubtb_2:.2f}")
print(f"Conversations used: {len(convo_lengths_2)}")

# === Plot for File 1 (Anthropic) ===
plt.figure(figsize=(8, 5))
bins1 = range(0, max(convo_lengths_1) + 50, 50)

plt.hist(convo_lengths_1, bins=bins1, color='blue', alpha=0.7)
plt.xlabel('Conversation Length (tokens)')
plt.ylabel('Number of Conversations')
plt.title('Histogram of Conversation Token Lengths\nFile 1 (Anthropic)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("histogram_anthropic.png")
plt.close()

# === Plot for File 2 (Tulu) ===
plt.figure(figsize=(8, 5))
bins2 = range(0, max(convo_lengths_2) + 50, 50)

plt.hist(convo_lengths_2, bins=bins2, color='green', alpha=0.7)
plt.xlabel('Conversation Length (tokens)')
plt.ylabel('Number of Conversations')
plt.title('Histogram of Conversation Token Lengths\nFile 2 (Tulu)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("histogram_tulu.png")
plt.close()

# Save selected conversations from file2 to finalconvos.jsonl
with open("finalconvos.jsonl", "w") as out_f:
    for convo in selected_convos_2:
        out_f.write(json.dumps(convo) + "\n")

print(f"Saved {len(selected_convos_2)} selected conversations to finalconvos.jsonl")

# stats_shp = compute_shp_stats()

# print("\n=== SHP Dataset Statistics ===")
# print(f"Avg. distinct tokens:       {stats_shp['avg_distinct_tokens']:.2f}")
# print(f"Avg. distinct bigrams:      {stats_shp['avg_distinct_bigrams']:.2f}")
# print(f"Avg. conversation length:   {stats_shp['avg_conversation_length']:.2f} tokens")
# print(f"Avg. turn length:           {stats_shp['avg_turn_length']:.2f}")
# print(f"Word/token ratio:           {stats_shp['word_to_token_ratio']:.4f}")
# print(f"Unique bigram/total ratio:  {stats_shp['unique_bigram_to_total_bigram_ratio']:.4f}")
# print(f"Examples processed:         {len(stats_shp['convo_lengths'])}")

# plt.figure(figsize=(8, 5))
# bins = range(0, max(stats_shp["convo_lengths"]) + 50, 50)

# plt.hist(stats_shp["convo_lengths"], bins=bins, color='purple', alpha=0.7)
# plt.xlabel('Token Length')
# plt.ylabel('Number of Examples')
# plt.title('Histogram of Token Lengths — SHP Dataset')
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.savefig("histogram_shp.png")
# plt.close()
