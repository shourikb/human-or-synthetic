import matplotlib.pyplot as plt
import json
import re
from collections import Counter
from transformers import AutoTokenizer
from datasets import load_dataset

BIN_SIZE = 50

NEW_SOURCE_FILE   = "filtered_sample6.jsonl"
EXISTING_CONVOS   = "finalconvos.jsonl"
OUTPUT_FILE       = "finalconvos2.jsonl"
ANTHROPIC_FILE    = "dpo-training/datasets/pt-anthropic.jsonl"

model_name = "allenai/Llama-3.1-Tulu-3-8B-SFT"
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
    return list(zip(tokens, tokens[1:]))

def compute_averages_from_jsonl(path, use_combined=False, return_selected=False, target_bins=None):
    selected_conversations = []
    bins_used = Counter()
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
            item = json.loads(line)
            if use_combined:
                text = item['conversation'] + " " + item['response_A']
            else:
                text = item['chosen']

            convo_len = len(tokenize(text))
            bin_id = get_bin(convo_len, BIN_SIZE)

            if target_bins is not None and bins_used[bin_id] >= target_bins.get(bin_id, 0):
                continue

            bins_used[bin_id] += 1
            if return_selected:
                selected_conversations.append(item)

            convo_lengths.append(convo_len)
            total_token_counts += count_distinct_tokens(text)
            bgs = bigrams(text)
            total_unique_bigram_counts += len(set(bgs))
            total_bigrams += len(bgs)
            total_convo_length += convo_len
            total_turns += text.count("Human:")
            global_unique_words.update(tokenize(text))
            count += 1

    avg_tokens   = total_token_counts / count if count else 0
    avg_bigrams  = total_unique_bigram_counts / count if count else 0
    avg_convo    = total_convo_length / count if count else 0
    avg_turns    = total_turns / count if count else 0
    wtr          = len(global_unique_words) / total_convo_length if total_convo_length else 0
    ubtb         = total_unique_bigram_counts / total_bigrams if total_bigrams else 0

    if return_selected:
        return avg_tokens, avg_bigrams, avg_convo, avg_turns, convo_lengths, wtr, ubtb, selected_conversations
    else:
        return avg_tokens, avg_bigrams, avg_convo, avg_turns, convo_lengths, wtr, ubtb


# ── Step 1: compute Anthropic bin distribution ────────────────────────────────
print("Computing Anthropic bin distribution...")
avg_tokens_1, avg_bigrams_1, avg_convos_1, avg_turn_1, convo_lengths_1, wtr_1, ubtb_1 = \
    compute_averages_from_jsonl(ANTHROPIC_FILE, use_combined=False)

anthropic_bins = Counter()
for length in convo_lengths_1:
    anthropic_bins[get_bin(length, BIN_SIZE)] += 1

total_anthropic = sum(anthropic_bins.values())
anthropic_probs = {k: v / total_anthropic for k, v in anthropic_bins.items()}
print(f"Anthropic bin counts: {anthropic_bins}")


# ── Step 2: load existing finalconvos and count which bins are already filled ─
print(f"\nLoading existing conversations from {EXISTING_CONVOS}...")
existing_convos = []
existing_bins_used = Counter()

with open(EXISTING_CONVOS, 'r') as f:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line)
        text = item['conversation'] + " " + item['response_A']
        convo_len = len(tokenize(text))
        bin_id = get_bin(convo_len, BIN_SIZE)
        existing_bins_used[bin_id] += 1
        existing_convos.append(item)

print(f"Loaded {len(existing_convos)} existing conversations.")


# ── Step 3: figure out how many MORE we can take from new source per bin ──────
# The new source is larger, so scale up the target proportionally.
# We count how many conversations are available in the new source per bin,
# then for each bin take as many as are available while respecting the
# Anthropic ratio relative to the other bins.

print(f"\nScanning {NEW_SOURCE_FILE} to count available conversations per bin...")
available_per_bin = Counter()
available_items_per_bin = {}

with open(NEW_SOURCE_FILE, 'r') as f:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line)
        text = item['conversation'] + " " + item['response_A']
        convo_len = len(tokenize(text))
        bin_id = get_bin(convo_len, BIN_SIZE)
        available_per_bin[bin_id] += 1
        available_items_per_bin.setdefault(bin_id, []).append(item)

print(f"Total available in new source: {sum(available_per_bin.values())}")

# For each bin, the max we can take is limited by availability.
# Find the largest N such that for all bins: N * anthropic_probs[bin] <= available[bin]
# i.e. N <= available[bin] / anthropic_probs[bin] for all bins with nonzero prob
max_n = float('inf')
for bin_id, prob in anthropic_probs.items():
    if prob > 0:
        available = available_per_bin.get(bin_id, 0)
        max_n = min(max_n, available / prob)

max_n = int(max_n)
print(f"Max additional conversations while preserving distribution: {max_n}")

# Compute per-bin targets for the new source
new_bin_targets = {b: int(anthropic_probs.get(b, 0) * max_n) for b in anthropic_bins}


# ── Step 4: select new conversations from new source ─────────────────────────
print(f"\nSelecting new conversations from {NEW_SOURCE_FILE}...")
new_convos = []
new_bins_used = Counter()

with open(NEW_SOURCE_FILE, 'r') as f:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line)
        text = item['conversation'] + " " + item['response_A']
        convo_len = len(tokenize(text))
        bin_id = get_bin(convo_len, BIN_SIZE)

        if new_bins_used[bin_id] >= new_bin_targets.get(bin_id, 0):
            continue

        new_bins_used[bin_id] += 1
        new_convos.append(item)

print(f"Selected {len(new_convos)} new conversations.")


# ── Step 5: write finalconvos2 = existing + new ───────────────────────────────
all_convos = existing_convos + new_convos
print(f"\nWriting {len(all_convos)} total conversations to {OUTPUT_FILE}...")

with open(OUTPUT_FILE, 'w') as f:
    for convo in all_convos:
        f.write(json.dumps(convo) + "\n")

print(f"Saved {len(all_convos)} conversations to {OUTPUT_FILE}")
print(f"  - Existing (preserved):  {len(existing_convos)}")
print(f"  - New (from {NEW_SOURCE_FILE}): {len(new_convos)}")


# ── Step 6: verify final bin distribution ────────────────────────────────────
print("\nVerifying final bin distribution...")
final_bins = existing_bins_used + new_bins_used
total_final = sum(final_bins.values())
final_probs = {k: v / total_final for k, v in final_bins.items()}

print(f"{'Bin':>6}  {'Anthropic%':>10}  {'Final%':>8}  {'Count':>8}")
print("-" * 40)
for bin_id in sorted(anthropic_probs.keys()):
    ap = anthropic_probs.get(bin_id, 0) * 100
    fp = final_probs.get(bin_id, 0) * 100
    count = final_bins.get(bin_id, 0)
    print(f"{bin_id:>6}  {ap:>10.2f}%  {fp:>7.2f}%  {count:>8}")


# ── Step 7: plot histograms ───────────────────────────────────────────────────
all_lengths = []
with open(OUTPUT_FILE, 'r') as f:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line)
        text = item['conversation'] + " " + item['response_A']
        all_lengths.append(len(tokenize(text)))

plt.figure(figsize=(8, 5))
plt.hist(all_lengths, bins=range(0, max(all_lengths) + 50, 50), color='green', alpha=0.7)
plt.xlabel('Conversation Length (tokens)')
plt.ylabel('Number of Conversations')
plt.title(f'Histogram of Conversation Token Lengths\n{OUTPUT_FILE}')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("histogram_finalconvos2.png")
plt.close()

print("\nDone.")