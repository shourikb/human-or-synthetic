import matplotlib.pyplot as plt
import json
import re
from collections import Counter
from transformers import AutoTokenizer
from datasets import load_dataset

BIN_SIZE = 50

NEW_SOURCE_FILE   = "filtered_sample7.jsonl"
EXISTING_CONVOS   = "finalconvos2.jsonl"
OUTPUT_FILE       = "finalconvos3.jsonl"
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

if EXISTING_CONVOS:
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
else:
    print("No existing conversations specified, starting fresh.")


# ── Step 3: figure out how many we can take to match Anthropic distribution ──
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

# Phase 1: fill each bin as much as possible up to Anthropic proportions
# For each bin, take min(available, anthropic_target) independently
# First find what a "fully filled" Anthropic distribution would look like at our scale
total_available = sum(available_per_bin.values())

anthropic_bin_targets = {}
for b in anthropic_bins:
    prob = anthropic_probs.get(b, 0)
    ideal = int(prob * total_available)  # ideal count if we had unlimited data
    available = max(0, available_per_bin.get(b, 0) - existing_bins_used.get(b, 0))
    anthropic_bin_targets[b] = min(ideal, available)  # take as much as we can

print(f"Anthropic bin targets: {anthropic_bin_targets}")

# ── Step 4: select conversations from new source ─────────────────────────────
print(f"\nSelecting new conversations from {NEW_SOURCE_FILE}...")
new_convos = []
new_bins_used = Counter()
leftover_items = {}  # items not used in phase 1

with open(NEW_SOURCE_FILE, 'r') as f:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line)
        text = item['conversation'] + " " + item['response_A']
        convo_len = len(tokenize(text))
        bin_id = get_bin(convo_len, BIN_SIZE)

        if new_bins_used[bin_id] < anthropic_bin_targets.get(bin_id, 0):
            new_bins_used[bin_id] += 1
            new_convos.append(item)
        else:
            leftover_items.setdefault(bin_id, []).append(item)

print(f"Selected {len(new_convos)} conversations matching Anthropic distribution.")

# Phase 2: take remaining convos preserving current distribution
current_total = len(existing_convos) + len(new_convos)
current_bins = existing_bins_used + new_bins_used
current_total_bins = sum(current_bins.values())
current_probs = {k: v / current_total_bins for k, v in current_bins.items()} if current_total_bins else {}

all_leftovers = [(bin_id, item) for bin_id, items in leftover_items.items() for item in items]
total_leftovers = len(all_leftovers)

# Find max we can add while preserving current distribution
max_extra = float('inf')
for bin_id, prob in current_probs.items():
    if prob > 0:
        available_extra = len(leftover_items.get(bin_id, []))
        if available_extra == 0:
            continue
        max_extra = min(max_extra, available_extra / prob)

if max_extra == float('inf'):
    max_extra = 0

max_extra = int(max_extra)
print(f"Max additional conversations preserving current distribution: {max_extra}")

extra_bin_targets = {b: int(prob * max_extra) for b, prob in current_probs.items()}
extra_bins_used = Counter()

for bin_id, item in all_leftovers:
    if extra_bins_used[bin_id] < extra_bin_targets.get(bin_id, 0):
        extra_bins_used[bin_id] += 1
        new_convos.append(item)

print(f"Selected {len(new_convos)} total new conversations (Anthropic fill + extras).")


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
final_bins = existing_bins_used + new_bins_used + extra_bins_used
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
plt.rcParams["pdf.fonttype"] = 42
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
plt.savefig("histogram_finalconvos2.pdf")
plt.close()

print("\nDone.")