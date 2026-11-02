import matplotlib.pyplot as plt
import json
import re
from collections import Counter
from transformers import AutoTokenizer

BIN_SIZE = 50

NEW_SOURCE_FILE   = "filtered_sample9.jsonl"       # synthetic summarization posts (post/summary_A/summary_B)
EXISTING_POSTS    = None            # previously-selected synthetic posts, same format
OUTPUT_FILE       = "finalposts2.jsonl"
REDDIT_FILE       = "dpo-training/datasets/pt-reddit_full.jsonl"  # human reference (chosen/rejected format)

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

def extract_post_text(item):
    """Return just the post text, regardless of whether the item comes from
    a pre-label synthetic file (has a 'post' field directly) or the human
    reference file (post is embedded inside 'chosen' as
    'Human: Summarize the following text\n\n{post}\n\nAssistant: {summary}')."""
    if "post" in item:
        return item["post"]

    chosen = item.get("chosen", "")
    match = re.search(r"Summarize the following text\n\n(.*?)\n\nAssistant:", chosen, re.DOTALL)
    if match:
        return match.group(1)
    # fallback: strip the leading "Human: " tag if the marker above isn't found
    return re.sub(r"^Human:\s*", "", chosen)

def compute_averages_from_jsonl(path, return_selected=False, target_bins=None):
    selected_posts = []
    bins_used = Counter()
    total_token_counts = 0
    total_unique_bigram_counts = 0
    total_post_length = 0
    post_lengths = []
    count = 0
    global_unique_words = set()
    total_bigrams = 0

    with open(path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            text = extract_post_text(item)

            post_len = len(tokenize(text))
            bin_id = get_bin(post_len, BIN_SIZE)

            if target_bins is not None and bins_used[bin_id] >= target_bins.get(bin_id, 0):
                continue

            bins_used[bin_id] += 1
            if return_selected:
                selected_posts.append(item)

            post_lengths.append(post_len)
            total_token_counts += count_distinct_tokens(text)
            bgs = bigrams(text)
            total_unique_bigram_counts += len(set(bgs))
            total_bigrams += len(bgs)
            total_post_length += post_len
            global_unique_words.update(tokenize(text))
            count += 1

    avg_tokens  = total_token_counts / count if count else 0
    avg_bigrams = total_unique_bigram_counts / count if count else 0
    avg_post    = total_post_length / count if count else 0
    wtr         = len(global_unique_words) / total_post_length if total_post_length else 0
    ubtb        = total_unique_bigram_counts / total_bigrams if total_bigrams else 0

    if return_selected:
        return avg_tokens, avg_bigrams, avg_post, post_lengths, wtr, ubtb, selected_posts
    else:
        return avg_tokens, avg_bigrams, avg_post, post_lengths, wtr, ubtb


# ── Step 1: compute human (Reddit) post-length bin distribution ──────────────
print("Computing Reddit human post-length bin distribution...")
avg_tokens_1, avg_bigrams_1, avg_post_1, post_lengths_1, wtr_1, ubtb_1 = \
    compute_averages_from_jsonl(REDDIT_FILE)

reddit_bins = Counter()
for length in post_lengths_1:
    reddit_bins[get_bin(length, BIN_SIZE)] += 1

total_reddit = sum(reddit_bins.values())
reddit_probs = {k: v / total_reddit for k, v in reddit_bins.items()}
print(f"Reddit bin counts: {reddit_bins}")


# ── Step 2: load existing selected posts and count which bins are already filled ─
print(f"\nLoading existing posts from {EXISTING_POSTS}...")
existing_posts = []
existing_bins_used = Counter()

if EXISTING_POSTS:
    with open(EXISTING_POSTS, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            text = extract_post_text(item)
            post_len = len(tokenize(text))
            bin_id = get_bin(post_len, BIN_SIZE)
            existing_bins_used[bin_id] += 1
            existing_posts.append(item)
    print(f"Loaded {len(existing_posts)} existing posts.")
else:
    print("No existing posts specified, starting fresh.")


# ── Step 3: figure out how many we can take to match Reddit distribution ─────
print(f"\nScanning {NEW_SOURCE_FILE} to count available posts per bin...")
available_per_bin = Counter()

with open(NEW_SOURCE_FILE, 'r') as f:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line)
        text = extract_post_text(item)
        post_len = len(tokenize(text))
        bin_id = get_bin(post_len, BIN_SIZE)
        available_per_bin[bin_id] += 1

print(f"Total available in new source: {sum(available_per_bin.values())}")

total_available = sum(available_per_bin.values())

reddit_bin_targets = {}
for b in reddit_bins:
    prob = reddit_probs.get(b, 0)
    ideal = int(prob * total_available)  # ideal count if we had unlimited data
    available = max(0, available_per_bin.get(b, 0) - existing_bins_used.get(b, 0))
    reddit_bin_targets[b] = min(ideal, available)  # take as much as we can

print(f"Reddit bin targets: {reddit_bin_targets}")

# ── Step 4: select posts from new source ──────────────────────────────────────
print(f"\nSelecting new posts from {NEW_SOURCE_FILE}...")
new_posts = []
new_bins_used = Counter()
leftover_items = {}  # items not used in phase 1

with open(NEW_SOURCE_FILE, 'r') as f:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line)
        text = extract_post_text(item)
        post_len = len(tokenize(text))
        bin_id = get_bin(post_len, BIN_SIZE)

        if new_bins_used[bin_id] < reddit_bin_targets.get(bin_id, 0):
            new_bins_used[bin_id] += 1
            new_posts.append(item)
        else:
            leftover_items.setdefault(bin_id, []).append(item)

print(f"Selected {len(new_posts)} posts matching Reddit distribution.")

# Phase 2: take remaining posts preserving current distribution
current_bins = existing_bins_used + new_bins_used
current_total_bins = sum(current_bins.values())
current_probs = {k: v / current_total_bins for k, v in current_bins.items()} if current_total_bins else {}

all_leftovers = [(bin_id, item) for bin_id, items in leftover_items.items() for item in items]

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
print(f"Max additional posts preserving current distribution: {max_extra}")

extra_bin_targets = {b: int(prob * max_extra) for b, prob in current_probs.items()}
extra_bins_used = Counter()

for bin_id, item in all_leftovers:
    if extra_bins_used[bin_id] < extra_bin_targets.get(bin_id, 0):
        extra_bins_used[bin_id] += 1
        new_posts.append(item)

print(f"Selected {len(new_posts)} total new posts (Reddit fill + extras).")

# ── Step 5: write output = existing + new ─────────────────────────────────────
all_posts = existing_posts + new_posts
print(f"\nWriting {len(all_posts)} total posts to {OUTPUT_FILE}...")

with open(OUTPUT_FILE, 'w') as f:
    for post in all_posts:
        f.write(json.dumps(post) + "\n")

print(f"Saved {len(all_posts)} posts to {OUTPUT_FILE}")
print(f"  - Existing (preserved):  {len(existing_posts)}")
print(f"  - New (from {NEW_SOURCE_FILE}): {len(new_posts)}")


# ── Step 6: verify final bin distribution ─────────────────────────────────────
print("\nVerifying final bin distribution...")
final_bins = existing_bins_used + new_bins_used + extra_bins_used
total_final = sum(final_bins.values())
final_probs = {k: v / total_final for k, v in final_bins.items()}

print(f"{'Bin':>6}  {'Reddit%':>10}  {'Final%':>8}  {'Count':>8}")
print("-" * 40)
for bin_id in sorted(reddit_probs.keys()):
    rp = reddit_probs.get(bin_id, 0) * 100
    fp = final_probs.get(bin_id, 0) * 100
    count = final_bins.get(bin_id, 0)
    print(f"{bin_id:>6}  {rp:>10.2f}%  {fp:>7.2f}%  {count:>8}")


# ── Step 7: plot histograms ────────────────────────────────────────────────────
plt.rcParams["pdf.fonttype"] = 42
all_lengths = []
with open(OUTPUT_FILE, 'r') as f:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line)
        text = extract_post_text(item)
        all_lengths.append(len(tokenize(text)))

plt.figure(figsize=(8, 5))
plt.hist(all_lengths, bins=range(0, max(all_lengths) + 50, 50), color='green', alpha=0.7)
plt.xlabel('Post Length (tokens)')
plt.ylabel('Number of Posts')
plt.title(f'Histogram of Post Token Lengths\n{OUTPUT_FILE}')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("histogram_finalposts.pdf")
plt.close()

print("\nDone.")