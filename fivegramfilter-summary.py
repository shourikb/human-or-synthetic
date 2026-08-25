import json
from collections import Counter
from nltk.util import ngrams
import nltk

nltk.download('punkt')

def get_5grams(text):
    tokens = nltk.word_tokenize(text.lower())
    return list(ngrams(tokens, 5))

REFERENCE_FILES = ["sample8.jsonl", "sample9.jsonl"]  # files to compute 5-gram frequencies from
TARGET_FILE = "sample9.jsonl"        # file to filter
OUTPUT_FILE = "filtered_sample9.jsonl"
MAX_REPEATS_IN_POST = 2   # allow up to 2 repeats within a single post, filter if >2
THRESHOLD_FRAC = 0.05  # Adjust this threshold as needed

def repeated_5grams(text):
    grams = get_5grams(text)
    counts = Counter(grams)
    return [gram for gram, c in counts.items() if c > MAX_REPEATS_IN_POST]

all_5grams = []
posts_data = []

# Step 1: Read all posts and extract all 5-grams (based on post text only, not summaries)
for filepath in REFERENCE_FILES:
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            post_text = entry.get("post", "")
            posts_data.append((post_text, entry))
            all_5grams.extend(get_5grams(post_text))

# Step 2: Count frequencies of all 5-grams
freq_5grams = Counter(all_5grams)
threshold = int(len(posts_data) * THRESHOLD_FRAC)
print("5-gram frequency threshold for filtering: ", threshold)

common_5grams = set(gram for gram, count in freq_5grams.items() if count > threshold)

def contains_common_5gram(text):
    grams = get_5grams(text)
    if any(gram in common_5grams for gram in grams):
        return True, None
    repeats = repeated_5grams(text)
    if repeats:
        return True, repeats

    return False, None

posts_data = []
with open(TARGET_FILE, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        post_text = entry.get("post", "")
        posts_data.append((post_text, entry))

# Step 3: Filter out posts that contain any common 5-gram (based on post text only)
filtered_entries = []
for post_text, entry in posts_data:
    flagged, repeats = contains_common_5gram(post_text)
    if not flagged:
        filtered_entries.append(entry)

# Step 4: Save filtered posts to a new JSONL file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
    for entry in filtered_entries:
        json.dump(entry, f_out)
        f_out.write("\n")

print(f"Original number of posts: {len(posts_data)}")
print(f"Number of posts after filtering: {len(filtered_entries)}")
print(common_5grams)