"""
5-gram duplicate-detection + train/test contamination report.

Your pipeline applies (at least) two filters before landing on
finalconvos3.jsonl: a common-5-gram / near-duplicate filter, and a length
filter. Diffing the reference pool against finalconvos3.jsonl directly
conflates both. This script instead:

  1. Replicates the original 5-gram filter's exact flagging decision
     (common-gram match OR in-conversation self-repetition) directly against
     the reference pool, so we know precisely which entries were flagged
     *because of common 5-grams*, independent of any other filter.
  2. For "kept" entries (not flagged, and confirmed present in your real
     final dataset), finds actual near-duplicate conversations among the
     common-gram-flagged pool by directly counting how many non-exempt
     5-grams two conversations share with EACH OTHER — not by checking
     whether either one individually crosses the global "common" frequency
     bar. (A kept entry can never share a corpus-common gram with anything,
     by construction — having one is what would have gotten it flagged. So
     real duplicate detection has to compare conversations to each other,
     not to the global threshold set.)
  3. NEW: checks your held-out evaluation set for train/test contamination —
     for each eval example, finds how many literal 5-grams it shares with
     ANY conversation that actually made it into finalconvos3.jsonl (the
     real training data), and flags eval examples that look too close to
     something the model was trained on.

Output is written as JSONL:

    duplicate_examples.jsonl:
        {"conversation": "...", "dup1": "...", "dup2": "...", "dup3": "..."}
        (dup2/dup3 are null if fewer than 3 matching duplicates were found.)

    contamination_report.jsonl:
        {"eval_example": "...", "shared_grams": N, "train_match_1": "...",
         "train_match_2": "...", "train_match_3": "..."}
        (only written for eval examples whose top match meets
        MIN_SHARED_GRAMS_CONTAMINATION; train_match_2/3 are null if fewer
        matches were found.)
"""

import json
from collections import Counter
from nltk.util import ngrams
import nltk
import re

nltk.download('punkt', quiet=True)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REFERENCE_FILES = ["sample9.jsonl"]  # the full pool the 5-gram filter runs over
FINAL_FILE = "finalposts.jsonl"         # dataset after ALL filters — used both to confirm a "kept"
                                           # example actually survived into the real final output, and
                                           # as the "training data" side of the contamination check
TEST_FILE = "dpo-training/datasets/reddit_test.jsonl"              # SET THIS to your actual held-out evaluation file's path.
                                           # Expected to have the same "conversation" / "response_A"
                                           # shape as the other input files.
OUTPUT_FILE = "duplicate_examples_summary.jsonl"
CONTAMINATION_OUTPUT_FILE = "contamination_report_summary.jsonl"

THRESHOLD_FRAC = 0.05     # MUST match the exact value your real filtering run used to produce
                          # FINAL_FILE. This script only tells you the truth about what got
                          # filtered if it replicates the actual decision — changing this to
                          # "make matches appear" defeats the purpose. If this value results in
                          # common_grams being empty/tiny for your corpus size (the diagnostics
                          # below will tell you), that itself is real information: it means your
                          # actual run likely flagged little or nothing via common grams at this
                          # corpus size, and near-duplicates were caught some other way (or not
                          # at all). Confirm the real value from your filtering run before
                          # changing this.
MAX_REPEATS_IN_CONVO = 2  # a 5-gram repeated more than this many times within a single conversation
                          # self-flags that conversation, independent of cross-corpus commonality

TOP_N_GRAMS = 15                 # how many of the most frequent grams to print in the corpus summary
NUM_EXAMPLES = 3                 # how many KEPT conversations to showcase
NUM_DUPLICATES_PER_EXAMPLE = 3   # how many near-duplicates to record per kept example
MIN_SHARED_RAW_GRAMS = 5         # minimum literal 5-grams two conversations must share (excluding
                                  # exempt grams) to count as a near-duplicate pair

# --- contamination-check specific config ---
MIN_SHARED_GRAMS_CONTAMINATION = 5   # minimum literal 5-grams an eval example must share with a
                                      # training example to be reported as possibly contaminated.
                                      # Same semantics as MIN_SHARED_RAW_GRAMS but kept separate so
                                      # you can tune it independently (eval sets are often smaller,
                                      # so you may want a stricter or looser bar).
NUM_TRAIN_MATCHES_PER_EVAL_EXAMPLE = 3  # how many top train matches to record per eval example
CONTAMINATION_TOP_N_REPORT = 20         # how many worst-offending eval examples to print to console


# ---------------------------------------------------------------------------
# Shared helpers (mirrors the original filtering script)
# ---------------------------------------------------------------------------

def clean_tags_for_ngrams(text):
    return re.sub(r"(Human:|AI Assistant:)", "", text)


def get_5grams(text):
    cleaned = clean_tags_for_ngrams(text)
    tokens = nltk.word_tokenize(cleaned.lower())
    return list(ngrams(tokens, 5))


def convo_to_text(convo_str, response_str):
    return convo_str + " " + response_str


def is_exempt_5gram(gram):
    joined = " ".join(gram)
    if "human" in joined or "assistant" in joined:
        return True
    if joined == "i 'd be happy to help":
        return True
    return False


def repeated_5grams(grams):
    counts = Counter(grams)
    return [gram for gram, c in counts.items() if c > MAX_REPEATS_IN_CONVO]


def load_entries(filepath):
    entries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            convo = entry.get("conversation", "")
            response = entry.get("response_A", "")
            text = convo_to_text(convo, response)
            entries.append((text, entry))
    return entries


def entry_identity(entry, text):
    for key in ("id", "conversation_id", "uuid"):
        if entry.get(key) is not None:
            return entry[key]
    return text


def own_grams_for_text(text):
    """Non-exempt 5-grams for a single piece of text — the same unit used for
    both near-duplicate matching and contamination matching."""
    return frozenset(g for g in get_5grams(text) if not is_exempt_5gram(g))


# ---------------------------------------------------------------------------
# Corpus-level stats + flagging (same semantics as the original filter)
# ---------------------------------------------------------------------------

def compute_common_grams(entries_with_text, threshold_frac=THRESHOLD_FRAC, top_n=TOP_N_GRAMS):
    per_entry_grams = [get_5grams(text) for text, _ in entries_with_text]
    flat = [g for grams in per_entry_grams for g in grams]
    freq = Counter(flat)
    num_convos = len(entries_with_text)
    threshold = int(num_convos * threshold_frac)
    common_grams = set(
        gram for gram, count in freq.items()
        if count > threshold and not is_exempt_5gram(gram)
    )

    print(f"\n{'=' * 70}")
    print(f"REFERENCE CORPUS SUMMARY (combined: {', '.join(REFERENCE_FILES)})")
    print(f"{'=' * 70}")
    print(f"Conversations:                       {num_convos}")
    print(f"Total 5-grams (with repeats):         {len(flat)}")
    print(f"Unique 5-grams:                       {len(freq)}")
    print(f"Common-gram threshold (count >):      {threshold} "
          f"(THRESHOLD_FRAC={threshold_frac} x {num_convos} convos)")
    print(f"Size of common_grams set:              {len(common_grams)}")
    print(f"\nTop {top_n} most frequent 5-grams:")
    for gram, count in freq.most_common(top_n):
        print(f"  {count:5d}  {' '.join(gram)!r}")

    return common_grams, per_entry_grams


def summarize_corpus(label, entries_with_text, top_n=TOP_N_GRAMS):
    """Generic diversity/repetition summary for any file on its own terms
    (used for the FINAL_FILE 'after' comparison and for the eval set)."""
    per_entry_grams = [get_5grams(text) for text, _ in entries_with_text]
    flat = [g for grams in per_entry_grams for g in grams]
    freq = Counter(flat)
    num_convos = len(entries_with_text)
    total_grams = len(flat)
    unique_grams = len(freq)
    diversity_ratio = (unique_grams / total_grams) if total_grams else 0.0

    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {label}")
    print(f"{'=' * 70}")
    print(f"Conversations:                       {num_convos}")
    print(f"Total 5-grams (with repeats):         {total_grams}")
    print(f"Unique 5-grams:                       {unique_grams}")
    print(f"Diversity ratio (unique/total):       {diversity_ratio:.4f}  "
          f"(lower = more repetitive/similar)")
    print(f"\nTop {top_n} most frequent 5-grams:")
    for gram, count in freq.most_common(top_n):
        print(f"  {count:5d}  {' '.join(gram)!r}")


def flag_entries(entries_with_text, per_entry_grams, common_grams):
    """Replicates the original filter's decision per entry, and also stores
    each entry's own non-exempt 5-gram set for later near-duplicate matching."""
    results = []
    for (text, entry), grams in zip(entries_with_text, per_entry_grams):
        ident = entry_identity(entry, text)
        common_hit = set(grams) & common_grams
        self_repeat = repeated_5grams(grams)
        flagged = bool(common_hit) or bool(self_repeat)
        own_grams = frozenset(g for g in grams if not is_exempt_5gram(g))
        results.append({
            "ident": ident,
            "text": text,
            "common_hit": common_hit,
            "self_repeat": self_repeat,
            "flagged": flagged,
            "own_grams": own_grams,
        })
    return results


# ---------------------------------------------------------------------------
# Genuine near-duplicate matching via an inverted 5-gram index
# ---------------------------------------------------------------------------

def build_inverted_index(candidates):
    """candidates: list of result-dicts (or plain texts, see build_inverted_index_from_texts)
    with 'own_grams'. Maps gram -> list of candidate indices containing it."""
    index = {}
    for i, r in enumerate(candidates):
        for g in r["own_grams"]:
            index.setdefault(g, []).append(i)
    return index


def build_inverted_index_from_texts(texts):
    """Same idea as build_inverted_index, but for a flat list of raw texts
    (used for the training-side index in the contamination check, where we
    don't need any of the flagging metadata — just the grams)."""
    own_grams_list = [own_grams_for_text(t) for t in texts]
    index = {}
    for i, grams in enumerate(own_grams_list):
        for g in grams:
            index.setdefault(g, []).append(i)
    return index, own_grams_list


def top_near_duplicates(kept_own_grams, index, candidates, min_shared, num_duplicates):
    counts = Counter()
    for g in kept_own_grams:
        for idx in index.get(g, ()):
            counts[idx] += 1
    ranked = [(cnt, candidates[idx]["text"]) for idx, cnt in counts.items() if cnt >= min_shared]
    ranked.sort(key=lambda c: c[0], reverse=True)
    return ranked[:num_duplicates]


def find_near_duplicates(flagged_results, final_ids,
                          output_file=OUTPUT_FILE,
                          num_examples=NUM_EXAMPLES,
                          num_duplicates=NUM_DUPLICATES_PER_EXAMPLE,
                          min_shared=MIN_SHARED_RAW_GRAMS):
    kept_candidates = [r for r in flagged_results if not r["flagged"] and r["ident"] in final_ids]
    # Candidate duplicate pool: entries removed specifically because of a common-gram hit
    common_flagged = [r for r in flagged_results if r["common_hit"]]

    print(f"\n{'=' * 70}")
    print("DIAGNOSTICS")
    print(f"{'=' * 70}")
    print(f"Reference pool size:                          {len(flagged_results)}")
    print(f"Flagged by 5-gram filter (any reason):        {sum(1 for r in flagged_results if r['flagged'])}")
    print(f"  -> flagged due to common grams:              {len(common_flagged)}")
    print(f"  -> flagged due to self-repetition only:      "
          f"{sum(1 for r in flagged_results if r['flagged'] and not r['common_hit'])}")
    print(f"Not flagged by 5-gram filter:                 {sum(1 for r in flagged_results if not r['flagged'])}")
    print(f"  -> and confirmed present in {FINAL_FILE!r}: {len(kept_candidates)}")

    print(f"\nBuilding inverted 5-gram index over {len(common_flagged)} "
          f"common-gram-flagged conversations...")
    index = build_inverted_index(common_flagged)
    print(f"Index has {len(index)} distinct grams.")

    written = 0
    with open(output_file, "w", encoding="utf-8") as f_out:
        for kept in kept_candidates:
            top = top_near_duplicates(kept["own_grams"], index, common_flagged,
                                       min_shared, num_duplicates)
            if not top:
                continue

            top_texts = [text for _, text in top]
            top_texts += [None] * (num_duplicates - len(top_texts))

            row = {"conversation": kept["text"]}
            for i, dup_text in enumerate(top_texts, 1):
                row[f"dup{i}"] = dup_text

            json.dump(row, f_out)
            f_out.write("\n")
            written += 1
            if written >= num_examples:
                break

    print(f"\n{'=' * 70}")
    print("KEPT CONVERSATIONS AND THEIR NEAR-DUPLICATES")
    print(f"{'=' * 70}")
    if written == 0:
        print(f"No kept conversation shared >= {min_shared} literal 5-grams "
              f"with any common-gram-flagged conversation. Try lowering "
              f"MIN_SHARED_RAW_GRAMS.")
    else:
        print(f"Wrote {written} example(s) to {output_file!r} "
              f"(keys: conversation, dup1..dup{num_duplicates}).")


# ---------------------------------------------------------------------------
# NEW: Train/test contamination check
#
# Question this answers: does any conversation in the real training data
# (FINAL_FILE) share a suspicious number of literal 5-grams with a
# conversation in your held-out evaluation set (TEST_FILE)? If so, the eval
# set may not be measuring what you think it's measuring — the model could
# be doing well on it partly because it (or something extremely close to it)
# was trained on.
#
# This mirrors the near-duplicate logic above exactly, just with the two
# sides swapped: instead of comparing kept-vs-flagged within one pool, we
# compare eval-vs-final-train across two pools.
# ---------------------------------------------------------------------------

def check_train_test_contamination(final_entries, test_entries,
                                    output_file=CONTAMINATION_OUTPUT_FILE,
                                    min_shared=MIN_SHARED_GRAMS_CONTAMINATION,
                                    num_matches=NUM_TRAIN_MATCHES_PER_EVAL_EXAMPLE,
                                    top_n_report=CONTAMINATION_TOP_N_REPORT):
    train_texts = [text for text, _ in final_entries]
    test_texts = [text for text, _ in test_entries]

    print(f"\n{'=' * 70}")
    print("TRAIN/TEST CONTAMINATION CHECK")
    print(f"{'=' * 70}")
    print(f"Training conversations ({FINAL_FILE!r}):     {len(train_texts)}")
    print(f"Eval conversations ({TEST_FILE!r}):           {len(test_texts)}")

    print(f"\nBuilding inverted 5-gram index over {len(train_texts)} "
          f"training conversations...")
    train_index, train_own_grams = build_inverted_index_from_texts(train_texts)
    print(f"Index has {len(train_index)} distinct grams.")

    flagged_rows = []   # (shared_count, eval_text, [matched train texts])
    for test_text in test_texts:
        test_grams = own_grams_for_text(test_text)
        counts = Counter()
        for g in test_grams:
            for idx in train_index.get(g, ()):
                counts[idx] += 1
        ranked = [(cnt, train_texts[idx]) for idx, cnt in counts.items() if cnt >= min_shared]
        ranked.sort(key=lambda c: c[0], reverse=True)
        if ranked:
            top = ranked[:num_matches]
            flagged_rows.append((top[0][0], test_text, [t for _, t in top]))

    flagged_rows.sort(key=lambda r: r[0], reverse=True)

    written = 0
    with open(output_file, "w", encoding="utf-8") as f_out:
        for shared_count, eval_text, matches in flagged_rows:
            matches_padded = matches + [None] * (num_matches - len(matches))
            row = {
                "eval_example": eval_text,
                "shared_grams": shared_count,
            }
            for i, m in enumerate(matches_padded, 1):
                row[f"train_match_{i}"] = m
            json.dump(row, f_out)
            f_out.write("\n")
            written += 1

    print(f"\n{'=' * 70}")
    print("CONTAMINATION RESULTS")
    print(f"{'=' * 70}")
    print(f"Eval examples with >= {min_shared} shared literal 5-grams "
          f"with some training conversation: {written} / {len(test_texts)}")
    if written == 0:
        print("No contamination detected at this threshold. Try lowering "
              "MIN_SHARED_GRAMS_CONTAMINATION if you want a stricter sweep.")
    else:
        print(f"Wrote {written} flagged example(s) to {output_file!r} "
              f"(keys: eval_example, shared_grams, train_match_1..train_match_{num_matches}).")
        print(f"\nWorst {min(top_n_report, written)} offenders (by shared 5-gram count):")
        for shared_count, eval_text, _ in flagged_rows[:top_n_report]:
            preview = eval_text[:80].replace("\n", " ")
            print(f"  {shared_count:5d} shared grams  |  {preview!r}...")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    reference_entries = []
    for filepath in REFERENCE_FILES:
        reference_entries.extend(load_entries(filepath))

    common_grams, per_entry_grams = compute_common_grams(reference_entries)
    flagged_results = flag_entries(reference_entries, per_entry_grams, common_grams)

    try:
        final_entries = load_entries(FINAL_FILE)
    except FileNotFoundError:
        print(f"\n{FINAL_FILE!r} not found — can't confirm which kept "
              f"entries survived into the real final dataset, and can't run "
              f"the train/test contamination check.")
        return
    final_ids = {entry_identity(entry, text) for text, entry in final_entries}

    summarize_corpus(f"TARGET - AFTER filtering ({FINAL_FILE})", final_entries)

    find_near_duplicates(flagged_results, final_ids)

    try:
        test_entries = load_entries(TEST_FILE)
    except FileNotFoundError:
        print(f"\n{TEST_FILE!r} not found — skipping train/test "
              f"contamination check. Set TEST_FILE to your eval set's path "
              f"to enable it.")
        return

    summarize_corpus(f"EVAL SET ({TEST_FILE})", test_entries)
    check_train_test_contamination(final_entries, test_entries)


if __name__ == "__main__":
    main()