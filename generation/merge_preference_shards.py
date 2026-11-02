import json
import os
import glob
import argparse

BASE_DIR = '/coc/pskynet6/sbanerjee332'
NUM_SAMPLES = 110448
VOTES = 32
DATASET = "summarization"
TOTAL_RUNS = 12

parser = argparse.ArgumentParser()
parser.add_argument('--global_offset', type=int, default=0, help='Offset used during generation')
parser.add_argument('--total_conversations', type=int, default=110448, help='Total conversations expected')
args = parser.parse_args()

input_pattern = os.path.join(BASE_DIR, f"majority_votes_{VOTES}_{DATASET}_run*_offset{args.global_offset}.jsonl")
output_path = os.path.join(BASE_DIR, f"majority_votes_{NUM_SAMPLES}_{DATASET}_merged_offset{args.global_offset}.jsonl")

all_records = []
for path in sorted(glob.glob(input_pattern)):
    print(f"Reading {path}...")
    with open(path) as f:
        for line in f:
            try:
                all_records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Skipping bad line: {e}")

# Sort by global conversation index for reproducibility
all_records.sort(key=lambda x: x.get('global_conv_idx', 0))
print(f"Total records: {len(all_records)}")

with open(output_path, 'w') as f:
    for r in all_records:
        f.write(json.dumps(r) + "\n")

print(f"Merged output: {output_path}")

# Sanity check: count unique global_conv_idx
indices = [r.get('global_conv_idx') for r in all_records]
print(f"Unique conversation indices: {len(set(indices))}")

expected = set(range(args.global_offset, args.global_offset + args.total_conversations))
missing = expected - set(indices)
if missing:
    print(f"WARNING: {len(missing)} missing indices: {sorted(missing)[:20]}...")
else:
    print(f"All {args.total_conversations} conversations accounted for (offset {args.global_offset}).")
