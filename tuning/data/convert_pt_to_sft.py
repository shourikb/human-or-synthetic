import json
import re

def parse_conversation(text):
    messages = []
    # Split on Human:/Assistant: turns
    turns = re.split(r'\s+(Human|AI Assistant|Assistant):\s*', text.strip())
    # turns[0] is empty, then alternates role/content
    if turns[0] and not turns[0] == '':
        # Parse the leftover first turn manually
        match = re.match(r'^(Human|AI Assistant|Assistant): (.+)$', turns[0], re.DOTALL)
        if match:
            role = "user" if match.group(1) == "Human" else "assistant"
            messages.append({"role": role, "content": match.group(2).strip()})
    for i in range(1, len(turns), 2):
        role = "user" if turns[i] == "Human" else "assistant"
        content = turns[i + 1].strip()
        messages.append({"role": role, "content": content})
    return messages

with open("pt-synsum.jsonl") as fin, open("sft-synsum.jsonl", "w") as fout:
    for line in fin:
        example = json.loads(line)
        messages = parse_conversation(example["chosen"])
        fout.write(json.dumps({"messages": messages}) + "\n")