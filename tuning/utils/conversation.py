import re


def split_conversation(text):
    text = text.strip()
    turns = re.split(r'(?=Human:|Assistant:)', text)
    messages = []
    for turn in turns:
        turn = turn.strip()
        if not turn:
            continue
        if turn.startswith("Human:"):
            messages.append({"role": "user", "content": turn[len("Human:"):].strip()})
        elif turn.startswith("Assistant:"):
            messages.append({"role": "assistant", "content": turn[len("Assistant:"):].strip()})
    return messages


def split_prompt_and_response(text):
    messages = split_conversation(text)
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError(f"Expected conversation to end in an Assistant turn: {text[:200]}")
    return messages[:-1], messages[-1]["content"]
