import re
from unsloth.chat_templates import get_chat_template


CHAT_TEMPLATE_MAP = {
    "llama3-8B": "chatml",
    "olmo3-7B": "chatml",
    "qwen2.5-7B": "chatml",
}


def get_model_family(model_name):
    if model_name.startswith("llama3"):
        return "llama3-8B"
    if model_name.startswith("olmo3"):
        return "olmo3-7B"
    if model_name.startswith("qwen2.5"):
        return "qwen2.5-7B"
    raise ValueError(f"Could not infer model family from model_name={model_name!r}")


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


def prepare_dpo_fields(example):
    prompt_messages, chosen_response = split_prompt_and_response(example["chosen"])
    _, rejected_response = split_prompt_and_response(example["rejected"])

    example["prompt"] = prompt_messages
    example["chosen"] = chosen_response
    example["rejected"] = rejected_response
    return example


def _get_assistant_header(tokenizer):
    probe = tokenizer.apply_chat_template(
        [{"role": "assistant", "content": "___CONTENT___"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    return probe.split("___CONTENT___")[0]


def chat_template_func(tokenizer, model_name):
    preset = CHAT_TEMPLATE_MAP[get_model_family(model_name)]
    return get_chat_template(
        tokenizer,
        chat_template=preset,
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
        map_eos_token=True,
    )


def apply_chat_template(tokenizer, dataset, model_name):
    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}

    tokenizer = chat_template_func(tokenizer, model_name)
    dataset = dataset.filter(lambda x: len(x["messages"]) > 0)
    dataset = dataset.map(formatting_prompts_func, batched=True)

    return dataset


def apply_chat_template_pt(tokenizer, dataset, model_name):
    def _strip_prefix(s, pattern):
        return re.sub(f"^{re.escape(pattern)}", "", s)

    def _row_parses_ok(example):
        try:
            split_prompt_and_response(example["chosen"])
            split_prompt_and_response(example["rejected"])
            return True
        except ValueError:
            return False

    tokenizer = chat_template_func(tokenizer, model_name)
    assistant_header = _get_assistant_header(tokenizer)

    def formatting_prompts_func(example):
        example["prompt"] = tokenizer.apply_chat_template(example["prompt"], tokenize=False, add_generation_prompt=True)
        example["chosen"] = tokenizer.apply_chat_template(
            [{"role": "assistant", "content": example["chosen"]}], tokenize=False, add_generation_prompt=False
        )
        example["rejected"] = tokenizer.apply_chat_template(
            [{"role": "assistant", "content": example["rejected"]}], tokenize=False, add_generation_prompt=False
        )

        example["chosen"] = _strip_prefix(example["chosen"], assistant_header)
        example["rejected"] = _strip_prefix(example["rejected"], assistant_header)

        return example

    for split in dataset:
        n_before = len(dataset[split])
        dataset[split] = dataset[split].filter(_row_parses_ok)
        n_after = len(dataset[split])
        if n_after < n_before:
            print(f"=== SKIPPED {n_before - n_after}/{n_before} malformed rows in '{split}' split ===")

    dataset = dataset.map(prepare_dpo_fields)
    dataset = dataset.map(formatting_prompts_func, batched=False)
    return dataset
