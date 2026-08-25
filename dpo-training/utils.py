import re
from unsloth.chat_templates import get_chat_template
from datasets import DatasetDict, Dataset


# Only Llama3.1 and Olmo3 are in use right now — extend this map if/when
# Qwen2 or Olmo2 come back into rotation, but verify each new entry against
# that model's actual tokenizer_config.json before trusting the preset name.
CHAT_TEMPLATE_MAP = {
    "llama3-8B": "chatml",
    "olmo3-7B": "chatml",
    "qwen2.5-7B": "chatml",
}


def get_model_family(model_name):
    """Composite run names (e.g. 'olmo3-sft-h-200') embed the base model
    family as a prefix. Map back to the plain CHAT_TEMPLATE_MAP key."""
    if model_name.startswith("llama3"):
        return "llama3-8B"
    if model_name.startswith("olmo3"):
        return "olmo3-7B"
    if model_name.startswith("qwen2.5"):
        return "qwen2.5-7B"
    raise ValueError(f"Could not infer model family from model_name={model_name!r}")


def split_conversation(text):
    """Split a 'Human: ...\n\nAssistant: ...' transcript into role-tagged turns.
    Tolerant of missing '\n\n' before a turn marker (a known artifact in some
    synthetic data), by splitting on any occurrence of 'Human:'/'Assistant:'
    rather than requiring it be preceded by a double newline."""
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
    """Return (prompt_messages, final_response_text)."""
    messages = split_conversation(text)
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError(f"Expected conversation to end in an Assistant turn: {text[:200]}")
    return messages[:-1], messages[-1]["content"]


def prepare_dpo_fields(example):
    """Derive prompt/chosen/rejected in the shape apply_chat_template_pt expects,
    since raw data only has full 'chosen'/'rejected' conversation strings, no
    separate 'prompt' field. Chosen and rejected share the same preceding turns,
    so the prompt is taken from chosen.

    Assumes malformed rows have already been filtered out (see _row_parses_ok
    in apply_chat_template_pt) — raises if not, since that indicates a bug
    upstream rather than expected bad data.
    """
    prompt_messages, chosen_response = split_prompt_and_response(example["chosen"])
    _, rejected_response = split_prompt_and_response(example["rejected"])

    example["prompt"] = prompt_messages
    example["chosen"] = chosen_response
    example["rejected"] = rejected_response
    return example


def _get_assistant_header(tokenizer):
    """Discover exactly what prefix this tokenizer's chat template inserts
    before assistant content, so stripping works regardless of which model's
    template (ChatML, Llama3, etc.) is in use."""
    probe = tokenizer.apply_chat_template(
        [{"role": "assistant", "content": "___CONTENT___"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    return probe.split("___CONTENT___")[0]


def chat_template_func(tokenizer, model_name):
    family = get_model_family(model_name)

    if family not in CHAT_TEMPLATE_MAP:
        raise ValueError(
            f"No chat template mapping defined for family={family!r} "
            f"(from model_name={model_name!r}). Known families: {list(CHAT_TEMPLATE_MAP)}"
        )

    preset = CHAT_TEMPLATE_MAP[family]

    if preset is None:
        if tokenizer.chat_template is None:
            raise ValueError(
                f"family={family!r} is mapped to native template, but "
                f"tokenizer.chat_template is not set on this checkpoint."
            )
        return tokenizer

    tokenizer = get_chat_template(
        tokenizer,
        chat_template=preset,
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
        map_eos_token=True,
    )
    return tokenizer


def apply_chat_template(tokenizer, dataset, model_name):
    """SFT formatting: builds a single 'text' field per example from a full
    message list."""

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}

    tokenizer = chat_template_func(tokenizer, model_name)
    dataset = dataset.filter(lambda x: len(x["messages"]) > 0)
    dataset = dataset.map(formatting_prompts_func, batched=True)

    return dataset


def apply_chat_template_pt(tokenizer, dataset, model_name):
    """DPO formatting: derives prompt/chosen/rejected fields from raw
    conversation strings, then templates each.

    Malformed rows (conversation doesn't end in an Assistant turn) are
    dropped via a plain filter predicate before any field derivation or
    templating happens — no tracking columns are added to the dataset,
    to avoid split-dependent schema mismatches.
    """

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

        prompt = example["prompt"]

        if type(prompt) == str:
            message = [
                {"role": "system", "content": example["system_message"]},
                {"role": "user", "content": example["prompt"]},
            ]
        elif type(prompt) == list:
            message = prompt

        example["prompt"] = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
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


def get_kto_rows(dataset):
    def get_rows(dataset_split):
        rows = []
        for prompt, chosen, rejected in zip(dataset_split["prompt"], dataset_split["chosen"], dataset_split["rejected"]):
            rows.extend([
                {
                    "prompt": prompt[100:],
                    "completion": chosen,
                    "label": True,
                },
                {
                    "prompt": prompt[100:],
                    "completion": rejected,
                    "label": False,
                }
            ])

        return rows

    dataset_kto = DatasetDict()
    dataset_kto["train"] = Dataset.from_list(get_rows(dataset["train"]))
    dataset_kto["test"] = Dataset.from_list(get_rows(dataset["test"]))

    print(dataset_kto["train"][0])
    return dataset_kto