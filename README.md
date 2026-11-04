# PAPER TITLE

Code for the paper PAPER TITLE

![Method](assets/method.png)

Link - [ARXIV_URL](ARXIV_URL)

### Install

Training, inference, data generation and plotting:

```bash
conda create --name tuning python=3.10
conda activate tuning
pip install -r requirements.txt
```

The LLM judges (gpt-oss-20b and Qwen) need a newer vLLM, so they run in a separate environment:

```bash
conda create --name judge python=3.12
conda activate judge
pip install -r requirements-judge.txt
```

Run every command from the repository root.

### Synthetic data

Download the [PersonaHub](https://huggingface.co/datasets/proj-persona/PersonaHub) `persona.jsonl` file and the fastText [language-ID model](https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin), then keep the English personas:

```bash
python -m generation.filter_personas --input_file persona.jsonl --output_file personas.jsonl --lid_model lid.176.bin
```

Each step below is shown for helpfulness; replace `helpfulness` with `summarization` for the summarization task.

1. Generate conversations, each with two candidate responses (one shard per GPU):

```bash
bash generation/scripts/generate_helpfulness.sh personas.jsonl synthetic.jsonl
```

2. Remove conversations that share common 5-grams, then match the length distribution of the human data:

```bash
python -m generation.ngram_filter_helpfulness --reference_files synthetic.jsonl --input_file synthetic.jsonl --output_file synthetic_ngram.jsonl
python -m generation.length_match_helpfulness --input_file synthetic_ngram.jsonl --reference_file tuning/datasets/pt-anthropic.jsonl --output_file synthetic_filtered.jsonl
```

3. Label each pair with 32 sampled Qwen3-32B judgments. To split the work across jobs, set `TOTAL_RUNS` and give each job its own `RUN_ID`:

```bash
RUN_ID=0 TOTAL_RUNS=1 bash generation/scripts/label_preferences_helpfulness.sh synthetic_filtered.jsonl votes/
python -m generation.merge_preference_shards --input_glob "votes/majority_votes_32_synthetic_run*.jsonl" --output_file votes.jsonl
```

4. Turn the majority votes into chosen/rejected pairs:

```bash
python -m generation.build_pairs_helpfulness --items_file synthetic_filtered.jsonl --votes_file votes.jsonl --output_file tuning/datasets/pt-synthetic.jsonl
```

### Human data

Training data lives in `tuning/datasets/`: `pt-<name>.jsonl` holds preference pairs (`chosen` and `rejected` transcripts in `Human: ... Assistant: ...` format) and `sft-<name>.jsonl` holds SFT examples (`messages` lists). Convert the human preference datasets with:

```bash
python -m tuning.data.convert_helpsteer3 --input_file helpsteer3_train.jsonl --output_file tuning/datasets/pt-helpsteer3-g.jsonl
python -m tuning.data.convert_shp --input_file shp_train.jsonl --output_file tuning/datasets/pt-shp.jsonl
python -m tuning.data.convert_tldr --input_file tldr_comparisons.jsonl --output_file tuning/datasets/pt-reddit.jsonl
```

[Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) is already in this format and is used directly as `pt-anthropic.jsonl`. Make the SFT version of any preference file, and the human/synthetic mixtures, with:

```bash
python -m tuning.data.convert_pt_to_sft --input_file tuning/datasets/pt-anthropic.jsonl --output_file tuning/datasets/sft-anthropic.jsonl
python -m tuning.data.mix_datasets --sft_human_file tuning/datasets/sft-reddit.jsonl --sft_synthetic_file tuning/datasets/sft-synsum.jsonl \
    --pt_human_file tuning/datasets/pt-reddit.jsonl --pt_synthetic_file tuning/datasets/pt-synsum.jsonl \
    --output_dir tuning/datasets --output_suffix _sum
```

### Run

1. Supervised finetuning on `train_size` examples of a dataset:

```bash
bash tuning/scripts/train_sft.sh llama3-8B anthropic 769
```

2. DPO on top of the SFT model:

```bash
bash tuning/scripts/train_dpo.sh llama3-8B anthropic 769 --init_model tuning/models/llama3-8B_sft-anthropic-769
```

The base model is one of `llama3-8B`, `olmo3-7B` or `qwen2.5-7B`. Models are saved in `tuning/models/`. To search over DPO hyperparameters, use `bash tuning/scripts/sweep_dpo.sh`, which writes one row per run to a CSV.

3. Generate responses on the test set and judge them against the reference responses (in the `judge` environment):

```bash
bash tuning/scripts/generate_responses.sh tuning/models/<run_name> llama3-8B tuning/datasets/anthropic_test.jsonl responses/<run_name>.jsonl
bash tuning/scripts/judge.sh helpfulness gptoss responses/<run_name>.jsonl tuning/datasets/anthropic_test.jsonl judge_results.jsonl raw/<run_name>.jsonl
```

`judge.sh` takes `helpfulness` or `summarization` and `gptoss` or `qwen`. Each run appends its win rate to `judge_results.jsonl`.

4. Make the figures:

```bash
python analysis/plots/winrate_vs_cost_helpfulness.py --output_dir figures/
python analysis/plots/cost_sensitivity_helpfulness.py --output figures/cost_sensitivity_helpfulness.png
python analysis/plots/mixed_splits_helpfulness.py --output figures/mixed_helpfulness_by_split.png
python analysis/plots/response_lengths.py --responses_dir responses/ --out figures/response_lengths.png
python analysis/cost.py --generation_files synthetic.jsonl --preference_file votes.jsonl
```

## Reference

If you find our work or code useful, please cite the paper:

```
BIBTEX
```
