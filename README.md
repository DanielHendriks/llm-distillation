# Honey, I Shrunk the Language Model: Impact of Knowledge Distillation Methods on Performance and Explainability

> Reference implementation that accompanies the paper *Honey, I Shrunk the Language Model: Impact of Knowledge Distillation Methods on Performance and Explainability* by Daniel Hendriks, Philipp Spitzer, Niklas Kühl, and Gerhard Satzger (KIT). Preprint: https://arxiv.org/abs/2504.16056.

---

## Abstract

This repository studies how different knowledge distillation methods influence both task performance and interpretability for small language models. Teacher models generate explanations, critiques, counterfactuals, and revisions. We train smaller student models on the generated data and evaluate them on question answering and natural language inference benchmarks, such as CQA, ESNLI, and StrategyQA. The code supports full experiment reproduction, including data processing, student-model training, inference-time rationale generation, statistical analysis, and visualization.

---

## Repository at a Glance

- `src/` – end-to-end pipeline code: dataloaders, student training (`decoder_only_training.py`, `encoder_decoder_training.py`), explanation generation (`generate_*.py`), metrics, and utilities.
- `datasets/` – curated benchmark splits (CommonsenseQA, e-SNLI, StrategyQA, etc.) plus distilled teacher annotations.
- `llm_outputs/` – cached teacher outputs (rationales, counterfactuals, critiques, revisions) used to train students.
- `figures/` – plot scripts and exported figures for the manuscript.
- `misc/` – analysis notebooks (e.g., rationale comparisons, statistical robustness checks).
- `environment.yml` / `requirements.txt` – reproducible environment definitions.

---

## Getting Started

### 1. Clone and configure the environment

```bash
# clone (use SSH/HTTPS as appropriate)
git clone https://github.com/<org>/llm-distillation.git
cd llm-distillation

# create the conda environment
conda env create -f environment.yml
conda activate model-distillation
# alternatively: pip install -r requirements.txt
```

Optional (commented in `environment.yml`): enable Weights & Biases, SpaCy, or REST back-ends by uncommenting the relevant pip packages.

### 2. Authenticate Large Language Models (if needed)

If you access hosted teacher models (e.g., Hugging Face Hub), export the required tokens before running scripts:

```bash
export HF_HOME=~/.cache/huggingface
export HF_TOKEN=<your_token>
```

---

## Workflow Overview

1. **Prepare datasets** – Load benchmark splits and augment them with teacher explanations.
2. **Generate supervision signals** – Use `generate_counterfactuals.py`, `generate_rationales.py`, etc., to create enriched training corpora.
3. **Train student models** – Run decoder-only or encoder-decoder training scripts with unified arguments.
4. **Evaluate and analyze** – Compute metrics, run statistical tests (`misc/statistical_analysis`), and render figures (`figures/`).
5. **Document results** – Capture checkpoints, logs, and visualizations for the manuscript.

---

## Data Generation

Teacher outputs are stored under `llm_outputs/`. Each generator script shards outputs to ease recovery from interruptions. To run script, e.g. to generate explanations with the teacher, run the following:

```python ../src/generate_rationales.py \
    --checkpoint Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --output_name cqa-qwen-train \
    --dataset cqa \
    --split train \
    --batch_size 24 \
    --max_new_tokens 360 \
    --compile_model
```

Outputs are saved to a folder in `output_name`. Adjust `--dataset` (choices include `cqa`, `esnli`, `strategyqa`) and provide a Hugging Face checkpoint that supports chat-style prompting. You can also adjust batch size, number of generated tokens, turn on/off model compilation. Similarly, critiques, revisions, and couterfactuals can be generated, using the respectives scripts in `src/generate_*`. 

---

## Training Student Models

Both training entry points share a consistent CLI. Key arguments:

- `--dataset` – benchmark to train on.
- `--train_data_path` / `--test_data_path` – location of distilled datasets.
- `--model_name` – base student checkpoint (e.g., `Qwen/Qwen3-0.6B`).
- `--model_type` – supervision setting: `task_prefix`, `counterfactual_prefix`, or `both`.
- `--alpha` – loss weighting between prediction and explanation objectives.
- `--subsample` – fraction of the dataset for ablation or debugging.
- `--group_name` – Weights & Biases run grouping.

### Decoder-only students

```bash
python src/decoder_only_training.py \
  --dataset cqa \
  --model_name Qwen/Qwen3-0.6B \
  --model_type both \
  --train_data_path llm_outputs/consolidated/cqa-llama-train \
  --test_data_path llm_outputs/consolidated/cqa-llama-test \
  --target_rationale few_shot_positive_rationale \
  --output_dir student_models/qwen-cf-cqa \
  --batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --max_steps 5000
```


### Encoder-decoder students

```bash
python src/encoder_decoder_training.py \
  --dataset esnli \
  --model_name google/t5-v1_1-large \
  --train_data_path llm_outputs/consolidated/esnli-train \
  --test_data_path llm_outputs/consolidated/esnli-test \
  --target_rationale few_shot_positive_rationale \
  --output_dir student_models/flan-esnli \
  --learning_rate 3e-5 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --max_steps 5000
```


---

## Statistical Analysis

To reproduce our statistical analysis, please see `misc/statistical_analysis`. The ``R'' code in this folder can be used to reproduce results from our paper.

---

## Contact

For questions, reach out to Daniel Hendriks at daniel.hendriks@kit.edu. Issues and pull requests are welcome for clarifications, bug fixes, and extensions.
