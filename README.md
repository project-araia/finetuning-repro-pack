# Fine-Tuning Reproducibility Pack — ClimRR × LLaMA 3.1

Reproducibility pack for fine-tuning **Meta LLaMA 3.1 8B Instruct** on the **ClimRR** (Climate Risk & Resilience) dataset using LoRA/QLoRA via [Unsloth](https://github.com/unslothai/unsloth). The pack also includes inference scripts, a multi-model evaluation pipeline, and a hybrid scoring suite used in the Project ARIA research effort.

---

## Repository Structure

```
finetuning-repro-pack/
├── datasets/                          # ClimRR train/test splits (Git LFS)
│   ├── ClimRR_Dataset_Train_new_n_final_n.json
│   ├── ClimRR_Dataset_Train_filtered_new_n_final_n.json
│   ├── ClimRR_Dataset_Test_new_n_final_n.json
│   └── ClimRR_Dataset_Test_filtered_new_n_final_n.json
├── evaluations/                       # Pre-computed model evaluation outputs
│   ├── Evaluation_BASE_LLAMA_31_8B_filtered.json
│   ├── Evaluation_FINETUNED_LLAMA_31_8B_filtered.json
│   ├── Evaluation_GEMINI25PRO_filtered.json
│   ├── Evaluation_GPT4O.json
│   └── Evaluation_GPT4O_filtered.json
├── scoring/
│   └── score.py                       # Hybrid numeric + semantic + LLM-judge scorer
├── templates/
│   └── templates_climrr_queries.txt   # Query/answer templates for ClimRR variables
├── testing/
│   ├── argo.py                        # Argo API client (ANL internal)
│   ├── climparser.py                  # ClimRR dataset parser
│   ├── templater.py                   # Template engine for query generation
│   ├── test_argo.py                   # Run inference via Argo (GPT-4o, Gemini, etc.)
│   └── test_llama.py                  # Run inference with the fine-tuned LLaMA model
├── training/
│   └── train_llama.py                 # LoRA fine-tuning script (Unsloth + TRL)
└── finetuning_repo.yml                # Conda environment (Python 3.10, CUDA 12)
```

---

## Setup

### Prerequisites

- Linux with CUDA 12 GPU (A100 / H100 recommended for 16 k context)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- [Git LFS](https://git-lfs.com/) (datasets are stored with LFS)

### 1. Clone and pull LFS objects

```bash
git clone <repo-url>
cd finetuning-repro-pack
git lfs pull
```

### 2. Create the Conda environment

```bash
conda env create -f finetuning_repo.yml
conda activate finetuning-repro
```

> The environment pins all packages including `unsloth`, `trl`, `transformers`, `torch`, and CUDA libraries. Expect ~15 GB of downloads on a fresh install.

---

## Usage

### Training

Fine-tune LLaMA 3.1 8B Instruct with LoRA (rank 16, 4-bit QLoRA):

```bash
# Edit the dataset path inside the script first
# dataset_path = "datasets/ClimRR_Dataset_Train_filtered_new_n_final_n.json"
python training/train_llama.py
```

**Key hyperparameters** (see `training/train_llama.py`):

| Parameter | Value |
|---|---|
| Base model | `unsloth/Meta-Llama-3.1-8B-Instruct` |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| Quantization | 4-bit (QLoRA) |
| Epochs | 2 |
| Learning rate | 5e-5 |
| Batch size | 2 (× 4 grad. accum.) |
| Max sequence length | 16 384 |

The trained adapter is saved to `lora_model/` and a `loss_curve.png` is written to the working directory.

### Inference — Fine-Tuned LLaMA

```bash
# Edit the dataset path inside the script first
# testing_dataset = "datasets/ClimRR_Dataset_Test_filtered_new_n_final_n.json"
python testing/test_llama.py
```

Results are appended to `evaluations/evaluation_results.json` after each sample.

### Inference — Argo API Models (GPT-4o, Gemini, etc.)

Requires access to the ANL Argo API. Set your user via environment variable:

```bash
export ARGO_USER=your_argo_username
export PROJECT_HOME=$(pwd)
cd testing
python test_argo.py
```

The `MODEL` variable at the top of `test_argo.py` selects the backend (e.g., `"gpt4o"`, `"gemini25pro"`).

### Scoring

The scorer computes a **hybrid metric** combining:
- **Numeric agreement** (unit-normalized value matching, weight 0.7)
- **Semantic similarity** via `all-MiniLM-L6-v2` (weight 0.3)
- **LLM-as-judge** factual verdict (Chain-of-Thought, separate pass)

```bash
# Defaults to OpenAI GPT-4o as the judge; set LLM_PROVIDER=INTERNAL to use Argo
export OPENAI_API_KEY=sk-...          # or set LLM_PROVIDER=INTERNAL
export LLM_PROVIDER=OPENAI

# Edit file_path inside score.py to point at your evaluation JSON
python scoring/score.py
```

**Expected output:**

```
EVALUATION SUMMARY
========================================
Total Processed         : N
Avg Weighted Score      : X.XXXX
Avg Units Agreement     : X.XXXX
Avg Semantic Similarity : X.XXXX
LLM Judge Accuracy      : XX.XX%
========================================
```

---

## Dataset

The ClimRR datasets contain structured climate query–response pairs derived from the [ClimRR portal](https://disgeoportal.egs.anl.gov/ClimRR/). Each JSON entry has three fields:

| Field | Description |
|---|---|
| `user` | Natural-language climate query |
| `input` | Structured context data (location, variable values) |
| `assistant` | Reference answer with exact numerical values |

`_filtered_` variants contain a curated subset used for primary experiments.

---

## Evaluation Files

Pre-computed outputs (reference + model response pairs) are included for:

| File | Model |
|---|---|
| `Evaluation_BASE_LLAMA_31_8B_filtered.json` | LLaMA 3.1 8B (base, no fine-tuning) |
| `Evaluation_FINETUNED_LLAMA_31_8B_filtered.json` | LLaMA 3.1 8B (fine-tuned, this repo) |
| `Evaluation_GEMINI25PRO_filtered.json` | Gemini 2.5 Pro |
| `Evaluation_GPT4O_filtered.json` | GPT-4o (filtered set) |
| `Evaluation_GPT4O.json` | GPT-4o (full set) |

Pass any of these to `scoring/score.py` to reproduce reported metrics.

---

## License

Apache 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
