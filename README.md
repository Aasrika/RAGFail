# RAGFail: A Controlled Study of Noise-Conditioned Failure Modes and Faithfulness Evaluation in Retrieval-Augmented Generation--

## Overview

RAGFail is a systematic empirical framework for studying how Retrieval-Augmented Generation (RAG) systems fail when retrieved documents are noisy or corrupted. While RAG is widely deployed under the assumption that retrieval always helps, we challenge this assumption through controlled experimentation across five noise types and three severity levels.

This work introduces:
- A **controlled three-way experimental design** (Vanilla LLM vs. RAG-Clean vs. RAG-Noisy)
- A **five-category RAG failure mode taxonomy** validated empirically
- **CWF-NLI** — a training-free NLI-based faithfulness metric
- **ANDF** — an adaptive pre-retrieval noise detection and filtering algorithm


## Key Findings
RAG vs. Vanilla tipping point : RAG falls below Vanilla LLM at 75% irrelevant noise (F1: 0.061 vs. 0.064)
Failure mode accuracy :Each noise type maps to its failure mode with 79–82% accuracy 
CWF-NLI human correlation : r = 0.485 vs. F1 r = 0.326 (1.49× improvement)
ANDF recovery (irrelevant noise) :  16.5% F1 recovery 
Most damaging noise type : Irrelevant (−70% F1 from RAG-Clean at 75% noise)
Least damaging noise type : Outdated (−12% F1 from RAG-Clean at 75% noise) 

---

## Failure Mode Taxonomy

 Noise Type : Dominant Failure Mode : Hit Rate 

Irrelevant : Overconfidence Hallucination : 79.0% 
Contradictory : Conflict Resolution Failure : 82.4% 
Outdated : Evidence Hallucination : 80.4% 
Partial : Retrieval Anchoring : 82.0% 



## Experimental Design

We run **Llama-3.1-8B-Instruct** on 500 SQuAD validation queries under three conditions:

```
Query
  ├── Vanilla LLM        → No retrieval, parametric knowledge only
  ├── RAG-Clean          → Top-5 relevant passages via dense retrieval
  └── RAG-Noisy          → Top-5 passages with controlled noise injection
                              ├── Irrelevant  (25% / 50% / 75%)
                              ├── Contradictory (25% / 50% / 75%)
                              ├── Outdated    (25% / 50% / 75%)
                              ├── Partial     (25% / 50% / 75%)
                              └── Mixed       (25% / 50% / 75%)
```

**Total:** 15 noise configurations × 500 queries = **7,500 model generations**

---

## Novel Metrics

### CWF-NLI — Confidence Weighted Faithfulness with NLI Grounding

A training-free faithfulness metric defined as:

```
CWF-NLI = α · G_nli + β · U + γ · R
```

| Component | Description | Weight |
|-----------|-------------|--------|
| G_nli | Max NLI entailment probability across retrieved docs (DeBERTa-v3-small) | α = 0.60 |
| U | Answer specificity / uncertainty proxy | β = 0.25 |
| R | Mean FAISS retrieval cosine similarity | γ = 0.15 |

**Validated against human faithfulness labels:** r = 0.485 vs. F1 r = 0.326

### ANDF — Adaptive Noise Detection and Filtering

A pre-retrieval document filtering algorithm:

```
ANDF(d, q) = w1 · Relevance(d,q) + w2 · Consistency(d,D) + w3 · Temporal(d)
```

| Signal | Description | Weight |
|--------|-------------|--------|
| Relevance | Cosine similarity between document and query embeddings | w1 = 0.50 |
| Consistency | Mean cosine similarity between document and peer documents | w2 = 0.30 |
| Temporal | Normalized recency score from year references in document | w3 = 0.20 |

Documents scoring below threshold τ = 0.45 are filtered before generation.

---

## Setup and Usage

### Prerequisites

- Google Colab account (T4 GPU recommended)
- HuggingFace account with access to [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- Google Drive for checkpoint storage

### Installation

```bash
pip install transformers accelerate bitsandbytes datasets \
    sentence-transformers faiss-gpu rouge-score nltk \
    bert-score matplotlib seaborn scipy scikit-learn
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

### Running the Experiments

Run notebooks **in order**:

```
NB0 → NB1 → NB2 → NB3 → NB3b
```

**Step 1 — NB0_Setup.ipynb** (CPU runtime, ~15 min)
- Downloads SQuAD validation set
- Builds 2,067-passage retrieval corpus
- Creates FAISS flat inner-product index
- Saves corpus and index to Google Drive

**Step 2 — NB1_Vanilla_RAGClean.ipynb** (GPU runtime, ~45 min)
- Runs Vanilla LLM on 500 queries
- Runs RAG-Clean on 500 queries
- Saves results with checkpointing every 25 queries
- **Expected:** Vanilla F1 ≈ 0.064, RAG-Clean F1 ≈ 0.202

**Step 3 — NB2_NoiseExperiments.ipynb** (GPU runtime, ~2–3 hrs)
- Runs all 15 noise configurations
- Each configuration independently checkpointed
- Safe to pause and resume — completed configs are skipped automatically
- Set `QUICK_MODE = True` for faster testing (3 configs)

**Step 4 — NB3_Analysis.ipynb** (GPU runtime, ~45 min)
- Computes CWF scores and ANDF filtering
- Generates failure mode distributions
- Produces all result plots

**Step 5 — NB3b_NLI_HumanEval.ipynb** (GPU runtime, ~1.5 hrs)
- Computes NLI-based CWF-NLI scores
- Generates human evaluation annotation CSV
- Loads annotated CSV and computes correlation with human labels

### Configuration

Before running NB1, NB2, NB3, NB3b — update these two lines:

```python
# Your HuggingFace token (get from https://huggingface.co/settings/tokens)
HF_TOKEN = "hf_YOUR_TOKEN_HERE"

# Path to your Google Drive folder
BASE = '/content/drive/MyDrive/RAGFail'
```

---

## Checkpointing and Fault Tolerance

All notebooks save progress every 25 queries. If your GPU session ends mid-run:

1. Reopen the notebook
2. Rerun all cells from top to bottom
3. Completed configurations load instantly from Drive
4. Incomplete configurations resume from last checkpoint

No work is lost between GPU sessions.

---

## Results Summary

Full results are available in `results/final_summary.json`. Key numbers:

```json
{
  "vanilla_f1": 0.064,
  "rag_clean_f1": 0.202,
  "tipping_point": "irrelevant_75 (F1: 0.061)",
  "correlations": {
    "cwf_nli_r": 0.485,
    "f1_human_r": 0.326
  },
  "failure_mode_accuracy": "79-82%",
  "andf_recovery_irrelevant": "16.5%"
}
```

---

## Human Evaluation

50 answers were manually annotated for binary faithfulness (1 = faithful, 0 = not faithful):
- 25 samples from RAG-Clean condition
- 25 samples from irrelevant\_50 condition
- Annotation criterion: *Is this answer supported by the retrieved documents shown?*
- Result: 41 faithful, 9 unfaithful

Annotation file: `results/human_eval_50samples.csv`

---

## Model and Dataset

| Component | Details |
|-----------|---------|
| Language Model | Llama-3.1-8B-Instruct (4-bit NF4 quantization) |
| Dataset | SQuAD validation set (500 queries) |
| Retrieval Corpus | 2,067 Wikipedia passages from SQuAD contexts |
| Sentence Encoder | all-MiniLM-L6-v2 |
| Retrieval | FAISS flat inner-product index, top-5 passages |
| NLI Model | cross-encoder/nli-deberta-v3-small |
| Compute | NVIDIA T4 GPU (Google Colab), ~8 GPU-hours total |

---

## Team Members
- **Kambhampati Aasrika** — RV University, Bangalore
- **Hannah Beth Eappen** — RV University, Bangalore
- **Bhuvana S** — RV University, Bangalore





*Paper under review — repository will be made fully public upon acceptance.*
