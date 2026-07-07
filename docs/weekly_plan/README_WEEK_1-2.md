# Internship Project — Getting Started Guide
**Project:** Citation Recommendation with Dense & Sparse Retrieval  
**Weeks 1–2 Focus:** Understand the system, set up your environment, run the existing baselines  

---

## Table of Contents
1. [What This Project Is About](#1-what-this-project-is-about)
2. [What You Have Been Given](#2-what-you-have-been-given)
3. [Setting Up Your Environment](#3-setting-up-your-environment)
4. [Data: What the Files Are](#4-data-what-the-files-are)
5. [The Codebase: What Each Script Does](#5-the-codebase-what-each-script-does)
6. [How to Run Each Baseline](#6-how-to-run-each-baseline)
7. [Results You Need to Produce](#7-results-you-need-to-produce)
8. [How to Read Your Results](#8-how-to-read-your-results)
9. [How to Modify the Code](#9-how-to-modify-the-code)
10. [Common Errors and Fixes](#10-common-errors-and-fixes)

---

## 1. What This Project Is About

When a researcher writes a paper, they need to cite all the important prior work. Missing a key citation is a real problem — it can make your contribution look more novel than it is, or hide that someone already solved a similar problem.

**The goal of this project** is to build a system that takes a paper (just the title and abstract) and automatically finds which other papers it *must* cite.

We call this **must-cite citation recommendation**.

We have a large dataset of 67,761 AI/ML papers (2018–2024) as our "candidate pool". For each paper in our test set (papers from 2025), we want to retrieve the papers from that pool that the 2025 paper should have cited.

In Weeks 1–2, your job is to **reproduce the baseline results** using four retrieval methods:
- **SciBERT** — a BERT-based language model trained on scientific text
- **SPECTER** — a model trained using citation signals
- **SPECTER2** — an improved version of SPECTER
- **ColBERT** — a model that does "late interaction" (more on this below)

---

## 2. What You Have Been Given

### Papers to Read (before coding!)
1. **ColBERT** (Khattab & Zaharia, SIGIR 2020) — focus on Section 3 (late interaction). Skip BERT pretraining details.
2. **SPECTER** (Cohan et al., ACL 2020) — focus on the citation-based training signal. Skim the rest.
3. **Kücüktunç et al., arXiv 1209.5809** — "Diversifying Citation Recommendations". Focus on Section 2.4 on PPR-based methods. *(Ratul will provide this.)*

### Data (on the server at `/home/ratul/data/masterset_backup/train_eval_set/v2.0/`)
| File | What it contains |
|------|-----------------|
| `train_v2.0.parquet` | ~67,761 papers (2018–2024). This is your **candidate pool** — the papers you search over. |
| `eval_v2.0.parquet` | ~7,028 papers (2025). These are your **query papers** — the ones you run retrieval for, and compute metrics on. |
| `all_papers_with_refs_and_labels_v2.0.parquet` | All papers combined. You won't need this for the baselines, but it's there for reference. |

Each row in the parquet files contains: `paper_id`, `title`, `abstract`, `references` (a list of dicts with label scores).

### Codes (in the `codes/` folder)
```
codes/
├── SciBERT/
│   ├── scibert_utils.py          ← shared helper functions
│   ├── 1-scibert-embeddings.py   ← Step 1: generate embeddings
│   └── 2-scibert-eval.py         ← Step 2: evaluate
├── SPECTER-Pretrained/
│   ├── 1-specter-pretrained-embeddings.py
│   └── 2-specter-pretrained-eval.py
├── SPECTER2-Pretrained/
│   ├── 1-specter2-pretrained-embeddings.py
│   └── 2-specter2-pretrained-eval.py
└── ColBERT/
    ├── colbert_utils.py
    ├── 1-colbert-build-index.py
    └── 2-colbert-eval.py
```

---

## 3. Setting Up Your Environment

### Step 1: Create a conda environment

```bash
conda create -n citation python=3.10
conda activate citation
```

### Step 2: Install common dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers polars faiss-gpu tqdm python-dotenv numpy
pip install adapters   # needed only for SPECTER2
```

### Step 3: Install ColBERT

ColBERT needs to be installed separately from its GitHub repo:

```bash
pip install colbert-ai
```

> **Note:** ColBERT requires a GPU. Ask Ratul which machine to run it on and confirm you have CUDA access with `nvidia-smi`.

### Step 4: Set up the `.env` file

Create a file called `.env` in your working directory (same folder where you run the scripts from):

```
DATA_DIR=/home/ratul/data/masterset_backup/train_eval_set/v2.0
OUTPUT_DIR=/path/to/your/output/folder
```

Replace `/path/to/your/output/folder` with wherever you want results saved (e.g. `/home/yourname/outputs`). The scripts will automatically create subdirectories inside this folder.

### Step 5: Verify you can load the data

```python
import polars as pl
df = pl.read_parquet("/home/ratul/data/masterset_backup/train_eval_set/v2.0/train_v2.0.parquet")
print(df.shape)       # should print something like (67761, N)
print(df.columns)     # see what columns exist
print(df.head(2))     # look at first 2 rows
```

---

## 4. Data: What the Files Are

### The train parquet (`train_v2.0.parquet`)
Each row is one paper. Key columns:
- `paper_id` — unique ID (a UUID string like `"abc123-..."`)
- `title` — paper title
- `abstract` — paper abstract

This file is used as the **index** — the pool of candidates to search over.

### The eval parquet (`eval_v2.0.parquet`)
Same structure as train, but also has:
- `references` — a JSON list of dicts. Each dict is one cited paper. Example:
```json
[
  {
    "matched_paper_id": "xyz-...",   ← the paper ID this reference matched to in our pool
    "type_1_output": 1.0,           ← Type 1 label: 1 = this is a baseline, 0 = not
    "type_2_output": 4.5,           ← Type 2 label: 1–5 relevance score
    "type_3_output": 3.0            ← Type 3 label: how many times cited in this paper
  },
  ...
]
```

### The Three Label Types

A "must-cite" paper is one that satisfies **any** of these:

| Label | What it means | Threshold for "must-cite" |
|-------|--------------|--------------------------|
| **Type 1** | This paper was directly compared against as a baseline in the experiments | `type_1_output == 1` |
| **Type 2** | This paper is core to the method or task (1–5 relevance scale) | `type_2_output >= 4` |
| **Type 3** | This paper was mentioned many times throughout the paper | `type_3_output >= 3` |

Your job is to retrieve papers that match these labels. **You compute metrics separately for each label type.**

---

## 5. The Codebase: What Each Script Does

### The General Pattern (same for all 4 methods)

Every method has two scripts: **Step 1** generates/indexes, and **Step 2** evaluates.

```
Step 1: Encode all train papers → save embeddings/index to disk
Step 2: Load the index, run queries from eval set, compute metrics
```

---

### SciBERT

**What SciBERT does:** Takes a paper's `title [SEP] abstract`, passes it through the SciBERT transformer, and uses the `[CLS]` token — a single 768-dimensional vector — as the paper's embedding. To find similar papers, it computes cosine similarity between the query embedding and all candidate embeddings. Uses FAISS (a fast similarity search library) to do this quickly.

**`1-scibert-embeddings.py`**  
- Loads `train_v2.0.parquet` and `eval_v2.0.parquet`  
- Encodes every paper into a 768-dim vector  
- Saves `train_embeddings.pt`, `eval_embeddings.pt`, and ID mappings to `output/dense/SciBERT/embeddings/`

**`2-scibert-eval.py`**  
- Loads the saved embeddings  
- Builds a FAISS index over train embeddings  
- For each eval paper, finds the top-500 most similar train papers  
- Computes metrics against the ground truth labels

**`scibert_utils.py`**  
- Shared helper functions: the `PaperDataset` class, embedding generation, and all metric functions (MAP, MRR, Recall@K, nDCG@K, HR@K)

---

### SPECTER and SPECTER2

**What SPECTER does:** Same idea as SciBERT (single CLS vector), but the model was pre-trained using citation signals. Papers that cite each other are trained to have similar embeddings. So SPECTER "knows" that BERT and RoBERTa are related, not just because of vocabulary overlap, but because papers about one often cite the other.

SPECTER2 is an improved version with adapter modules.

Both follow the exact same two-script pattern as SciBERT.

**Key difference for SPECTER2:** It uses the `adapters` library (`AutoAdapterModel`) instead of `transformers`. It loads a base model (`allenai/specter2_base`) plus an adapter (`allenai/specter2`). The adapter is a small extra module trained for the "proximity" task (finding related papers).

---

### ColBERT

**What ColBERT does:** Instead of encoding a whole paper into *one* vector, ColBERT keeps a **separate vector for every token**. When scoring a query against a document, it computes **MaxSim**: for each query token, find the most similar document token, then sum those max-similarities. This is called "late interaction."

This is more expensive to store (many vectors per paper) but captures finer-grained matching than a single CLS vector.

**`colbert_utils.py`**  
- Constants (model name, index settings) and helper functions  
- Contains the label type definitions and all metric functions (same as SciBERT)

**`1-colbert-build-index.py`** — Two steps:
1. Chunks each train paper's text into passages (≤180 wordpieces), saves as a TSV file + a `pid_to_paperid.npy` mapping
2. Builds a ColBERT index over those passages using the `colbert-ai` library

**`2-colbert-eval.py`**  
- Loads the ColBERT index and initializes a `Searcher`  
- For each eval paper, searches the index for top-500 passages, maps them back to paper IDs  
- Computes the same three-type metrics

---

## 6. How to Run Each Baseline

Make sure your `.env` file is set up and you're in the right conda environment before running anything.

### SciBERT

```bash
# Step 1: Generate embeddings (takes ~20–40 min on GPU)
cd codes/SciBERT
python 1-scibert-embeddings.py

# Step 2: Evaluate (takes ~10–20 min)
python 2-scibert-eval.py
```

### SPECTER

```bash
cd codes/SPECTER-Pretrained
python 1-specter-pretrained-embeddings.py
python 2-specter-pretrained-eval.py
```

### SPECTER2

```bash
cd codes/SPECTER2-Pretrained
python 1-specter2-pretrained-embeddings.py
python 2-specter2-pretrained-eval.py
```

### ColBERT

```bash
cd codes/ColBERT
python 1-colbert-build-index.py   # builds the index (slow, 30–60+ min)
python 2-colbert-eval.py          # runs evaluation
```

> **Tip:** Scripts already check if output files exist and skip them, so it's safe to re-run. If you need to re-generate from scratch, delete the relevant output folder.

---

## 7. Results You Need to Produce

After running all four methods, you should have a results table like this for **each of the three label types**:

### Example Format (fill in your numbers)

**Type 1 (Experimental Baseline, binary label == 1)**

| Method | MAP | MRR | nDCG@10 | nDCG@20 | nDCG@30 | R@50 | R@100 | HR@10 | HR@20 |
|--------|-----|-----|---------|---------|---------|------|-------|-------|-------|
| SciBERT | | | | | | | | | |
| SPECTER | | | | | | | | | |
| SPECTER2 | | | | | | | | | |
| ColBERT | | | | | | | | | |

Do the same table for **Type 2** and **Type 3**.

Also produce a summary table (**Recall@100 only**, all types side by side):

| Method | Type 1 R@100 | Type 2 R@100 | Type 3 R@100 |
|--------|-------------|-------------|-------------|
| SciBERT | | | |
| SPECTER | | | |
| SPECTER2 | | | |
| ColBERT | | | |

The numbers are saved automatically by each eval script — both as `.txt` and `.json`. Find them under your `OUTPUT_DIR`:
```
output/
├── dense/
│   ├── SciBERT/evaluation_results/scibert_evaluation_metrics.txt
│   ├── SPECTER-pretrained/evaluation_results/specter_pretrained_metrics.txt
│   ├── SPECTER2-pretrained/evaluation_results/specter2_pretrained_metrics.txt
│   └── ColBERT/evaluation_results/colbert_evaluation_metrics.txt
```

---

## 8. How to Read Your Results

**MAP (Mean Average Precision):** Overall ranking quality. Higher = better ranked results across all positions.

**MRR (Mean Reciprocal Rank):** How quickly you find the *first* must-cite paper. MRR=1.0 means it was always rank 1.

**Recall@K:** Of all must-cite papers for a query, what fraction did you find in the top K results? Recall@100 = 0.30 means you recovered 30% of all must-cite papers in your top 100 results.

**nDCG@K:** Similar to MAP, but focuses on the top-K only, with a penalty for lower-ranked relevant papers.

**HR@K:** Hit Rate — did you find *at least one* must-cite paper in the top K? 1 if yes, 0 if no.

**Key thing to understand:** Recall@K is the most important metric for this task. Missing a must-cite paper is costly, so recovering as many as possible matters more than perfect ranking.

---

## 9. How to Modify the Code

### Changing which K values are evaluated

In each `*_utils.py` or eval script, find this block and edit the lists:

```python
EVAL_K_VALUES = {
    "recall": [10, 50, 100, 500],   # ← add/remove K values here
    "ndcg": [10, 20, 30, 50],
    "hr": [10, 20],
}
```

### Changing the label thresholds

```python
TYPE_2_THRESHOLD = 4.0   # papers with type_2_output >= 4 are "must-cite"
TYPE_3_THRESHOLD = 3.0   # papers cited >= 3 times are "must-cite"
```

Try changing `TYPE_2_THRESHOLD` to `3.0` or `5.0` and see how it affects results.

### Trying a different model (e.g., SciNCL)

For any single-vector model, the pattern is the same. For example, to use `allenai/scibert_scivocab_cased` instead of uncased, change:

```python
MODEL_NAME = "allenai/scibert_scivocab_cased"
```

Or to try SciNCL (`malteos/scincl`), copy the SciBERT scripts, change `MODEL_NAME`, and change the output directory paths.

### Changing batch size

If you get out-of-memory (OOM) errors on GPU, reduce the batch size:

```python
BATCH_SIZE = 64   # was 128, halved
```

### Running on a small subset for testing

To quickly test your code without waiting for all 67,761 papers, add this after loading the parquet:

```python
train_df = pl.read_parquet(TRAIN_PARQUET).head(500)   # only first 500 papers
```

Remove it when running the real experiment.

---

## 10. Common Errors and Fixes

**`ModuleNotFoundError: No module named 'colbert'`**  
→ Run `pip install colbert-ai` and make sure your conda environment is activated.

**`ModuleNotFoundError: No module named 'adapters'`**  
→ Run `pip install adapters` (needed only for SPECTER2).

**`FileNotFoundError: ... train_embeddings.pt`**  
→ You haven't run Step 1 yet. Always run the `1-*.py` script before `2-*.py`.

**`CUDA out of memory`**  
→ Reduce `BATCH_SIZE` to 32 or 64 in the script.

**`KeyError` or `ValueError` when loading parquet**  
→ Check that your `DATA_DIR` in `.env` points to the right folder. Print `df.columns` to verify column names.

**Results seem too low (near zero)**  
→ Check you're running evaluation on `eval_v2.0.parquet` (not train). Also confirm the embeddings loaded correctly by checking `train_embeddings.shape`.

---

## Questions to Bring to Your Next Meeting with Ratul

After getting everything running, think about and prepare answers to:

1. Which method performed best on each label type? Did the same method win across all three types?
2. For Recall@100 — what's the best you achieved? What does that mean in practical terms (how many must-cite papers are you still missing)?
3. SciBERT uses a single CLS vector. ColBERT uses per-token vectors. What do you think the tradeoff is between accuracy and storage cost?
4. SPECTER was trained with citation supervision. Did it outperform SciBERT (which was not)? By how much?
5. What does PPR (Personalized PageRank) do differently from the embedding-based methods above? (Hint: it uses the citation graph structure rather than text.)

---

*Good luck! Ping Ratul on Slack if you're stuck for more than 2 hours on setup or a coding issue.*