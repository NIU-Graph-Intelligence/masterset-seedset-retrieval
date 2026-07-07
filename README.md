# MasterSet: Set-Based Retrieval

Set-based citation retrieval over the [MasterSet](https://mustcite.com) benchmark. Given a set of seed papers representing a research topic, retrieve relevant papers from the candidate pool using aggregated semantic similarity.

## Repository Structure

```
masterset-seedset-retrieval/
├── src/                    # All scripts
│   ├── embeddings/         # Embedding generation (SciBERT, SPECTER, SPECTER2, ColBERT)
│   ├── set_retrieval/      # Set-based query aggregation (Mean, Max, Soft-AND, Cluster)
│   └── evaluation/         # Evaluation and comparison utilities
├── data/                   # Data files (not tracked)
│   └── train_eval_set/
│       └── v2.0/
|           ├── all_papers_with_refs_and_labels_v2.0.parquet
│           ├── train_v2.0.parquet
│           └── eval_v2.0.parquet
├── output/                 # Generated embeddings, indices, and results
├── notebooks/              # Analysis and visualization notebooks if you prefer notebooks
├── .env                    # Local config (DATA_DIR, OUTPUT_DIR)
└── README.md
```

## Setup

```bash
conda create -n masterset python=3.10 -y
conda activate masterset
pip install torch transformers polars numpy tqdm python-dotenv faiss-cpu adapters scikit-learn
```

Create a `.env` file:

```
DATA_DIR=/path/to/data
OUTPUT_DIR=./output
```