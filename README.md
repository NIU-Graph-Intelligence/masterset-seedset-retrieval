# MasterSet Seed-Set Retrieval

Set-based citation retrieval over the MasterSet benchmark. Given a *set* of
seed papers that together define a research topic, retrieve the papers a
researcher working on that topic should cite.

Three signals are combined:

1. **Semantic** — SPECTER2 embeddings, four strategies for aggregating
   per-seed similarity into one score (Mean, Max, Soft-AND, Cluster).
2. **Graph** — Personalized PageRank over the intra-pool citation graph,
   restarting from the seeds.
3. **Fusion** — Reciprocal Rank Fusion over the two.

Because MasterSet's must-cite labels are defined for *single-paper* queries,
set retrieval has no ground truth. An LLM judge supplies the relevance signal.

---

## Setup

```bash
git clone <repo> && cd masterset-seedset-retrieval
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # then edit .env for this machine
```

`.env` is the **only** file containing absolute paths. Moving to a different
server means editing `.env` and nothing else. Values may be absolute, or
relative to the repository root.

Two locations, deliberately kept separate:

| What | Where | Why |
|---|---|---|
| `seeds.json` | **repo** — `data/seeds.json` | versioned with the code; it defines the experiment |
| corpus parquet, embeddings | **`DATA_DIR`** — external | shared dataset, often on another volume, too large to track |

So `SEEDS_JSON` resolves against the repository root, while `CORPUS_PARQUET`
resolves against `DATA_DIR`. Setting `DATA_DIR` to an external path does not
drag `seeds.json` out of the repo with it.

Two helper scripts you will use repeatedly:

```bash
python scripts/inspect_data.py                    # what is on disk, and the .env to match
python scripts/find_papers.py "in-context learning"  # find seeds that exist in the pool
```

`find_papers.py` searches the candidate pool only, so anything it prints is
guaranteed to resolve in every phase. `--json` emits ready-to-paste
`seeds.json` entries; `--min-in-degree` filters for well-cited papers, which
give PPR more to work with.

Discover what is on disk and get the matching `.env` lines printed for you:

```bash
python scripts/inspect_data.py
```

It lists every parquet with its row count and columns, identifies which one has
a usable references column (and the cited-id key inside it), locates the
embedding files, and prints a suggested `.env`. Then confirm nothing is
`[MISSING]`:

```bash
python -c "from src.common import config; print(config.describe())"
```

---

## Running

```bash
bash scripts/run_all.sh          # everything, in order
```

Or phase by phase — each script is independent and re-runnable:

```bash
# Phase 2a — citation graph (needed by seed validation)
python -m src.phase2_graph.build_citation_graph

# Phase 1a — validate seeds. ALWAYS run this after editing seeds.json.
python -m src.phase1_semantic.validate_seeds --strict

# Phase 1b — semantic retrieval
python -m src.phase1_semantic.run_semantic_retrieval

# Phase 2b — Personalized PageRank
python -m src.phase2_graph.run_ppr_retrieval --damping 0.85 0.90 0.95

# Phase 3 — RRF fusion
python -m src.phase3_fusion.run_fusion --rrf-k 60 20 100

# Phase 4 — LLM judge (see the cost note below)
python -m src.phase4_judge.run_judge --dry-run     # cost estimate first
python -m src.phase4_judge.run_judge
python -m src.phase4_judge.run_consistency

# Phase 5 — analysis and paper tables
python -m src.phase5_analysis.compute_overlap
python -m src.phase5_analysis.compute_metrics
python -m src.phase5_analysis.export_tables
```

Every script takes `--topic "<name>"` (repeatable) to run one query set, and
`--help` for its full options.

---

## Layout

```
.env                    absolute paths — the only machine-specific file
data/
    seeds.json          all query sets; the single source of truth
    train_v1.2.parquet  candidate pool
src/
    common/             shared by every phase
        config.py         resolves .env into paths
        seeds.py          loads and validates seeds.json
        corpus.py         parquet access (cached)
        embeddings.py     corpus matrix + seed vector lookup
        metrics.py        jaccard, precision@k, avg relevance
        io_utils.py       JSON/JSONL, slugs, run manifests
    phase1_semantic/
        validate_seeds.py           run first, after any seeds.json edit
        aggregate.py                the four strategies (pure functions)
        run_semantic_retrieval.py
    phase2_graph/
        build_citation_graph.py
        run_ppr_retrieval.py
    phase3_fusion/
        fuse.py                     RRF (pure function)
        run_fusion.py
    phase4_judge/
        judge_client.py             prompt, call, parse, cache
        run_judge.py
        run_consistency.py
    phase5_analysis/
        compute_overlap.py
        compute_metrics.py
        export_tables.py            emits LaTeX for the paper
scripts/
    run_all.sh
    check_seed_leakage.py           standing verification gate
outputs/                gitignored; one subdirectory per phase
```

### Design rules

**One script, one job.** No script both computes embeddings and evaluates
them. Retrieval, fusion, judging and analysis are separate entry points, so any
one of them can be re-run without redoing the others.

**Phases talk through files, never imports.** Phase 3 reads phase 1's and
phase 2's saved JSON rather than importing their functions. Re-running fusion
with a different `--rrf-k` takes seconds and touches no GPU. No phase module
imports from another phase module; anything genuinely shared lives in
`src/common/`.

**No data in code.** Seed papers exist only in `data/seeds.json`. Every path
comes from `.env` via `src/common/config.py`.

**Every phase writes a manifest.** `manifest.json` next to each phase's results
records the parameters that produced them, so any number in the paper can be
traced back to the run that generated it.

---

## Two things worth knowing before you interpret results

### Seeds are looked up, not re-encoded

Every seed has a `paper_id` in the candidate pool, so its SPECTER2 vector is
already in the precomputed matrix. Re-encoding a seed from its title and
abstract produces a *different* vector than the one the index was built with,
because the corpus encoder saw a particular text format and truncation. The
pipeline looks seeds up by ID, and fails loudly if one is absent rather than
quietly encoding it. `--allow-encode-fallback` opts into encoding, and every
seed encoded that way is recorded in the output.

### The pool and the eval set are different files

Under the CoreML-168K setting the candidate pool is cut at year ≤ 2025
(145,948 papers, `candidate_pool_v7.0.parquet` / `candidates_embeddings.pt`)
while evaluation queries are 2026 papers (`eval_v7.0.parquet` /
`eval_embeddings.pt`). Retrieval always ranks the **pool**.

`CORPUS_PARQUET` and `EMBEDDINGS_FILE` must therefore describe the *same*
papers. Pointing `CORPUS_PARQUET` at `all_papers*.parquet` (168,837) while the
embeddings hold 145,948 vectors leaves ~23k pool papers unreachable, with no
error. `scripts/inspect_data.py` matches row counts against vector counts and
refuses to suggest a mismatched pair.

A 2026 seed paper is legitimately *not* in the pool. It therefore:

- **has no vector in `candidates_embeddings.pt`** — so `SEED_EMBEDDINGS_FILE`
  provides a secondary lookup (`eval_embeddings.pt`). Both files come from the
  same encoder, so the vectors are comparable. This affects seed lookup only.
- **is not a node in the citation graph**, which is built from the pool — so it
  contributes nothing to PPR. This is the "query has no graph" problem, and it
  is reported rather than hidden: `validate_seeds.py` marks the seed
  `vector from 'eval', not the pool (so not a graph node)`, and phase 1 records
  a `seed_vector_sources` map in its output.

A query set whose seeds are mostly 2026 papers will have a strong semantic
signal and a weak or empty graph signal. That is a real property of the setting,
not a bug — but it must be stated when the results are reported.

### PageRank damping vs. teleport probability

networkx calls its argument `alpha`, and it is the **damping factor** — the
probability of *following an edge*. The teleport probability is `1 - damping`.

A reported teleport probability of 0.15 therefore means:

```python
nx.pagerank(G, alpha=0.85, personalization=...)   # damping 0.85, teleport 0.15
```

Passing `0.15` as networkx's `alpha` inverts it: the walk returns to the seeds
85% of the time and barely traverses the graph, so results sit in the seeds'
immediate neighbourhood and are nearly invariant to the parameter. **A damping
sweep that shows "no effect" is the symptom of this.** This pipeline uses
`--damping` and prints both numbers on every run.

---

## Judging cost

The same paper appears in the semantic, graph and fused lists for a query, so
judgments are cached by `(topic, seed set, paper, prompt version, model)`.
Across 12 query sets at top-50, roughly 75% of ranked slots are duplicates.

```bash
ollama pull qwen2.5:7b-instruct          # installing Ollama does NOT pull models
python scripts/check_judge.py --n 4      # server up? model present? replies parse?
python -m src.phase4_judge.run_judge --dry-run
```

`check_judge.py` runs real judgments against two fixed samples -- one on-topic,
one unrelated -- and fails loudly if the server is unreachable, the model is not
installed, or replies cannot be parsed. It also reports seconds per judgment, so
`--estimate <N>` converts the dry-run count into a wall-clock figure. Run it
before detaching a long job.

`--dry-run` reports the unique-judgment count before anything runs. The cache
(`outputs/phase4_judge/judge_cache.jsonl`) also makes an interrupted run
resumable at no cost. `run_consistency.py` deliberately bypasses it — a cached
judgment would report perfect agreement by construction.

The judge runs at `temperature=0` with a fixed seed.

---

## Verification

```bash
python scripts/check_seed_leakage.py
```

Asserts that no seed paper appears in any result list, across every phase. A
seed scores 1.0 against itself, so one leak puts it at rank 1 and inflates
every metric. Run this after any change to retrieval.

`validate_seeds.py` classifies each query set:

| Status | Meaning |
|---|---|
| `OK` | every seed is in the pool and in the graph |
| `PARTIAL_GRAPH` | some seeds missing from the graph; PPR runs on the rest |
| `NO_GRAPH_SIGNAL` | no seed in the graph — PPR returns nothing and fusion collapses to the semantic list |
| `MISSING_FROM_POOL` | a `paper_id` does not resolve; fix `seeds.json` |

`NO_GRAPH_SIGNAL` is recorded explicitly in the phase 3 output rather than
being allowed to look like a normal fusion result.
