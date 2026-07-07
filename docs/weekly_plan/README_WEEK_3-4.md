# Weeks 3–4: Set-Based Query Aggregation

## Where We Are

In weeks 1–2 you ran four baselines (SciBERT, SPECTER, SPECTER2, ColBERT) that each take a **single paper** as the query and retrieve similar papers from the candidate pool. Those baselines give us a performance floor to beat.

Now we change the problem: instead of one query paper, you get a **set of seed papers** — a group of papers that together define a research topic. Your job is to retrieve papers relevant to the *set as a whole*, not just to any single member.

This is the core research question of the project: **how do you aggregate similarity scores across multiple seed papers to produce a single ranked list?**

---

## The Example Queries

Dr. Zhang has provided three set queries. Each is a group of papers that together represent a coherent research topic:

| Query | Topic | Number of Seeds |
|-------|-------|----------------|
| Query 1 | Neural Algorithmic Reasoning | 10 papers |
| Query 2 | LLM Memory | 5 papers |
| Query 3 | Transformer Theory | 4 papers |

**Query 1 — Neural Algorithmic Reasoning:**
- https://mustcite.com/paper/0133e9c0-f893-5504-b8c1-b7b05d869d95
- https://mustcite.com/paper/5bf0c02f-8ed2-5e97-9161-541558feab35
- https://mustcite.com/paper/dfbeee5c-e0e2-5942-9441-280635e57976
- https://mustcite.com/paper/1bc7f6e0-b0ae-5038-8787-5c119e4af13f
- https://mustcite.com/paper/b72c39fd-bd6f-5725-95df-9a2039c6c3a3
- https://mustcite.com/paper/e5b7c941-e9e9-5906-b170-68c1e3e27ad2
- https://mustcite.com/paper/8a7ccafe-8c36-5c4a-b729-470a5c679190
- https://mustcite.com/paper/a161b3e2-3e20-56a8-a30b-a1e2686fd7cb
- https://mustcite.com/paper/5a59220e-42bf-52b8-b824-ff2dd7004d1f
- https://mustcite.com/paper/1d2cc124-9fa8-5a19-b0d0-5a331a65f35

**Query 2 — LLM Memory:**
- https://mustcite.com/paper/a8527971-b28a-5210-85b1-19f74e267a2a
- https://mustcite.com/paper/2d32d8d5-9dfc-50d5-b580-ed0f8f1e5a86
- https://mustcite.com/paper/5dac5135-0eae-5666-a7de-33b5e8fbaf3c
- https://mustcite.com/paper/af017fa8-28d5-5bfb-b237-696b2f85cde9
- https://mustcite.com/paper/2004913a-d0b3-592e-8e33-ded71e6c16af

**Query 3 — Transformer Theory:**
- https://mustcite.com/paper/be37ea0d-a72f-5e0f-9faf-cf49e6bdfa6a
- https://mustcite.com/paper/1c06ca82-15a9-57f7-a022-152aec8d56eb
- https://mustcite.com/paper/625418d7-8628-5405-9b15-65a996001fa1
- https://mustcite.com/paper/0c1590eb-b25c-59a0-bb7a-ef483498a5b6

### Creating Your Own Query Sets

In addition to Dr. Zhang's queries, I suggest you to create your own sets as well. A few guidelines:

- **Cohesive sets:** Pick a research topic you find interesting and gather 5 papers on it. The papers in a set should usually be on the **same research topic** (e.g., "graph neural networks for molecules," "diffusion models for image generation").
- **One mixed-topic set:** Create one special set by **intentionally mixing papers from different research topics** (e.g., 2 papers on reinforcement learning + 2 papers on NLP parsing + 1 paper on computer vision). Our hypothesis is that aggregation strategies will perform poorly on this mixed set — the retrieved papers will be incoherent and not clearly relevant to any single topic. Testing this hypothesis is a useful finding.
- **Before running experiments on your custom sets, verify them with Me (Ratul).** Send him the links and a one-line description of the topic for each set. This avoids wasting time on poorly chosen seeds.

---

## What You Need to Build

### The Big Picture

You already have embeddings for every paper in the train set (from weeks 1–2). For each seed paper in a query set, you can compute a similarity score against every candidate paper. The question is: given K seed papers, each producing a similarity score for each candidate, how do you combine those K scores into one final score per candidate?

```
Seed paper q1  →  sim(q1, c) for every candidate c
Seed paper q2  →  sim(q2, c) for every candidate c
...
Seed paper qK  →  sim(qK, c) for every candidate c
                     ↓
              Aggregation strategy
                     ↓
            Final score(c) for every candidate c
                     ↓
            Ranked list of candidates
```

### The Four Aggregation Strategies to Implement

**1. Mean Aggregation**

The simplest approach. For each candidate, average the similarity scores across all seed papers.

```python
final_score(c) = (1/K) * sum(sim(qi, c) for qi in seeds)
```

When to expect it works well: when all seeds are equally important and the topic is cohesive.

**2. Max Aggregation**

For each candidate, take the highest similarity score across all seed papers.

```python
final_score(c) = max(sim(qi, c) for qi in seeds)
```

When to expect it works well: when the seed set covers diverse sub-topics and a candidate only needs to match one of them.

**3. Soft-AND**

Penalize candidates that score high on only one seed but low on others. This is the opposite philosophy from Max — it rewards candidates that are broadly relevant to the whole set.

One way to implement it:

```python
# For each candidate, compute the mean and min across seeds
mean_score = mean(sim(qi, c) for qi in seeds)
min_score  = min(sim(qi, c) for qi in seeds)
final_score(c) = alpha * mean_score + (1 - alpha) * min_score
```

Start with `alpha = 0.5`. You can experiment with other values later.

When to expect it works well: when the seed set is narrow/focused and you want candidates relevant to all seeds, not just one.

**4. Cluster-then-Aggregate**

If the seed set has sub-groups (e.g., 10 seeds where 4 are about architecture and 6 are about training), cluster the seeds first, take max within each cluster, then average across clusters.

```python
# Step 1: Cluster seed embeddings (e.g., k-means with k=2 or 3)
clusters = cluster(seed_embeddings, n_clusters=min(3, len(seeds)))

# Step 2: For each cluster, take max sim across seeds in that cluster
cluster_scores = []
for cluster in clusters:
    cluster_score = max(sim(qi, c) for qi in cluster)
    cluster_scores.append(cluster_score)

# Step 3: Average across clusters
final_score(c) = mean(cluster_scores)
```

Use `sklearn.cluster.KMeans` for clustering. Start with `n_clusters = min(3, len(seeds))`.

When to expect it works well: when seeds span multiple sub-topics and you want balanced coverage.

---

## Implementation Plan

### Step 1: Build the Set Query Pipeline (Days 1–2)

Start with the model whose code you're most comfortable with (probably SPECTER2 since it performed best in weeks 1–2).

**What to write:**

Create a new script, e.g., `set_query_eval.py`, that:

1. Loads the pre-computed train embeddings (you already generated these)
2. Loads the pre-computed eval embeddings
3. For a given set of seed paper IDs:
   - Looks up each seed's embedding
   - Computes cosine similarity between each seed and all candidates in the train set
   - Applies an aggregation strategy (start with Mean)
   - Returns a ranked list of candidates

**Skeleton:**

```python
import torch
import numpy as np
import json

def load_embeddings(embed_dir, split="train"):
    data = torch.load(embed_dir / f"{split}_embeddings.pt", map_location="cpu", weights_only=False)
    embeddings = data["embeddings"].numpy()
    paper_ids = data["paper_ids"]
    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms
    return embeddings, paper_ids

def get_seed_embeddings(seed_ids, all_embeddings, all_paper_ids):
    """Look up embeddings for the seed papers."""
    id_to_idx = {pid: i for i, pid in enumerate(all_paper_ids)}
    indices = [id_to_idx[sid] for sid in seed_ids if sid in id_to_idx]
    return all_embeddings[indices]

def aggregate_mean(seed_embs, candidate_embs):
    """Mean aggregation: average similarity across seeds."""
    # seed_embs: (K, D), candidate_embs: (N, D)
    # sim_matrix: (K, N) — each row is one seed's similarity to all candidates
    sim_matrix = seed_embs @ candidate_embs.T
    return sim_matrix.mean(axis=0)  # (N,)

def aggregate_max(seed_embs, candidate_embs):
    sim_matrix = seed_embs @ candidate_embs.T
    return sim_matrix.max(axis=0)   # (N,)

def aggregate_soft_and(seed_embs, candidate_embs, alpha=0.5):
    sim_matrix = seed_embs @ candidate_embs.T
    mean_scores = sim_matrix.mean(axis=0)
    min_scores = sim_matrix.min(axis=0)
    return alpha * mean_scores + (1 - alpha) * min_scores

def retrieve_for_set(seed_ids, aggregation_fn, train_embs, train_ids,
                     all_embs, all_ids, top_k=100):
    """Retrieve top-K candidates for a seed set."""
    seed_embs = get_seed_embeddings(seed_ids, all_embs, all_ids)
    scores = aggregation_fn(seed_embs, train_embs)

    # Exclude seed papers from results
    seed_set = set(seed_ids)
    ranked_indices = np.argsort(-scores)
    results = []
    for idx in ranked_indices:
        pid = train_ids[idx]
        if pid not in seed_set:
            results.append((pid, float(scores[idx])))
        if len(results) >= top_k:
            break
    return results
```

**Where do the seed papers come from?** The seed paper IDs from Dr. Zhang's queries are UUIDs. Some of them might be in the train set, some in the eval set. First check which set they belong to:

```python
train_ids_set = set(train_paper_ids)
eval_ids_set = set(eval_paper_ids)
for sid in seed_ids:
    if sid in train_ids_set:
        print(f"{sid} → train")
    elif sid in eval_ids_set:
        print(f"{sid} → eval")
    else:
        print(f"{sid} → NOT FOUND")
```

You may need to load embeddings from **both** train and eval to look up seed embeddings, but you always search **against the train set** (candidate pool).

### Step 2: Implement All Four Strategies (Days 3–4)

Once Mean works end-to-end, implement Max, Soft-AND, and Cluster-then-Aggregate. For Cluster-then-Aggregate:

```bash
pip install scikit-learn
```

### Step 3: Run on All Query Sets (Days 5–6)

For each query set (Dr. Zhang's 3 + your own) × 4 strategies:

1. Retrieve the top 100 candidates
2. Inspect the results qualitatively — do the retrieved papers actually look relevant to the topic?
3. Record the top-20 papers for each run (title + score)

Since we don't have ground-truth "must-cite" labels for set queries (that's a different setting from the single-paper eval), the evaluation at this stage is **qualitative**. Look at the titles and ask:
- Does this paper seem relevant to the topic (e.g., "Neural Algorithmic Reasoning")?
- Are there obvious papers missing?
- Where do the strategies disagree? Those disagreements are the interesting cases.

### Step 4: Compare Strategies (Days 7–8)

Build a comparison notebook or script that:

1. **Overlap analysis:** For each query, how much do the top-50 lists overlap across strategies? Use Jaccard similarity:
   ```python
   def jaccard(set_a, set_b):
       return len(set_a & set_b) / len(set_a | set_b)
   ```

2. **Unique contributions:** Which papers appear in Max's top-50 but not Mean's? Vice versa? These are the cases that reveal when each strategy adds value.

3. **Single-paper baseline comparison:** For each seed paper individually, run the single-paper retrieval you built in weeks 1–2. Then compare: does the set-based approach find papers that no single seed retrieves on its own? This is the key value proposition.

---

## Comparison with Single-Paper Baselines

This is important. For each query set, you should also run each seed paper **individually** through your SPECTER2 (or whichever model you use) pipeline and get its top-100. Then:

- **Union baseline:** Take the union of all individual top-100 lists. How does this compare to the set-based top-100?
- **Intersection:** Which papers appear in every individual seed's top-100? These are probably the most obviously relevant.
- **Set-only finds:** Papers in the set-based top-100 that don't appear in *any* individual seed's top-100. These are the "discovery" candidates — papers the set query finds that no single seed would.

---

## Deliverables by End of Week 4

1. **Working code:** A script that takes a list of seed paper IDs and an aggregation strategy, and returns a ranked list of candidates.
2. **Results notebook:** Side-by-side comparisons of all 4 strategies on Dr. Zhang's 3 queries + your custom sets, including overlap analysis and comparison with single-paper baselines. Pay special attention to the mixed-topic set — does it confirm our hypothesis that mixed topics degrade results?
3. **Observations document (half a page):** What did you notice? Which strategies disagreed? Did any strategy consistently find papers the others missed? How did the mixed-topic set behave differently? Bring this to your meeting with Me (Ratul).

---

## How This Connects to What Comes Next

**Weeks 5–6 (Graph Space):** You'll add a second retrieval signal — Personalized PageRank on the citation graph. The seed set becomes the personalization vector for the random walk. Then you'll fuse the semantic rankings (what you're building now) with the graph rankings using Reciprocal Rank Fusion (RRF).

**Weeks 7–8 (Evaluation):** You'll build an LLM-as-judge evaluation framework to systematically assess whether the retrieved papers are genuinely relevant to the seed set.

The aggregation code you write now will be reused directly in those later phases — so write it cleanly, with clear function signatures and docstrings.

---

## Tips

- **Start simple.** Get Mean working end-to-end before touching the other strategies. A working pipeline you can iterate on beats four half-finished implementations.
- **Print intermediate results.** After computing similarities, print the top 5 with their titles. This catches bugs fast (e.g., if your top result is the seed paper itself, you forgot to exclude seeds).
- **Use SPECTER2 as your primary model.** It had the best pretrained performance in weeks 1–2. You can always re-run with other models later.
- **Commit frequently.** Small, focused commits. "Add mean aggregation" is better than "add all aggregation strategies and fix bugs."