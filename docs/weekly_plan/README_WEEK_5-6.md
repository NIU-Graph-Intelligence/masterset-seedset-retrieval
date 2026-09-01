# Weeks 5–6: Graph-Based Retrieval & Rank Fusion

## Important: New Data Version

New parquet files — **version 1.2** (train, eval, and all papers) — are shared in the drive link. Use these going forward instead of v2.0. The labels are mostly the same, but since the data has changed, you need to:

1. **Re-generate embeddings** for train and eval using the new v1.2 parquet files (same scripts from weeks 1–2, just change the input file path).
2. **Re-run your weeks 3–4 set retrieval and overlap analysis** with the new embeddings.

This shouldn't take much time — you already have the code, you're just swapping the input file. Update your `.env` or file paths to point to the v1.2 parquets. Do this **before** starting the graph work below.

---

## Where We Are

In weeks 1–2 you built single-paper baselines using semantic similarity (embeddings). In weeks 3–4 you extended that to set-based queries using four aggregation strategies. All of that lives in **semantic space** — the model reads the text and decides how similar two papers are.

Now we add a second signal: the **citation graph**. The idea is simple — papers that are connected through citations are likely related, even if their text doesn't look similar. A paper might use very different terminology but still be closely related because it builds on the same line of work.

By combining both signals (text similarity + citation graph), we should retrieve papers that neither signal would find alone.

---

## Key Concepts

### The Citation Graph

Our dataset has ~67,000 train papers. Each paper has a list of references. We can build a **directed graph** where:

- Each paper is a node
- An edge from paper A to paper B means A cites B

Not all references point to papers in our pool — only those with a `matched_paper_id` are in the graph. Ignore the rest.

### Personalized PageRank (PPR)

Regular PageRank finds the most "important" nodes in a graph — the ones that many other nodes point to. **Personalized** PageRank does the same but biased toward a specific set of starting nodes.

Think of it like this: imagine you're randomly walking through the citation graph. At each step you either follow a random citation link, or with some probability (called **alpha**, typically 0.15) you "teleport" back to one of your seed papers. After many steps, the nodes you visit most often are the ones most relevant to your seeds — according to the graph structure.

For a set of seed papers `{s1, s2, ..., sK}`, the personalization vector gives equal weight to each seed:

```python
personalization = {}
for paper_id in all_papers:
    if paper_id in seed_set:
        personalization[paper_id] = 1.0 / len(seed_set)
    else:
        personalization[paper_id] = 0.0
```

PPR returns a score for every node in the graph. Higher score = more relevant to the seed set according to citation structure.

### Reciprocal Rank Fusion (RRF)

You now have two ranked lists for the same query set:
1. **Semantic ranking** — from your aggregation strategies (weeks 3–4)
2. **Graph ranking** — from PPR

RRF is a simple way to merge them. For each candidate paper, its fused score is:

```
RRF_score(c) = 1/(k + rank_semantic(c)) + 1/(k + rank_graph(c))
```

where `k` is a constant (typically 60). Papers that rank high in **both** lists get the highest fused scores. Papers that rank high in only one list still get some credit.

If a paper appears in one list but not the other, use a large default rank (e.g., 1000).

---

## What You Need to Build

### Step 1: Build the Citation Graph (Days 1–2)

Create a script `src/graph/build_graph.py` that:

1. Loads `train_v1.2.parquet`
2. For each paper, reads its `references` field (JSON list)
3. For each reference with a `matched_paper_id` that exists in the train set, adds a directed edge
4. Saves the graph

```python
import networkx as nx
import polars as pl
import json
from pathlib import Path

def build_citation_graph(train_parquet_path):
    df = pl.read_parquet(train_parquet_path)
    train_ids = set(df["paper_id"].to_list())
    
    G = nx.DiGraph()
    G.add_nodes_from(train_ids)
    
    for row in df.iter_rows(named=True):
        paper_id = row["paper_id"]
        refs = row.get("references")
        if refs is None:
            continue
        if isinstance(refs, str):
            refs = json.loads(refs)
        for ref in refs:
            cited_id = ref.get("matched_paper_id")
            if cited_id and cited_id in train_ids:
                G.add_edge(paper_id, cited_id)
    
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G
```

**Things to check after building:**
- How many edges does the graph have? (This tells us how connected the papers are.)
- What fraction of papers have zero outgoing edges? Zero incoming edges?
- What's the average number of citations per paper?

Print these stats — they'll help you understand the graph before running PPR.

### Step 2: Run Personalized PageRank (Days 3–4)

Create `src/graph/ppr_retrieval.py`:

```python
import networkx as nx

def ppr_retrieve(G, seed_ids, alpha=0.15, top_k=100):
    """Run Personalized PageRank from seed papers."""
    # Build personalization vector
    personalization = {node: 0.0 for node in G.nodes()}
    valid_seeds = [s for s in seed_ids if s in G]
    
    if not valid_seeds:
        print("Warning: no seed papers found in graph")
        return []
    
    for sid in valid_seeds:
        personalization[sid] = 1.0 / len(valid_seeds)
    
    # Run PPR
    ppr_scores = nx.pagerank(G, alpha=alpha, personalization=personalization)
    
    # Sort and exclude seeds
    seed_set = set(seed_ids)
    ranked = sorted(ppr_scores.items(), key=lambda x: -x[1])
    results = [(pid, score) for pid, score in ranked if pid not in seed_set]
    
    return results[:top_k]
```

**Important notes:**
- `alpha` is the teleport probability. Higher alpha = stays closer to seeds. Lower alpha = explores further in the graph. Start with 0.15 (the standard value). Try 0.1 and 0.2 as well.
- NetworkX's `pagerank` can be slow on 67K nodes. If it takes too long, try `pagerank_scipy` instead — same results, faster computation.
- Some seed papers from Dr. Zhang's queries might not be in the train set. Check and handle this gracefully (print a warning, skip missing seeds).

### Step 3: Implement Rank Fusion (Days 5–6)

Create `src/evaluation/rank_fusion.py`:

```python
def reciprocal_rank_fusion(semantic_results, graph_results, k=60):
    """Fuse two ranked lists using RRF.
    
    semantic_results: list of (paper_id, score) from set aggregation
    graph_results: list of (paper_id, score) from PPR
    """
    # Build rank maps
    semantic_ranks = {pid: rank for rank, (pid, _) in enumerate(semantic_results, 1)}
    graph_ranks = {pid: rank for rank, (pid, _) in enumerate(graph_results, 1)}
    
    # All candidate papers from both lists
    all_papers = set(semantic_ranks.keys()) | set(graph_ranks.keys())
    
    default_rank = 1000  # for papers missing from one list
    
    fused = {}
    for pid in all_papers:
        sem_rank = semantic_ranks.get(pid, default_rank)
        graph_rank = graph_ranks.get(pid, default_rank)
        fused[pid] = 1.0 / (k + sem_rank) + 1.0 / (k + graph_rank)
    
    # Sort by fused score descending
    ranked = sorted(fused.items(), key=lambda x: -x[1])
    return ranked
```

### Step 4: Run Experiments & Compare (Days 7–8)

For each query set, produce three ranked lists:

1. **Semantic only** — your best aggregation strategy from weeks 3–4
2. **Graph only** — PPR
3. **Fused** — RRF of semantic + graph

Compare them the same way you compared strategies in weeks 3–4:
- Jaccard overlap between the three lists (top 50)
- Unique papers each approach finds
- Qualitative check — look at the titles. Does the graph find papers that text similarity missed? Does fusion improve over either alone?

Also run on the mixed-topic set. PPR might actually handle mixed topics differently from semantic similarity — citation links can connect papers across topics if they share methodology even when their text is about different things.

---

## What to Look For

The interesting questions for your observations document:

1. **Does PPR find different papers than semantic retrieval?** If the overlap is low, that means the two signals are complementary — exactly what we want.
2. **Does fusion (RRF) improve over either signal alone?** The hope is that fused results contain the best of both.
3. **How does alpha affect PPR results?** Lower alpha explores more of the graph; higher alpha stays near seeds.
4. **What about papers with few citations?** PPR can't find papers that aren't well-connected in the graph. Are there good papers that PPR misses because they're isolated nodes?
5. **How does the mixed-topic set behave?** PPR doesn't care about text — it follows citation links. Does it handle mixed topics better or worse than semantic strategies?

---

## Deliverables by End of Week 6

1. **Graph construction script** with printed stats (node count, edge count, average degree).
2. **PPR retrieval script** that takes seed paper IDs and returns a ranked list.
3. **RRF fusion script** that combines semantic and graph rankings.
4. **Results comparison** — side-by-side of semantic-only vs graph-only vs fused, for all query sets. Same format as your weeks 3–4 notebook.
5. **Observations document (half a page)** answering the questions above.

As before: make a pull request to main when your code is ready, and upload results to the shared drive.

---

## How This Connects to What Comes Next

**Weeks 7–8 (LLM-as-Judge Evaluation):** Right now your evaluation is qualitative — you look at titles and judge relevance yourself. In weeks 7–8 you'll build an automated evaluation using an LLM to judge whether each retrieved paper is relevant to the seed set. This gives you a scalable way to compare strategies with actual relevance scores instead of eyeballing.

---

## Setup

Install NetworkX if you don't have it:

```bash
pip install networkx
```

You already have everything else from weeks 3–4.

---

## Tips

- **Build the graph once, save it.** Graph construction reads the full parquet and is slow. Save the graph object with `nx.write_gpickle(G, "output/citation_graph.gpickle")` and load it in subsequent scripts.
- **Check your seed papers.** Before running PPR, confirm which seeds are actually in the graph. If a seed has zero citations in the graph, PPR can't do much with it.
- **Start with semantic Mean + PPR fusion.** Get one fusion pipeline working end-to-end before trying other combinations.
- **Commit frequently.** Same as before — small, focused commits.