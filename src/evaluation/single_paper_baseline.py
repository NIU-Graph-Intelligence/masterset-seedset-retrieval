import os
import json
import torch
import numpy as np
import polars as pl
from pathlib import Path
from dotenv import load_dotenv

import sys
sys.path.append(str(Path(__file__).parent.parent / "set_retrieval"))
from set_query import (
    load_embeddings, load_titles, load_specter2, embed_seeds,
    aggregate_mean, retrieve
)

load_dotenv()
TRAIN_PARQUET = Path(os.getenv("DATA_DIR", "data")) / "train_v2.0.parquet"
OUTPUT_DIR = Path("output/set_retrieval_results")
TOP_K = 50

SEED_SETS = {
    "Query 1: Neural Algorithmic Reasoning": [
        ("Tropical Attention: Neural Algorithmic Reasoning for Combinatorial Algorithms",
         "We introduce Tropical Attention, an attention mechanism grounded in tropical geometry that lifts the attention kernel into tropical projective space, where reasoning is piecewise-linear and 1-Lipschitz."),
        ("Primal-Dual Neural Algorithmic Reasoning",
         "We introduce a general NAR framework grounded in the primal-dual paradigm. By leveraging a bipartite representation between primal and dual variables, we establish an alignment between primal-dual algorithms and Graph Neural Networks."),
        ("Discrete Neural Algorithmic Reasoning",
         "We propose to force neural reasoners to maintain the execution trajectory as a combination of finite predefined states."),
        ("Understanding Transformer Reasoning Capabilities via Graph Algorithms",
         "We investigate transformer scaling regimes able to perfectly solve different classes of algorithmic problems."),
        ("Transformers Can Do Arithmetic with the Right Embeddings",
         "We mend the poor performance of transformers on arithmetic by adding an embedding to each digit that encodes its position relative to the start of the number."),
        ("PUZZLES: A Benchmark for Neural Algorithmic Reasoning",
         "We introduce PUZZLES, a benchmark based on Simon Tatham's Portable Puzzle Collection, aimed at fostering progress in algorithmic and logical reasoning in RL."),
        ("Open-Book Neural Algorithmic Reasoning",
         "We propose a novel open-book learning framework where the network can access and utilize all instances in the training dataset when reasoning for a given instance."),
        ("Deep Equilibrium Algorithmic Reasoning",
         "We study neurally solving algorithms by finding the solution directly by solving an equilibrium equation."),
        ("Simulation of Graph Algorithms with Looped Transformers",
         "We prove by construction that this architecture can simulate Dijkstra's shortest path, Breadth- and Depth-First Search, and Kosaraju's strongly connected components."),
        ("On the Markov Property of Neural Algorithmic Reasoning: Analyses and Methods",
         "We present ForgetNet, which does not use historical embeddings and thus is consistent with the Markov nature of algorithmic reasoning tasks."),
    ],
    "Query 2: LLM Memory": [
        ("Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs",
         "We compare unsupervised fine-tuning and retrieval-augmented generation (RAG). RAG consistently outperforms fine-tuning for both existing and new knowledge."),
        ("Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?",
         "We demonstrate that LLMs struggle to acquire new factual knowledge through fine-tuning, and that learned new knowledge linearly increases hallucination."),
        ("Deciphering the Interplay of Parametric and Non-parametric Memory in Retrieval-augmented Language Models",
         "We explore how the Atlas RAG model decides between parametric and non-parametric knowledge. The model relies more on retrieved context than parametric knowledge."),
        ("IRGen: Generative Modeling for Image Retrieval",
         "We present IRGen, reframing image retrieval as generative modeling using a sequence-to-sequence model achieving state-of-the-art on three benchmarks."),
        ("Transformer Memory as a Differentiable Search Index",
         "We introduce DSI, a new paradigm where a single Transformer maps string queries directly to relevant docids using only its parameters."),
    ],
    "Query 3: Transformer Theory": [
        ("Unique Hard Attention: A Tale of Two Sides",
         "We show that finite-precision transformers with leftmost-hard attention correspond to a strictly weaker fragment of Linear Temporal Logic than those with rightmost-hard attention."),
        ("Logical Languages Accepted by Transformer Encoders with Hard Attention",
         "We study formal languages recognized by UHAT and AHAT transformer encoders and show UHAT encoders recognize all languages definable in first-order logic with unary numerical predicates."),
        ("Representational Strengths and Limitations of Transformers",
         "We establish positive and negative results on the representation power of attention layers focusing on width, depth, and embedding dimension."),
        ("Tighter Bounds on the Expressivity of Transformer Encoders",
         "We identify a variant of first-order logic with counting quantifiers that is simultaneously an upper and lower bound for transformer encoders."),
    ],
}


def run_single_paper_baselines(query_name, seed_texts, train_embs, train_pids,
                                pid_to_title, tokenizer, model):
    print(f"\n{'='*70}")
    print(f"{query_name}")
    print(f"{'='*70}")

    # run each seed individually
    individual_results = {}
    for title, abstract in seed_texts:
        seed_emb = embed_seeds([(title, abstract)], tokenizer, model)
        results = retrieve(seed_emb, aggregate_mean, train_embs, train_pids, k=TOP_K)
        individual_results[title[:50]] = set(pid for pid, _ in results)

    # union of all individual results
    union_set = set().union(*individual_results.values())

    # intersection of all individual results
    intersection_set = set.intersection(*individual_results.values())

    # set-based mean result
    all_seed_embs = embed_seeds(seed_texts, tokenizer, model)
    set_results = retrieve(all_seed_embs, aggregate_mean, train_embs, train_pids, k=TOP_K)
    set_result_pids = set(pid for pid, _ in set_results)

    # discovery candidates - in set results but not in any individual result
    discovery = set_result_pids - union_set

    print(f"\n  Union of individual top-{TOP_K}: {len(union_set)} unique papers")
    print(f"  Intersection of individual top-{TOP_K}: {len(intersection_set)} papers")
    print(f"  Set-based Mean top-{TOP_K}: {len(set_result_pids)} papers")
    print(f"  Discovery candidates (set finds, no individual finds): {len(discovery)}")

    print(f"\n-- Intersection (papers every individual seed finds) --")
    for pid in list(intersection_set)[:10]:
        title = pid_to_title.get(pid, "[unknown]")
        print(f"  {title[:80]}")

    print(f"\n-- Discovery candidates (set query finds, no single seed finds) --")
    for pid in list(discovery)[:10]:
        title = pid_to_title.get(pid, "[unknown]")
        print(f"  {title[:80]}")

    # save to json
    output = {
        "query": query_name,
        "union_count": len(union_set),
        "intersection_count": len(intersection_set),
        "set_mean_count": len(set_result_pids),
        "discovery_count": len(discovery),
        "intersection_papers": list(intersection_set),
        "discovery_papers": list(discovery),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = query_name.lower().replace(" ", "_").replace(":", "").replace("/", "") + "_baseline.json"
    with open(OUTPUT_DIR / filename, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    train_embs, train_pids = load_embeddings()
    pid_to_title = load_titles()
    tokenizer, model = load_specter2()

    for query_name, seed_texts in SEED_SETS.items():
        run_single_paper_baselines(query_name, seed_texts, train_embs, train_pids,
                                   pid_to_title, tokenizer, model)

    print("\nDone!")