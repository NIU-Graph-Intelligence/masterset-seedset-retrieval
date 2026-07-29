import os
import json
import pickle
import torch
import numpy as np
import polars as pl
from pathlib import Path
from dotenv import load_dotenv

import sys
sys.path.append(str(Path(__file__).parent.parent / "set_retrieval"))
sys.path.append(str(Path(__file__).parent.parent / "graph"))

from set_query import (
    load_embeddings, load_titles, load_specter2, embed_seeds,
    aggregate_mean, retrieve
)
from ppr_retrieval import load_graph, ppr_retrieve

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path("output")
RESULTS_DIR = OUTPUT_DIR / "fusion_results"
TRAIN_PARQUET = DATA_DIR / "train_v1.2.parquet"
TOP_K = 50


def reciprocal_rank_fusion(semantic_results, graph_results, k=60):
    semantic_ranks = {pid: rank for rank, (pid, _) in enumerate(semantic_results, 1)}
    graph_ranks = {pid: rank for rank, (pid, _) in enumerate(graph_results, 1)}
    all_papers = set(semantic_ranks.keys()) | set(graph_ranks.keys())
    default_rank = 1000
    fused = {}
    for pid in all_papers:
        sem_rank = semantic_ranks.get(pid, default_rank)
        graph_rank = graph_ranks.get(pid, default_rank)
        fused[pid] = 1.0 / (k + sem_rank) + 1.0 / (k + graph_rank)
    return sorted(fused.items(), key=lambda x: -x[1])


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def print_results(results, pid_to_title, top_n=10):
    for rank, (pid, score) in enumerate(results[:top_n], 1):
        title = pid_to_title.get(pid, "[unknown]")
        print(f"  {rank:2d}. [{score:.6f}] {title[:70]}")


def save_json(query_name, semantic, graph, fused, pid_to_title):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = query_name.lower().replace(" ", "_").replace(":", "").replace("/", "") + "_fusion.json"

    def to_list(results):
        return [{"rank": i+1, "paper_id": pid, "title": pid_to_title.get(pid, "[unknown]"), "score": score}
                for i, (pid, score) in enumerate(results)]

    sem_set = set(pid for pid, _ in semantic)
    graph_set = set(pid for pid, _ in graph)
    fused_set = set(pid for pid, _ in fused)

    data = {
        "query": query_name,
        "semantic_only": to_list(semantic),
        "graph_only": to_list(graph),
        "fused_rrf": to_list(fused),
        "overlap": {
            "semantic_vs_graph_jaccard": round(jaccard(sem_set, graph_set), 3),
            "semantic_vs_fused_jaccard": round(jaccard(sem_set, fused_set), 3),
            "graph_vs_fused_jaccard": round(jaccard(graph_set, fused_set), 3),
            "graph_only_papers": [
                {"paper_id": pid, "title": pid_to_title.get(pid, "[unknown]")}
                for pid in graph_set - sem_set
            ],
        }
    }

    with open(RESULTS_DIR / filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved to {RESULTS_DIR / filename}")


SEED_SETS = {
    "Query 1: Neural Algorithmic Reasoning": [
        ("Tropical Attention: Neural Algorithmic Reasoning for Combinatorial Algorithms",
         "We introduce Tropical Attention, an attention mechanism grounded in tropical geometry."),
        ("Primal-Dual Neural Algorithmic Reasoning",
         "We introduce a general NAR framework grounded in the primal-dual paradigm."),
        ("Discrete Neural Algorithmic Reasoning",
         "We propose to force neural reasoners to maintain the execution trajectory as finite predefined states."),
        ("Understanding Transformer Reasoning Capabilities via Graph Algorithms",
         "We investigate transformer scaling regimes able to solve different classes of algorithmic problems."),
        ("Transformers Can Do Arithmetic with the Right Embeddings",
         "We mend the poor performance of transformers on arithmetic by adding position embeddings to each digit."),
        ("PUZZLES: A Benchmark for Neural Algorithmic Reasoning",
         "We introduce PUZZLES, a benchmark aimed at fostering progress in algorithmic and logical reasoning in RL."),
        ("Open-Book Neural Algorithmic Reasoning",
         "We propose a novel open-book learning framework where the network can access all training instances."),
        ("Deep Equilibrium Algorithmic Reasoning",
         "We study neurally solving algorithms by finding the solution directly by solving an equilibrium equation."),
        ("Simulation of Graph Algorithms with Looped Transformers",
         "We prove this architecture can simulate Dijkstra, BFS, DFS, and Kosaraju's algorithm."),
        ("On the Markov Property of Neural Algorithmic Reasoning: Analyses and Methods",
         "We present ForgetNet, consistent with the Markov nature of algorithmic reasoning tasks."),
    ],
    "Query 2: LLM Memory": [
        ("Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs",
         "RAG consistently outperforms fine-tuning for both existing and new knowledge."),
        ("Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?",
         "LLMs struggle to acquire new factual knowledge through fine-tuning, linearly increasing hallucination."),
        ("Deciphering the Interplay of Parametric and Non-parametric Memory in Retrieval-augmented Language Models",
         "The model relies more on retrieved context than its parametric knowledge."),
        ("IRGen: Generative Modeling for Image Retrieval",
         "We reframe image retrieval as generative modeling using a sequence-to-sequence model."),
        ("Transformer Memory as a Differentiable Search Index",
         "We introduce DSI, mapping string queries directly to relevant docids using only model parameters."),
    ],
    "Query 3: Transformer Theory": [
        ("Unique Hard Attention: A Tale of Two Sides",
         "Leftmost-hard attention corresponds to a strictly weaker fragment of Linear Temporal Logic."),
        ("Logical Languages Accepted by Transformer Encoders with Hard Attention",
         "UHAT encoders recognize all languages definable in first-order logic with unary numerical predicates."),
        ("Representational Strengths and Limitations of Transformers",
         "We establish positive and negative results on the representation power of attention layers."),
        ("Tighter Bounds on the Expressivity of Transformer Encoders",
         "A variant of first-order logic with counting quantifiers bounds transformer encoders."),
    ],
}

SEED_IDS = {
    "Query 1: Neural Algorithmic Reasoning": [
        "0133e9c0-f893-5504-b8c1-b7b05d869d95",
        "5bf0c02f-8ed2-5e97-9161-541558feab35",
        "dfbeee5c-e0e2-5942-9441-280635e57976",
        "1bc7f6e0-b0ae-5038-8787-5c119e4af13f",
        "b72c39fd-bd6f-5725-95df-9a2039c6c3a3",
        "e5b7c941-e9e9-5906-b170-68c1e3e27ad2",
        "8a7ccafe-8c36-5c4a-b729-470a5c679190",
        "a161b3e2-3e20-56a8-a30b-a1e2686fd7cb",
        "5a59220e-42bf-52b8-b824-ff2dd7004d1f",
        "1d2cc124-9fa8-5a19-b0d0-5a331a65f35",
    ],
    "Query 2: LLM Memory": [
        "a8527971-b28a-5210-85b1-19f74e267a2a",
        "2d32d8d5-9dfc-50d5-b580-ed0f8f1e5a86",
        "5dac5135-0eae-5666-a7de-33b5e8fbaf3c",
        "af017fa8-28d5-5bfb-b237-696b2f85cde9",
        "2004913a-d0b3-592e-8e33-ded71e6c16af",
    ],
    "Query 3: Transformer Theory": [
        "be37ea0d-a72f-5e0f-9faf-cf49e6bdfa6a",
        "1c06ca82-15a9-57f7-a022-152aec8d56eb",
        "625418d7-8628-5405-9b15-65a996001fa1",
        "0c1590eb-b25c-59a0-bb7a-ef483498a5b6",
    ],
}


if __name__ == "__main__":
    train_embs, train_pids = load_embeddings()
    pid_to_title = load_titles()
    tokenizer, model = load_specter2()
    G = load_graph()

    for query_name, seed_texts in SEED_SETS.items():
        print(f"\n{'='*70}")
        print(f"{query_name}")
        print(f"{'='*70}")

        seed_embs = embed_seeds(seed_texts, tokenizer, model)
        semantic_results = retrieve(seed_embs, aggregate_mean, train_embs, train_pids, k=TOP_K)

        seed_ids = SEED_IDS[query_name]
        graph_results = ppr_retrieve(G, seed_ids, top_k=TOP_K)

        fused_results = reciprocal_rank_fusion(semantic_results, graph_results)[:TOP_K]

        sem_set = set(pid for pid, _ in semantic_results)
        graph_set = set(pid for pid, _ in graph_results)
        fused_set = set(pid for pid, _ in fused_results)

        print(f"\n-- Semantic only (Mean) --")
        print_results(semantic_results, pid_to_title)

        print(f"\n-- Graph only (PPR) --")
        print_results(graph_results, pid_to_title)

        print(f"\n-- Fused (RRF) --")
        print_results(fused_results, pid_to_title)

        print(f"\n-- Overlap --")
        print(f"  Semantic vs Graph Jaccard: {jaccard(sem_set, graph_set):.3f}")
        print(f"  Semantic vs Fused Jaccard: {jaccard(sem_set, fused_set):.3f}")
        print(f"  Graph vs Fused Jaccard:    {jaccard(graph_set, fused_set):.3f}")

        graph_only = graph_set - sem_set
        print(f"\n-- Papers PPR finds that semantic misses ({len(graph_only)}) --")
        for pid in list(graph_only)[:5]:
            title = pid_to_title.get(pid, "[unknown]")
            print(f"    - {title[:70]}")

        save_json(query_name, semantic_results, graph_results, fused_results, pid_to_title)

    print("\nDone!")