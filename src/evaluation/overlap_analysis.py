import os
import json
import torch
import numpy as np
import polars as pl
from pathlib import Path
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

# reuse functions from set_query.py
import sys
sys.path.append(str(Path(__file__).parent.parent / "set_retrieval"))
from set_query import (
    load_embeddings, load_titles, load_specter2, embed_seeds,
    aggregate_mean, aggregate_max, aggregate_soft_and, aggregate_cluster,
    retrieve
)

load_dotenv()

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
    "Mixed Set: RL + Formal Language + Efficient Transformers": [
        ("Expected Policy Gradients",
         "We derive expected policy gradients, which extend the REINFORCE estimator by marginalizing over actions, reducing variance while maintaining unbiasedness in reinforcement learning settings."),
        ("Distributed Distributional Deterministic Policy Gradients",
         "We present a distributed distributional approach to continuous control with deep reinforcement learning, combining distributional value estimation with deterministic policy gradients."),
        ("Representing Formal Languages: A Comparison Between Finite Automata and Recurrent Neural Networks",
         "We compare finite automata and recurrent neural networks for representing formal languages, analyzing their expressive power and learning efficiency on regular and context-free languages."),
        ("On the Ability and Limitations of Transformers to Recognize Formal Languages",
         "We study which formal languages transformers can recognize, showing they can express any language in AC0 but struggle with counting and certain regular languages without positional encodings."),
        ("Reformer: The Efficient Transformer",
         "We introduce the Reformer, an efficient transformer using locality-sensitive hashing attention and reversible residual layers to reduce memory and computational costs for long sequences."),
    ],
}


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def save_results_json(query_name, results, result_sets, output_dir, pid_to_title):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = query_name.lower().replace(" ", "_").replace(":", "").replace("/", "") + ".json"

    data = {
        "query": query_name,
        "top_50_results": {
            name: [{"rank": i+1, "paper_id": pid, "title": pid_to_title.get(pid, "[unknown]"), "score": score}
                   for i, (pid, score) in enumerate(res)]
            for name, res in results.items()
        },
        "overlap_analysis": {
            "papers_in_all_strategies": [
                {"paper_id": pid, "title": pid_to_title.get(pid, "[unknown]")}
                for pid in set.intersection(*result_sets.values())
            ],
            "unique_per_strategy": {
                name: [
                    {"paper_id": pid, "title": pid_to_title.get(pid, "[unknown]")}
                    for pid in s - set().union(*[v for k, v in result_sets.items() if k != name])
                ]
                for name, s in result_sets.items()
            }
        }
    }

    filepath = output_dir / filename
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Results saved to {filepath}")


def overlap_analysis(query_name, seed_texts, train_embs, train_pids,
                     pid_to_title, tokenizer, model, top_k=50):
    print(f"\n{'='*70}")
    print(f"{query_name}")
    print(f"{'='*70}")

    seed_embs = embed_seeds(seed_texts, tokenizer, model)

    strategies = {
        "Mean":     aggregate_mean,
        "Max":      aggregate_max,
        "Soft-AND": aggregate_soft_and,
        "Cluster":  aggregate_cluster,
    }

    results = {}
    for name, fn in strategies.items():
        results[name] = retrieve(seed_embs, fn, train_embs, train_pids, k=top_k)

    result_sets = {name: set(pid for pid, _ in res) for name, res in results.items()}

    # jaccard similarity between each pair
    print("\n-- Jaccard Similarity (top 50 overlap) --")
    strat_names = list(strategies.keys())
    for i in range(len(strat_names)):
        for j in range(i + 1, len(strat_names)):
            a, b = strat_names[i], strat_names[j]
            j_score = jaccard(result_sets[a], result_sets[b])
            print(f"  {a} vs {b}: {j_score:.3f}")

    # papers in all 4
    all_four = set.intersection(*result_sets.values())
    print(f"\n-- Papers in ALL 4 strategies ({len(all_four)}) --")
    for pid in list(all_four)[:10]:
        title = pid_to_title.get(pid, "[unknown]")
        print(f"  {title[:80]}")

    # unique to each strategy
    print("\n-- Unique papers per strategy --")
    for name, s in result_sets.items():
        others = set().union(*[v for k, v in result_sets.items() if k != name])
        unique = s - others
        print(f"\n  {name} only ({len(unique)} papers):")
        for pid in list(unique)[:5]:
            title = pid_to_title.get(pid, "[unknown]")
            print(f"    - {title[:75]}")

    save_results_json(query_name, results, result_sets, "output/set_retrieval_results", pid_to_title)


if __name__ == "__main__":
    train_embs, train_pids = load_embeddings()
    pid_to_title = load_titles()
    tokenizer, model = load_specter2()

    for query_name, seed_texts in SEED_SETS.items():
        overlap_analysis(query_name, seed_texts, train_embs, train_pids,
                         pid_to_title, tokenizer, model)

    print("\nDone!")