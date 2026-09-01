import os
import json
import time
import polars as pl
from pathlib import Path
from dotenv import load_dotenv

import sys
sys.path.append(str(Path(__file__).parent.parent / "set_retrieval"))
sys.path.append(str(Path(__file__).parent.parent / "graph"))

from set_query import (
    load_embeddings, load_titles, load_specter2, embed_seeds,
    aggregate_mean, aggregate_soft_and, aggregate_max, retrieve
)
from ppr_retrieval import load_graph, ppr_retrieve
from rank_fusion import reciprocal_rank_fusion
from llm_judge import (
    load_abstracts, judge_paper,
    average_relevance, precision_at_k, relevance_distribution
)

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path("output")
RESULTS_DIR = OUTPUT_DIR / "judge_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TOP_K = 50

QUERIES = {
    "Query 1: Neural Algorithmic Reasoning": {
        "seed_texts": [
            ("Tropical Attention: Neural Algorithmic Reasoning for Combinatorial Algorithms",
             "We introduce Tropical Attention, an attention mechanism grounded in tropical geometry."),
            ("Primal-Dual Neural Algorithmic Reasoning",
             "We introduce a general NAR framework grounded in the primal-dual paradigm."),
            ("Discrete Neural Algorithmic Reasoning",
             "We propose to force neural reasoners to maintain the execution trajectory as finite predefined states."),
            ("Open-Book Neural Algorithmic Reasoning",
             "We propose a novel open-book learning framework where the network can access all training instances."),
            ("Deep Equilibrium Algorithmic Reasoning",
             "We study neurally solving algorithms by finding the solution directly by solving an equilibrium equation."),
        ],
        "seed_ids": [
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
        "topic_name": "Neural Algorithmic Reasoning",
    },
    "Query 2: LLM Memory": {
        "seed_texts": [
            ("Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs",
             "RAG consistently outperforms fine-tuning for both existing and new knowledge."),
            ("Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?",
             "LLMs struggle to acquire new factual knowledge through fine-tuning."),
            ("Deciphering the Interplay of Parametric and Non-parametric Memory",
             "The model relies more on retrieved context than its parametric knowledge."),
            ("Transformer Memory as a Differentiable Search Index",
             "We introduce DSI, mapping string queries directly to relevant docids."),
        ],
        "seed_ids": [
            "a8527971-b28a-5210-85b1-19f74e267a2a",
            "2d32d8d5-9dfc-50d5-b580-ed0f8f1e5a86",
            "5dac5135-0eae-5666-a7de-33b5e8fbaf3c",
            "af017fa8-28d5-5bfb-b237-696b2f85cde9",
            "2004913a-d0b3-592e-8e33-ded71e6c16af",
        ],
        "topic_name": "LLM Memory and Knowledge Injection",
    },
    "Query 3: Transformer Theory": {
        "seed_texts": [
            ("Unique Hard Attention: A Tale of Two Sides",
             "Leftmost-hard attention corresponds to a strictly weaker fragment of Linear Temporal Logic."),
            ("Logical Languages Accepted by Transformer Encoders with Hard Attention",
             "UHAT encoders recognize all languages definable in first-order logic."),
            ("Representational Strengths and Limitations of Transformers",
             "We establish positive and negative results on the representation power of attention layers."),
            ("Tighter Bounds on the Expressivity of Transformer Encoders",
             "A variant of first-order logic with counting quantifiers bounds transformer encoders."),
        ],
        "seed_ids": [
            "be37ea0d-a72f-5e0f-9faf-cf49e6bdfa6a",
            "1c06ca82-15a9-57f7-a022-152aec8d56eb",
            "625418d7-8628-5405-9b15-65a996001fa1",
            "0c1590eb-b25c-59a0-bb7a-ef483498a5b6",
        ],
        "topic_name": "Theoretical Expressivity of Transformers",
    },
}


def judge_all(method_name, paper_ids, topic_name, seed_titles, paper_map, delay=13):
    print(f"\n  Judging {method_name} ({len(paper_ids)} papers)...")
    results = []
    for i, pid in enumerate(paper_ids):
        info = paper_map.get(pid, {})
        title = info.get("title", "[unknown]")
        abstract = info.get("abstract", "")
        score, justification = judge_paper(topic_name, seed_titles, title, abstract)
        results.append({
            "rank": i + 1,
            "paper_id": pid,
            "title": title,
            "relevance_score": score,
            "justification": justification,
        })
        print(f"    [{i+1:2d}/{len(paper_ids)}] Score={score} | {title[:50]}")
        time.sleep(delay)
    return results


if __name__ == "__main__":
    train_embs, train_pids = load_embeddings()
    pid_to_title = load_titles()
    tokenizer, model = load_specter2()
    G = load_graph()
    paper_map = load_abstracts()

    all_results = {}

    for query_name, config in QUERIES.items():
        print(f"\n{'='*70}")
        print(f"{query_name}")
        print(f"{'='*70}")

        topic_name = config["topic_name"]
        seed_texts = config["seed_texts"]
        seed_ids = config["seed_ids"]
        seed_titles = [t for t, _ in seed_texts]

        # get embeddings and ranked lists
        seed_embs = embed_seeds(seed_texts, tokenizer, model)
        semantic_mean = retrieve(seed_embs, aggregate_mean, train_embs, train_pids, k=TOP_K)
        semantic_softand = retrieve(seed_embs, aggregate_soft_and, train_embs, train_pids, k=TOP_K)
        semantic_max = retrieve(seed_embs, aggregate_max, train_embs, train_pids, k=TOP_K)
        graph = ppr_retrieve(G, seed_ids, top_k=TOP_K)

        fused_mean = reciprocal_rank_fusion(semantic_mean, graph)[:TOP_K]
        fused_softand = reciprocal_rank_fusion(semantic_softand, graph)[:TOP_K]
        fused_max = reciprocal_rank_fusion(semantic_max, graph)[:TOP_K]

        sem_ids = [pid for pid, _ in semantic_mean]
        graph_ids = [pid for pid, _ in graph]
        fused_mean_ids = [pid for pid, _ in fused_mean]
        fused_softand_ids = [pid for pid, _ in fused_softand]
        fused_max_ids = [pid for pid, _ in fused_max]

        # judge all methods
        sem_judged = judge_all("Semantic (Mean)", sem_ids, topic_name, seed_titles, paper_map)
        graph_judged = judge_all("Graph (PPR)", graph_ids, topic_name, seed_titles, paper_map)
        fused_mean_judged = judge_all("Fused Mean+PPR", fused_mean_ids, topic_name, seed_titles, paper_map)
        fused_softand_judged = judge_all("Fused Soft-AND+PPR", fused_softand_ids, topic_name, seed_titles, paper_map)
        fused_max_judged = judge_all("Fused Max+PPR", fused_max_ids, topic_name, seed_titles, paper_map)

        def metrics(judged):
            return {
                "avg_relevance_top10": average_relevance(judged, 10),
                "avg_relevance_top50": average_relevance(judged, 50),
                "precision_at_10": precision_at_k(judged, 10),
                "precision_at_20": precision_at_k(judged, 20),
                "distribution": relevance_distribution(judged),
            }

        results = {
            "query": query_name,
            "topic": topic_name,
            "semantic_mean": {"papers": sem_judged, "metrics": metrics(sem_judged)},
            "graph": {"papers": graph_judged, "metrics": metrics(graph_judged)},
            "fused_mean": {"papers": fused_mean_judged, "metrics": metrics(fused_mean_judged)},
            "fused_softand": {"papers": fused_softand_judged, "metrics": metrics(fused_softand_judged)},
            "fused_max": {"papers": fused_max_judged, "metrics": metrics(fused_max_judged)},
        }
        all_results[query_name] = results

        # print summary
        print(f"\n-- Summary --")
        for method, key in [
            ("Semantic Mean", "semantic_mean"),
            ("Graph PPR", "graph"),
            ("Fused Mean+PPR", "fused_mean"),
            ("Fused Soft-AND+PPR", "fused_softand"),
            ("Fused Max+PPR", "fused_max"),
        ]:
            m = results[key]["metrics"]
            print(f"  {method}: avg@10={m['avg_relevance_top10']} P@10={m['precision_at_10']}")

        # save
        filename = query_name.lower().replace(" ", "_").replace(":", "").replace("/", "") + "_judged_v2.json"
        with open(RESULTS_DIR / filename, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to {RESULTS_DIR / filename}")

    print("\nDone!")