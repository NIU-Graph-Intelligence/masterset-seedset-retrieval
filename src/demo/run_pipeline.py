import os
import json
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv

import sys
sys.path.append(str(Path(__file__).parent.parent / "set_retrieval"))
sys.path.append(str(Path(__file__).parent.parent / "graph"))
sys.path.append(str(Path(__file__).parent.parent / "evaluation"))

from set_query import (
    load_embeddings, load_titles, load_specter2, embed_seeds,
    aggregate_mean, aggregate_soft_and, aggregate_max, aggregate_cluster, retrieve
)
from ppr_retrieval import load_graph, ppr_retrieve
from rank_fusion import reciprocal_rank_fusion
from llm_judge import load_abstracts, judge_paper

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path("output")


def load_seeds(seed_file):
    with open(seed_file) as f:
        data = json.load(f)
    topic = data.get("topic", "Unknown Topic")
    papers = data.get("papers", [])
    seed_texts = [(p["title"], p.get("abstract", "")) for p in papers]
    seed_titles = [p["title"] for p in papers]
    print(f"Loaded {len(seed_texts)} seed papers for topic: '{topic}'")
    return topic, seed_texts, seed_titles


def get_strategy(name):
    strategies = {
        "mean": aggregate_mean,
        "max": aggregate_max,
        "soft-and": aggregate_soft_and,
        "cluster": aggregate_cluster,
    }
    return strategies.get(name, aggregate_mean)


def print_table(results, pid_to_title, top_n=20, show_scores=True):
    print(f"\n{'='*80}")
    print(f"{'Rank':<5} {'Title':<55} {'Ret.':<8} {'Rel.':<5}")
    print(f"{'='*80}")
    for r in results[:top_n]:
        rank = r["rank"]
        title = pid_to_title.get(r["paper_id"], r.get("title", "[unknown]"))
        title_short = title[:54] + "..." if len(title) > 54 else title
        ret_score = f"{r.get('retrieval_score', 0):.4f}"
        rel_score = str(r.get("relevance_score", "-")) if show_scores else "-"
        print(f"{rank:<5} {title_short:<55} {ret_score:<8} {rel_score:<5}")
    print(f"{'='*80}")


def run_demo(seed_file, strategy="mean", use_graph=True, use_judge=True, top_k=20, rerank=False):
    print("\n" + "="*60)
    print("MasterSet Citation Retrieval Pipeline")
    print("="*60)

    # load everything
    train_embs, train_pids = load_embeddings()
    pid_to_title = load_titles()
    tokenizer, specter_model = load_specter2()
    paper_map = load_abstracts() if use_judge else {}

    G = None
    if use_graph:
        G = load_graph()

    # load seeds
    topic, seed_texts, seed_titles = load_seeds(seed_file)

    # semantic retrieval
    print(f"\nRunning semantic retrieval (strategy={strategy})...")
    seed_embs = embed_seeds(seed_texts, tokenizer, specter_model)
    strat_fn = get_strategy(strategy)
    semantic_results = retrieve(seed_embs, strat_fn, train_embs, train_pids, k=max(top_k * 3, 100))
    print(f"  Retrieved {len(semantic_results)} candidates")

    # graph retrieval
    graph_results = []
    if use_graph and G is not None:
        print(f"\nRunning graph retrieval (PPR)...")
        seed_ids_in_graph = []
        for title, _ in seed_texts:
            for pid in train_pids:
                t = pid_to_title.get(pid, "")
                if t and title.lower()[:30] in t.lower():
                    seed_ids_in_graph.append(pid)
                    break
        if not seed_ids_in_graph:
            print(f"  No seeds found in graph — falling back to semantic only")
            use_graph = False
        elif len(seed_ids_in_graph) < len(seed_texts):
            print(f"  Found {len(seed_ids_in_graph)}/{len(seed_texts)} seeds in graph")
            graph_results = ppr_retrieve(G, seed_ids_in_graph, top_k=max(top_k * 3, 100))
            print(f"  Retrieved {len(graph_results)} graph candidates")
        else:
            graph_results = ppr_retrieve(G, seed_ids_in_graph, top_k=max(top_k * 3, 100))
            print(f"  Found {len(seed_ids_in_graph)} seeds in graph, retrieved {len(graph_results)} candidates")

    # fusion
    if use_graph and graph_results:
        print(f"\nFusing semantic and graph results with RRF...")
        fused = reciprocal_rank_fusion(semantic_results, graph_results)[:top_k]
        final_results = [{"rank": i+1, "paper_id": pid, "retrieval_score": score}
                        for i, (pid, score) in enumerate(fused)]
        print(f"  Fused {len(final_results)} results")
    else:
        final_results = [{"rank": i+1, "paper_id": pid, "retrieval_score": score}
                        for i, (pid, score) in enumerate(semantic_results[:top_k])]

    # llm judge
    if use_judge:
        print(f"\nScoring results with LLM judge...")
        for i, r in enumerate(final_results):
            pid = r["paper_id"]
            info = paper_map.get(pid, {})
            title = info.get("title", pid_to_title.get(pid, "[unknown]"))
            abstract = info.get("abstract", "")
            score, justification = judge_paper(topic, seed_titles, title, abstract)
            r["relevance_score"] = score
            r["justification"] = justification
            r["title"] = title
            print(f"  [{i+1:2d}/{len(final_results)}] Score={score} | {title[:50]}")
            time.sleep(13)

        # rerank by relevance score if requested
        if rerank:
            print("\nReranking by relevance score...")
            final_results = sorted(final_results, key=lambda x: x.get("relevance_score", 0), reverse=True)
            for i, r in enumerate(final_results):
                r["rank"] = i + 1

    # print results
    print_table(final_results, pid_to_title, top_n=top_k, show_scores=use_judge)

    # save
    output_path = OUTPUT_DIR / f"demo_results_{Path(seed_file).stem}.json"
    output = {
        "topic": topic,
        "strategy": strategy,
        "use_graph": use_graph,
        "use_judge": use_judge,
        "rerank": rerank,
        "top_k": top_k,
        "results": final_results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MasterSet Citation Retrieval Pipeline")
    parser.add_argument("--seeds", required=True, help="Path to seed papers JSON file")
    parser.add_argument("--strategy", default="mean",
                        choices=["mean", "max", "soft-and", "cluster"],
                        help="Aggregation strategy for semantic retrieval")
    parser.add_argument("--no-graph", action="store_true", help="Skip graph retrieval")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM relevance scoring")
    parser.add_argument("--top-k", type=int, default=20, help="Number of results to return")
    parser.add_argument("--rerank", action="store_true", help="Rerank results by LLM relevance score")
    args = parser.parse_args()

    run_demo(
        seed_file=args.seeds,
        strategy=args.strategy,
        use_graph=not args.no_graph,
        use_judge=not args.no_judge,
        top_k=args.top_k,
        rerank=args.rerank
    )