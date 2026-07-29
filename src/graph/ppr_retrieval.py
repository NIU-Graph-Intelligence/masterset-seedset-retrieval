import os
import json
import pickle
import networkx as nx
import polars as pl
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path("output")

GRAPH_PATH = OUTPUT_DIR / "citation_graph.pkl"
TRAIN_PARQUET = DATA_DIR / "train_v1.2.parquet"
TOP_K = 100
ALPHA = 0.15  # teleport probability


def load_graph():
    print("Loading citation graph...")
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def load_titles():
    df = pl.read_parquet(TRAIN_PARQUET, columns=["paper_id", "title"])
    return {row["paper_id"]: row["title"] for row in df.iter_rows(named=True)}


def ppr_retrieve(G, seed_ids, alpha=ALPHA, top_k=TOP_K):
    # check which seeds are in the graph
    valid_seeds = [s for s in seed_ids if s in G]
    missing = [s for s in seed_ids if s not in G]

    if missing:
        print(f"  Warning: {len(missing)} seed(s) not in graph — skipping them")
    if not valid_seeds:
        print("  Error: no seeds found in graph")
        return []

    # build personalization vector
    personalization = {node: 0.0 for node in G.nodes()}
    for sid in valid_seeds:
        personalization[sid] = 1.0 / len(valid_seeds)

    # run ppr
    print(f"  Running PPR with alpha={alpha}, {len(valid_seeds)} seeds...")
    ppr_scores = nx.pagerank(G, alpha=alpha, personalization=personalization)

    # sort and exclude seeds
    seed_set = set(seed_ids)
    ranked = sorted(ppr_scores.items(), key=lambda x: -x[1])
    results = [(pid, score) for pid, score in ranked if pid not in seed_set]

    return results[:top_k]


if __name__ == "__main__":
    G = load_graph()
    pid_to_title = load_titles()

    # test different alpha values on query 3 (most seeds in graph)
    seed_ids = [
        "be37ea0d-a72f-5e0f-9faf-cf49e6bdfa6a",
        "1c06ca82-15a9-57f7-a022-152aec8d56eb",
        "625418d7-8628-5405-9b15-65a996001fa1",
        "0c1590eb-b25c-59a0-bb7a-ef483498a5b6",
    ]

    for alpha in [0.10, 0.15, 0.20]:
        print(f"\n{'='*60}")
        print(f"Query 3: Transformer Theory — alpha={alpha}")
        print(f"{'='*60}")
        results = ppr_retrieve(G, seed_ids, alpha=alpha, top_k=10)
        for rank, (pid, score) in enumerate(results, 1):
            title = pid_to_title.get(pid, "[unknown]")
            print(f"  {rank:2d}. [{score:.6f}] {title[:70]}")