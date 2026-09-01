import json
import pickle
import networkx as nx
import polars as pl
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path("output")

TRAIN_PARQUET = DATA_DIR / "train_v1.2.parquet"
GRAPH_PATH = OUTPUT_DIR / "citation_graph.pkl"


def build_citation_graph():
    print("Loading train parquet...")
    df = pl.read_parquet(TRAIN_PARQUET)
    train_ids = set(df["paper_id"].to_list())
    print(f"  {len(train_ids)} papers loaded")

    print("Building citation graph...")
    G = nx.DiGraph()
    G.add_nodes_from(train_ids)

    edge_count = 0
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
                edge_count += 1

    print(f"\n-- Graph Stats --")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")

    # papers with no outgoing edges (never cites anyone in the dataset)
    no_out = sum(1 for n in G.nodes() if G.out_degree(n) == 0)
    # papers with no incoming edges (never cited by anyone in the dataset)
    no_in = sum(1 for n in G.nodes() if G.in_degree(n) == 0)
    avg_out = G.number_of_edges() / G.number_of_nodes()

    print(f"  Papers with zero outgoing edges: {no_out} ({100*no_out/len(train_ids):.1f}%)")
    print(f"  Papers with zero incoming edges: {no_in} ({100*no_in/len(train_ids):.1f}%)")
    print(f"  Average citations per paper: {avg_out:.2f}")

    # save graph
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(G, f)
    print(f"\n  Graph saved to {GRAPH_PATH}")

    return G


if __name__ == "__main__":
    build_citation_graph()