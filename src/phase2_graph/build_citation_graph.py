#!/usr/bin/env python3
"""Phase 2 -- Step 1: build the intra-pool citation graph.

A directed edge u -> v means "paper u cites paper v". Only papers inside the
candidate pool become nodes, and only references that resolve to a pool paper
become edges, so the graph and the retrieval pool describe the same universe.

Output: outputs/phase2_graph/citation_graph.pkl
        outputs/phase2_graph/graph_stats.json

    python -m src.phase2_graph.build_citation_graph
"""
import argparse
import json
import pickle
import sys

import networkx as nx
import polars as pl

from ..common import config
from ..common.io_utils import ensure_dir, write_json, write_manifest


def _iter_reference_ids(value, id_key: str):
    """Yield cited paper ids from one row's reference field.

    Handles the shapes the column has taken across dataset versions: a JSON
    string, a list of dicts, or a list of bare id strings.
    """
    if value is None:
        return
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return
    for ref in value:
        if isinstance(ref, dict):
            cited = ref.get(id_key)
        elif isinstance(ref, str):
            cited = ref
        else:
            cited = None
        if cited:
            yield cited


def build(parquet_path=None, id_key: str = None, verbose: bool = True):
    parquet_path = parquet_path or config.CORPUS_PARQUET
    id_key = id_key or config.REFERENCE_ID_KEY

    schema = pl.read_parquet_schema(parquet_path)
    if config.REFERENCES_COLUMN not in schema:
        raise ValueError(
            f"Column {config.REFERENCES_COLUMN!r} not in {parquet_path}.\n"
            f"Available: {sorted(schema)}\n"
            f"Set REFERENCES_COLUMN in .env."
        )

    if verbose:
        print(f"Reading {parquet_path} ...")
    df = pl.read_parquet(parquet_path,
                         columns=["paper_id", config.REFERENCES_COLUMN])
    pool_ids = set(df["paper_id"].to_list())
    if verbose:
        print(f"  {len(pool_ids):,} papers in pool")

    graph = nx.DiGraph()
    graph.add_nodes_from(pool_ids)

    resolved = unresolved = rows_without_refs = 0
    for row in df.iter_rows(named=True):
        source = row["paper_id"]
        refs = list(_iter_reference_ids(row[config.REFERENCES_COLUMN], id_key))
        if not refs:
            rows_without_refs += 1
            continue
        for cited in refs:
            if cited in pool_ids:
                if cited != source:          # drop self-citations
                    graph.add_edge(source, cited)
                    resolved += 1
            else:
                unresolved += 1

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    no_out = sum(1 for n in graph.nodes if graph.out_degree(n) == 0)
    no_in = sum(1 for n in graph.nodes if graph.in_degree(n) == 0)
    isolated = sum(1 for n in graph.nodes
                   if graph.out_degree(n) == 0 and graph.in_degree(n) == 0)

    stats = {
        "corpus_parquet": str(parquet_path),
        "nodes": n_nodes,
        "edges": n_edges,
        "avg_out_degree": round(n_edges / n_nodes, 3) if n_nodes else 0.0,
        "resolved_reference_links": resolved,
        "references_outside_pool": unresolved,
        "papers_with_no_reference_field": rows_without_refs,
        "papers_with_zero_outgoing": no_out,
        "papers_with_zero_outgoing_pct": round(100 * no_out / n_nodes, 2) if n_nodes else 0,
        # A paper nobody in the pool cites can never be surfaced by PPR --
        # this is the hard ceiling on graph-only recall.
        "papers_with_zero_incoming": no_in,
        "papers_with_zero_incoming_pct": round(100 * no_in / n_nodes, 2) if n_nodes else 0,
        "fully_isolated_papers": isolated,
        "fully_isolated_pct": round(100 * isolated / n_nodes, 2) if n_nodes else 0,
    }

    if verbose:
        print("\n-- Graph statistics --")
        for key, value in stats.items():
            if key != "corpus_parquet":
                print(f"  {key:<34} {value:,}" if isinstance(value, int)
                      else f"  {key:<34} {value}")

    return graph, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if the graph already exists.")
    args = parser.parse_args()

    print(config.describe())
    print()

    if config.GRAPH_PATH.exists() and not args.force:
        print(f"Graph already exists: {config.GRAPH_PATH}\nPass --force to rebuild.")
        return 0

    graph, stats = build()

    ensure_dir(config.PHASE2_DIR)
    with open(config.GRAPH_PATH, "wb") as fh:
        pickle.dump(graph, fh, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(config.GRAPH_STATS_PATH, stats)
    write_manifest(config.PHASE2_DIR / "manifest_build.json",
                   "phase2_build_citation_graph", stats)

    print(f"\nGraph  -> {config.GRAPH_PATH}")
    print(f"Stats  -> {config.GRAPH_STATS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
