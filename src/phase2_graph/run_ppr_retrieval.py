#!/usr/bin/env python3
"""Phase 2 -- Step 2: Personalized PageRank retrieval.

A random walk over the citation graph that restarts from the seed papers,
ranking pool papers by how much of the walk's stationary mass lands on them.

    python -m src.phase2_graph.run_ppr_retrieval
    python -m src.phase2_graph.run_ppr_retrieval --damping 0.85 0.90 0.95

IMPORTANT -- damping vs. teleport probability
---------------------------------------------
networkx names its argument ``alpha``, and it is the DAMPING factor: the
probability of following an outgoing edge at each step. The teleport
probability, i.e. the chance of jumping back to a seed, is ``1 - damping``.

The paper reports a teleport probability of 0.15, which means

    damping = 0.85       teleport = 0.15        <- correct
    nx.pagerank(G, alpha=0.85, personalization=...)

Passing 0.15 as networkx's alpha inverts this: the walk teleports back to the
seeds 85% of the time and almost never traverses an edge, so the ranking
barely leaves the seeds' immediate in-neighbourhood and is nearly invariant to
the parameter. A damping sweep that shows "no effect" is the symptom.

This script uses damping and prints both numbers so the two cannot be confused.
"""
import argparse
import pickle
import sys
import time

import networkx as nx

from ..common import config
from ..common.corpus import load_metadata
from ..common.io_utils import (ensure_dir, ranked_list_to_records,
                               write_json, write_manifest)
from ..common.seeds import load_query_sets, select


def load_graph(path=None, verbose: bool = True):
    path = path or config.GRAPH_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Citation graph not found: {path}\n"
            f"Run: python -m src.phase2_graph.build_citation_graph"
        )
    if verbose:
        print(f"Loading citation graph from {path} ...")
    with open(path, "rb") as fh:
        graph = pickle.load(fh)
    if verbose:
        print(f"  {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")
    return graph


def ppr_rank(graph, seed_ids, damping: float = config.DEFAULT_DAMPING,
             top_k: int = config.DEFAULT_TOP_K, tol: float = 1e-8,
             max_iter: int = 200):
    """Run PPR and return (ranked, diagnostics). Seeds excluded from results."""
    present = [s for s in seed_ids if s in graph]
    absent = [s for s in seed_ids if s not in graph]

    diagnostics = {
        "n_seeds": len(seed_ids),
        "n_seeds_in_graph": len(present),
        "seeds_not_in_graph": absent,
        "damping": damping,
        "teleport_probability": round(1.0 - damping, 4),
    }

    if not present:
        # Not an exception: a legitimate finding about the query set, and the
        # reason downstream fusion must be told the graph signal is empty.
        diagnostics["status"] = "NO_SEEDS_IN_GRAPH"
        return [], diagnostics

    weight = 1.0 / len(present)
    personalization = {node: 0.0 for node in graph.nodes}
    for sid in present:
        personalization[sid] = weight

    scores = nx.pagerank(graph, alpha=damping, personalization=personalization,
                         tol=tol, max_iter=max_iter, dangling=personalization)

    exclude = set(seed_ids)
    ranked = [(pid, score) for pid, score
              in sorted(scores.items(), key=lambda kv: -kv[1])
              if pid not in exclude][:top_k]

    diagnostics["status"] = "OK" if not absent else "PARTIAL_SEEDS_IN_GRAPH"
    diagnostics["n_nonzero_scores"] = sum(1 for v in scores.values() if v > 1e-12)
    return ranked, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topic", action="append", dest="topics")
    parser.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    parser.add_argument("--damping", type=float, nargs="+",
                        default=[config.DEFAULT_DAMPING],
                        help="One or more damping factors. The first is the "
                             "primary run; the rest form a sensitivity sweep.")
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--max-iter", type=int, default=200)
    args = parser.parse_args()

    print(config.describe())
    print()

    query_sets = select(load_query_sets(), args.topics)
    graph = load_graph()
    metadata = load_metadata()
    ensure_dir(config.PHASE2_DIR)

    primary = args.damping[0]
    sweep = args.damping[1:]
    print(f"\nPrimary damping={primary} (teleport={1 - primary:.2f})")
    if sweep:
        print(f"Sensitivity sweep: {sweep}")
    print()

    for qs in query_sets:
        print(f"{'=' * 72}\n{qs.topic}  ({len(qs)} seeds)\n{'=' * 72}")
        started = time.time()

        ranked, diag = ppr_rank(graph, qs.paper_ids, damping=primary,
                                top_k=args.top_k, tol=args.tol,
                                max_iter=args.max_iter)

        print(f"  seeds in graph: {diag['n_seeds_in_graph']}/{diag['n_seeds']}"
              f"   status: {diag['status']}")
        if diag["seeds_not_in_graph"]:
            for sid in diag["seeds_not_in_graph"]:
                title = metadata.get(sid, {}).get("title", "[not in pool]")
                print(f"      absent: {title[:60]}")

        if ranked:
            for rec in ranked_list_to_records(ranked, metadata)[:3]:
                print(f"      {rec['rank']}. [{rec['score']:.6f}] {rec['title'][:62]}")
        else:
            print("      no graph signal for this query")

        payload = {
            "topic": qs.topic,
            "slug": qs.slug,
            "n_seeds": len(qs),
            "seed_paper_ids": qs.paper_ids,
            "top_k": args.top_k,
            "diagnostics": diag,
            "results": ranked_list_to_records(ranked, metadata),
        }

        if sweep and ranked:
            payload["damping_sweep"] = {}
            for d in sweep:
                sweep_ranked, sweep_diag = ppr_rank(
                    graph, qs.paper_ids, damping=d, top_k=args.top_k,
                    tol=args.tol, max_iter=args.max_iter)
                payload["damping_sweep"][str(d)] = {
                    "diagnostics": sweep_diag,
                    "results": ranked_list_to_records(sweep_ranked, metadata),
                }

        write_json(config.PHASE2_DIR / f"{qs.slug}.json", payload)
        print(f"  -> {qs.slug}.json  ({time.time() - started:.1f}s)\n")

    write_manifest(config.PHASE2_DIR / "manifest_ppr.json", "phase2_ppr_retrieval", {
        "graph_path": str(config.GRAPH_PATH),
        "primary_damping": primary,
        "primary_teleport_probability": round(1 - primary, 4),
        "damping_sweep": sweep,
        "top_k": args.top_k,
        "tol": args.tol,
        "max_iter": args.max_iter,
        "seeds_excluded_from_results": True,
        "topics": [qs.topic for qs in query_sets],
    })
    print(f"Done. Results in {config.PHASE2_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
