#!/usr/bin/env python3
"""Phase 1 -- Step 1: verify every seed before any retrieval runs.

Run this FIRST, every time seeds.json changes.

A seed that is absent from the candidate pool cannot be looked up in the
embedding matrix. A seed that is absent from the citation graph contributes
nothing to Personalized PageRank -- and if every seed of a query is absent,
PPR returns nothing at all and the "fused" result silently collapses to the
semantic result. That is exactly how the original mixed query set produced a
zero-signal graph run without anything visibly failing.

This script surfaces those conditions up front instead.

    python -m src.phase1_semantic.validate_seeds
"""
import argparse
import sys

from ..common import config
from ..common.corpus import load_metadata, load_paper_ids
from ..common.io_utils import write_json
from ..common.seeds import load_query_sets, select


def _load_graph_nodes():
    """Graph node set, if phase 2 has already been run."""
    if not config.GRAPH_PATH.exists():
        return None
    import pickle
    with open(config.GRAPH_PATH, "rb") as fh:
        graph = pickle.load(fh)
    return graph


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topic", action="append", dest="topics",
                        help="Validate only this topic (repeatable).")
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero if any seed is missing from the pool.")
    args = parser.parse_args()

    print(config.describe())
    print()

    query_sets = select(load_query_sets(), args.topics)
    pool_ids = load_paper_ids()
    metadata = load_metadata()

    try:
        from ..common.embeddings import load_seed_sources
        seed_sources = load_seed_sources(verbose=False)
        embed_index = {}
        for name, corpus in seed_sources:
            print(f"Embeddings [{name}]: {len(corpus):,} papers, dim={corpus.dim}")
            for pid in corpus.paper_ids:
                embed_index.setdefault(pid, name)
        embedded_ids = set(embed_index)
    except FileNotFoundError as exc:
        embed_index, embedded_ids = None, None
        print(f"[skip] embeddings not available ({exc})")

    graph = _load_graph_nodes()
    if graph is None:
        print("[skip] citation graph not built yet "
              "(run src/phase2_graph/build_citation_graph.py to enable graph checks)")
    else:
        print(f"Citation graph: {graph.number_of_nodes():,} nodes, "
              f"{graph.number_of_edges():,} edges")

    print(f"Candidate pool:   {len(pool_ids):,} papers")
    print(f"Query sets:       {len(query_sets)}")
    print()

    report, any_problem, any_warning = [], False, False

    for qs in query_sets:
        rows, n_pool, n_emb, n_graph = [], 0, 0, 0

        for paper in qs.papers:
            pid = paper.paper_id
            in_pool = pid in pool_ids
            in_emb = None if embedded_ids is None else (pid in embedded_ids)
            emb_source = None if embed_index is None else embed_index.get(pid)
            in_graph = None if graph is None else (pid in graph)

            n_pool += int(in_pool)
            n_emb += int(bool(in_emb))
            n_graph += int(bool(in_graph))

            row = {
                "paper_id": pid,
                "title": paper.title,
                "year": paper.year,
                "venue": paper.venue,
                "in_pool": in_pool,
                "in_embeddings": in_emb,
                "embedding_source": emb_source,
                "in_graph": in_graph,
            }
            if in_graph:
                row["in_degree"] = graph.in_degree(pid)
                row["out_degree"] = graph.out_degree(pid)

            # Title drift between seeds.json and the corpus usually means the
            # paper_id was copied from the wrong record.
            if in_pool:
                corpus_title = metadata.get(pid, {}).get("title", "")
                if corpus_title and paper.title:
                    a = corpus_title.strip().lower()
                    b = paper.title.strip().lower()
                    if a != b and a[:40] != b[:40]:
                        row["title_mismatch"] = corpus_title
            rows.append(row)

        graph_pct = (100 * n_graph / len(qs)) if graph is not None else None
        status = "OK"
        if n_pool < len(qs):
            status, any_problem = "MISSING_FROM_POOL", True
        elif graph is not None and n_graph == 0:
            status, any_problem = "NO_GRAPH_SIGNAL", True
        elif graph is not None and n_graph < len(qs):
            status, any_warning = "PARTIAL_GRAPH", True

        flag = {"OK": "  ", "PARTIAL_GRAPH": "~ ",
                "NO_GRAPH_SIGNAL": "! ", "MISSING_FROM_POOL": "! "}[status]
        graph_str = "n/a" if graph_pct is None else f"{n_graph}/{len(qs)} ({graph_pct:.0f}%)"
        print(f"{flag}{qs.topic}")
        print(f"    seeds={len(qs):<3} in_pool={n_pool}/{len(qs)}  "
              f"in_graph={graph_str}  -> {status}")

        for row in rows:
            problems = []
            if not row["in_pool"]:
                problems.append("not in pool")
            if row.get("in_embeddings") is False:
                problems.append("not in ANY embedding file")
            elif row.get("embedding_source") not in (None, "pool"):
                problems.append(
                    f"vector from {row['embedding_source']!r}, not the pool "
                    f"(so not a graph node)")
            if row.get("in_graph") is False:
                problems.append("not in graph")
            if "title_mismatch" in row:
                problems.append(f"title differs from corpus: {row['title_mismatch'][:60]!r}")
            if problems:
                print(f"      - {row['title'][:58]:<58} {'; '.join(problems)}")

        report.append({
            "topic": qs.topic,
            "slug": qs.slug,
            "n_seeds": len(qs),
            "n_in_pool": n_pool,
            "n_in_embeddings": n_emb if embedded_ids is not None else None,
            "n_in_graph": n_graph if graph is not None else None,
            "status": status,
            "seeds": rows,
        })

    write_json(config.SEED_VALIDATION_PATH, {
        "corpus_parquet": str(config.CORPUS_PARQUET),
        "pool_size": len(pool_ids),
        "graph_available": graph is not None,
        "embeddings_available": embedded_ids is not None,
        "query_sets": report,
    })

    print(f"\nReport written to {config.SEED_VALIDATION_PATH}")

    if any_problem:
        print("\nBLOCKING problems found. A query set has seeds missing from the "
              "pool, or no seeds in the graph at all.\nFix seeds.json, or accept "
              "the limitation explicitly before running retrieval.")
        if args.strict:
            return 1
    elif any_warning:
        print("\nAll seeds are in the pool, but at least one query set has seeds "
              "missing from the citation graph.\nPPR will run on the remaining "
              "seeds only; note the reduced graph coverage when reporting results.")
    else:
        print("\nAll seeds resolve cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
