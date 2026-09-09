#!/usr/bin/env python3
"""Phase 5 -- Step 1: overlap analysis.

Two tables the paper depends on:
  1. Jaccard between aggregation strategies -- how much the choice of strategy
     actually changes the result, and how that varies with topic cohesion.
  2. Jaccard between semantic / graph / fused -- whether the two signals are
     decorrelated enough for fusion to be worth anything.

Output: outputs/phase5_analysis/overlap.json

    python -m src.phase5_analysis.compute_overlap
"""
import argparse
import sys
from itertools import combinations

from ..common import config
from ..common.corpus import load_metadata
from ..common.io_utils import (ensure_dir, read_json, records_to_ids,
                               write_json, write_manifest)
from ..common.metrics import jaccard, overlap_count
from ..common.seeds import load_query_sets, select


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topic", action="append", dest="topics")
    parser.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    args = parser.parse_args()

    query_sets = select(load_query_sets(), args.topics)
    metadata = load_metadata()
    ensure_dir(config.PHASE5_DIR)

    report = []
    for qs in query_sets:
        p1_path = config.PHASE1_DIR / f"{qs.slug}.json"
        if not p1_path.exists():
            print(f"[skip] {qs.topic}: no phase 1 output")
            continue

        print(f"{'=' * 72}\n{qs.topic}  ({len(qs)} seeds)\n{'=' * 72}")

        strategies = {name: set(records_to_ids(recs)[:args.top_k])
                      for name, recs in read_json(p1_path)["strategies"].items()}

        strategy_pairs = {}
        print("  Strategy vs strategy (Jaccard, top "
              f"{args.top_k}):")
        for a, b in combinations(sorted(strategies), 2):
            score = round(jaccard(strategies[a], strategies[b]), 4)
            strategy_pairs[f"{a}|{b}"] = score
            print(f"      {a:<9} vs {b:<9} {score:.3f}")

        in_all = set.intersection(*strategies.values()) if strategies else set()
        print(f"  Papers found by ALL {len(strategies)} strategies: {len(in_all)}")

        unique_per_strategy = {}
        for name, ids in strategies.items():
            others = set().union(*[v for k, v in strategies.items() if k != name])
            unique_per_strategy[name] = len(ids - others)

        entry = {
            "topic": qs.topic, "slug": qs.slug, "n_seeds": len(qs),
            "top_k": args.top_k,
            "strategy_jaccard": strategy_pairs,
            "n_in_all_strategies": len(in_all),
            "in_all_strategies_titles": [
                metadata.get(pid, {}).get("title", "") for pid in list(in_all)[:10]],
            "unique_per_strategy": unique_per_strategy,
        }

        # signal-level overlap
        p2_path = config.PHASE2_DIR / f"{qs.slug}.json"
        p3_path = config.PHASE3_DIR / f"{qs.slug}.json"
        if p2_path.exists() and p3_path.exists():
            graph_ids = set(records_to_ids(read_json(p2_path)["results"])[:args.top_k])
            p3 = read_json(p3_path)
            signal = {}
            for name, fusion in p3["fusions"].items():
                sem = strategies.get(name, set())
                fused = set(records_to_ids(fusion["results"])[:args.top_k])
                signal[name] = {
                    "semantic_vs_graph": round(jaccard(sem, graph_ids), 4),
                    "semantic_vs_fused": round(jaccard(sem, fused), 4),
                    "graph_vs_fused": round(jaccard(graph_ids, fused), 4),
                    "graph_only_papers": len(graph_ids - sem),
                    "shared_semantic_graph": overlap_count(sem, graph_ids),
                }
            entry["signal_jaccard"] = signal
            entry["graph_signal_available"] = bool(graph_ids)
            entry["graph_status"] = p3.get("graph_status")

            print("  Signal overlap (per semantic strategy):")
            for name, s in signal.items():
                print(f"      {name:<9} sem/graph={s['semantic_vs_graph']:.3f}  "
                      f"graph-only papers={s['graph_only_papers']}")

        report.append(entry)
        print()

    write_json(config.PHASE5_DIR / "overlap.json",
               {"top_k": args.top_k, "queries": report})
    write_manifest(config.PHASE5_DIR / "manifest_overlap.json",
                   "phase5_overlap", {"top_k": args.top_k,
                                      "n_queries": len(report)})
    print(f"Done -> {config.PHASE5_DIR / 'overlap.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
