#!/usr/bin/env python3
"""Phase 3: fuse the semantic and graph rankings with RRF.

Reads the saved output of phases 1 and 2 -- it does not import them or re-run
retrieval. Phases communicate through files, so this can be re-run with
different fusion parameters in seconds without touching a GPU.

Output: outputs/phase3_fusion/<slug>.json

    python -m src.phase3_fusion.run_fusion
    python -m src.phase3_fusion.run_fusion --rrf-k 20 60 100
    python -m src.phase3_fusion.run_fusion --strategy mean --strategy soft_and
"""
import argparse
import sys

from ..common import config
from ..common.corpus import load_metadata
from ..common.io_utils import (ensure_dir, ranked_list_to_records, read_json,
                               records_to_ids, write_json, write_manifest)
from ..common.metrics import jaccard
from ..common.seeds import load_query_sets, select
from .fuse import contribution_breakdown, reciprocal_rank_fusion


def _load_phase_output(directory, slug, phase_label, hint):
    path = directory / f"{slug}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{phase_label} output missing for {slug!r}: {path}\nRun: {hint}"
        )
    return read_json(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topic", action="append", dest="topics")
    parser.add_argument("--strategy", action="append", dest="strategies",
                        help="Semantic strategies to fuse with the graph "
                             "(default: every strategy found in phase 1).")
    parser.add_argument("--rrf-k", type=float, nargs="+",
                        default=[config.DEFAULT_RRF_K],
                        help="RRF smoothing constant(s). First is primary.")
    parser.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    parser.add_argument("--graph-weight", type=float, default=1.0)
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    args = parser.parse_args()

    print(config.describe())
    print()

    query_sets = select(load_query_sets(), args.topics)
    metadata = load_metadata()
    ensure_dir(config.PHASE3_DIR)

    primary_k = args.rrf_k[0]
    sweep_k = args.rrf_k[1:]
    weights = [args.semantic_weight, args.graph_weight]

    print(f"Primary RRF k={primary_k}   weights sem/graph="
          f"{args.semantic_weight}/{args.graph_weight}")
    if sweep_k:
        print(f"k sweep: {sweep_k}")
    print()

    for qs in query_sets:
        print(f"{'=' * 72}\n{qs.topic}\n{'=' * 72}")

        semantic = _load_phase_output(
            config.PHASE1_DIR, qs.slug, "Phase 1",
            "python -m src.phase1_semantic.run_semantic_retrieval")
        graph = _load_phase_output(
            config.PHASE2_DIR, qs.slug, "Phase 2",
            "python -m src.phase2_graph.run_ppr_retrieval")

        graph_ids = records_to_ids(graph["results"])
        graph_status = graph["diagnostics"]["status"]

        if not graph_ids:
            print(f"  no graph signal ({graph_status}) -- fusion would return "
                  f"the semantic list unchanged; recording that explicitly.")

        strategy_names = args.strategies or list(semantic["strategies"])
        exclude = set(qs.paper_ids)
        fused_all = {}

        for name in strategy_names:
            if name not in semantic["strategies"]:
                print(f"  [skip] strategy {name!r} not in phase 1 output")
                continue

            semantic_ids = records_to_ids(semantic["strategies"][name])

            fused = reciprocal_rank_fusion([semantic_ids, graph_ids],
                                           k=primary_k, weights=weights)
            fused = [(pid, s) for pid, s in fused if pid not in exclude][:args.top_k]
            fused_ids = [pid for pid, _ in fused]

            entry = {
                "results": ranked_list_to_records(fused, metadata),
                "overlap": {
                    "semantic_vs_graph_jaccard": round(jaccard(semantic_ids, graph_ids), 4),
                    "semantic_vs_fused_jaccard": round(jaccard(semantic_ids, fused_ids), 4),
                    "graph_vs_fused_jaccard": round(jaccard(graph_ids, fused_ids), 4),
                },
                "contribution": contribution_breakdown(
                    fused, [semantic_ids, graph_ids], ["semantic", "graph"]),
                "identical_to_semantic": fused_ids == semantic_ids[:len(fused_ids)],
            }

            if sweep_k:
                entry["rrf_k_sweep"] = {}
                for k_val in sweep_k:
                    alt = reciprocal_rank_fusion([semantic_ids, graph_ids],
                                                 k=k_val, weights=weights)
                    alt = [(p, s) for p, s in alt if p not in exclude][:args.top_k]
                    alt_ids = [p for p, _ in alt]
                    entry["rrf_k_sweep"][str(k_val)] = {
                        "jaccard_vs_primary": round(jaccard(fused_ids, alt_ids), 4),
                        "results": ranked_list_to_records(alt, metadata),
                    }

            fused_all[name] = entry

            c = entry["contribution"]
            print(f"  {name:<9} sem/graph J={entry['overlap']['semantic_vs_graph_jaccard']:.3f}"
                  f"  fused from graph-only: {c.get('unique_to_graph', 0)}"
                  f"  from semantic-only: {c.get('unique_to_semantic', 0)}")

        write_json(config.PHASE3_DIR / f"{qs.slug}.json", {
            "topic": qs.topic,
            "slug": qs.slug,
            "seed_paper_ids": qs.paper_ids,
            "rrf_k": primary_k,
            "weights": {"semantic": args.semantic_weight, "graph": args.graph_weight},
            "top_k": args.top_k,
            "graph_status": graph_status,
            "graph_signal_available": bool(graph_ids),
            "fusions": fused_all,
        })
        print(f"  -> {qs.slug}.json\n")

    write_manifest(config.PHASE3_DIR / "manifest.json", "phase3_fusion", {
        "primary_rrf_k": primary_k,
        "rrf_k_sweep": sweep_k,
        "semantic_weight": args.semantic_weight,
        "graph_weight": args.graph_weight,
        "top_k": args.top_k,
        "missing_rank_policy": "longest input list length + 1",
        "seeds_excluded_from_results": True,
        "topics": [qs.topic for qs in query_sets],
    })
    print(f"Done. Results in {config.PHASE3_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
