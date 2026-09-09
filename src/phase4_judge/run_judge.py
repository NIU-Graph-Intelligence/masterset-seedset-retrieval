#!/usr/bin/env python3
"""Phase 4: score every retrieved list with the LLM judge.

MasterSet's must-cite labels are defined for single-paper queries, not for set
queries, so there is no ground truth here. The judge supplies a consistent,
reproducible relevance signal in its place.

Reads phases 1-3 from disk. Judgments are cached, so re-running after a crash
or adding one new method costs only the genuinely new work.

Output: outputs/phase4_judge/<slug>.json
        outputs/phase4_judge/judge_cache.jsonl

    python -m src.phase4_judge.run_judge
    python -m src.phase4_judge.run_judge --topic "LLM Memory" --top-k 20
    python -m src.phase4_judge.run_judge --methods semantic_mean graph fused_mean
"""
import argparse
import sys
import time

from ..common import config
from ..common.corpus import load_metadata
from ..common.io_utils import (ensure_dir, read_json, records_to_ids,
                               write_json, write_manifest)
from ..common.metrics import summarise
from ..common.seeds import load_query_sets, select
from .judge_client import JudgeCache, OllamaJudge, judge_many


def collect_methods(slug, top_k, requested=None):
    """Gather every ranked list to judge, as {method_name: [paper_id, ...]}.

    Method names are stable across queries so the analysis phase can build
    comparison tables without special-casing.
    """
    methods = {}

    p1 = config.PHASE1_DIR / f"{slug}.json"
    if p1.exists():
        for name, records in read_json(p1)["strategies"].items():
            methods[f"semantic_{name}"] = records_to_ids(records)[:top_k]

    p2 = config.PHASE2_DIR / f"{slug}.json"
    if p2.exists():
        ids = records_to_ids(read_json(p2)["results"])[:top_k]
        if ids:
            methods["graph"] = ids

    p3 = config.PHASE3_DIR / f"{slug}.json"
    if p3.exists():
        for name, entry in read_json(p3)["fusions"].items():
            methods[f"fused_{name}"] = records_to_ids(entry["results"])[:top_k]

    if requested:
        unknown = set(requested) - set(methods)
        if unknown:
            raise ValueError(f"Unknown method(s) {sorted(unknown)}. "
                             f"Available: {sorted(methods)}")
        methods = {k: v for k, v in methods.items() if k in requested}
    return methods


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topic", action="append", dest="topics")
    parser.add_argument("--methods", nargs="+",
                        help="Restrict to these method names.")
    parser.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report how many judgments would run, then exit.")
    args = parser.parse_args()

    print(config.describe())
    print()

    query_sets = select(load_query_sets(), args.topics)
    metadata = load_metadata()
    ensure_dir(config.PHASE4_DIR)

    cache = JudgeCache(enabled=not args.no_cache)
    print(f"Judge cache: {len(cache)} existing judgment(s) at {cache.path}")

    plan = {}
    for qs in query_sets:
        methods = collect_methods(qs.slug, args.top_k, args.methods)
        if not methods:
            print(f"  [skip] {qs.topic}: no retrieval output found")
            continue
        plan[qs.slug] = (qs, methods)

    # Deduplicated across methods -- this is the number that actually hits the
    # model, and it is far smaller than the sum of the list lengths.
    total_slots = sum(len(ids) for _, m in plan.values() for ids in m.values())
    unique_pairs = sum(len({pid for ids in m.values() for pid in ids})
                       for _, m in plan.values())
    print(f"\nQueries: {len(plan)}   ranked slots: {total_slots}   "
          f"unique (query, paper) judgments: {unique_pairs}")
    print(f"Caching saves {total_slots - unique_pairs} redundant model calls.\n")

    if args.dry_run:
        for slug, (qs, methods) in plan.items():
            print(f"  {qs.topic}")
            for name, ids in methods.items():
                print(f"      {name:<22} {len(ids)} papers")
        return 0

    judge = OllamaJudge(temperature=args.temperature, seed=args.seed)
    print(f"Model: {judge.model} @ {config.OLLAMA_HOST} "
          f"(temperature={args.temperature}, seed={args.seed})\n")

    for slug, (qs, methods) in plan.items():
        print(f"{'=' * 72}\n{qs.topic}\n{'=' * 72}")
        started = time.time()

        judged, metrics = {}, {}
        for name, ids in methods.items():
            print(f"  {name} ({len(ids)} papers)")
            records = judge_many(judge, cache, qs.topic, qs.paper_ids,
                                 qs.titles, ids, metadata, label=name)
            judged[name] = records
            metrics[name] = summarise(records)

        write_json(config.PHASE4_DIR / f"{slug}.json", {
            "topic": qs.topic,
            "slug": slug,
            "model": judge.model,
            "temperature": args.temperature,
            "seed_paper_ids": qs.paper_ids,
            "top_k": args.top_k,
            "methods": {name: {"papers": judged[name], "metrics": metrics[name]}
                        for name in judged},
        })

        print(f"\n  {'method':<22} {'avg@10':>7} {'avg@50':>7} {'P@10':>6}")
        for name, m in metrics.items():
            print(f"  {name:<22} {m['avg_relevance_at_10']:>7.2f} "
                  f"{m['avg_relevance_at_50']:>7.2f} {m['precision_at_10']:>6.2f}")
        print(f"  -> {slug}.json  ({time.time() - started:.1f}s)\n")

    write_manifest(config.PHASE4_DIR / "manifest.json", "phase4_judge", {
        "model": judge.model,
        "host": config.OLLAMA_HOST,
        "temperature": args.temperature,
        "seed": args.seed,
        "top_k": args.top_k,
        "cache_enabled": not args.no_cache,
        "topics": [qs.topic for qs, _ in plan.values()],
    })
    print(f"Done. Results in {config.PHASE4_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
