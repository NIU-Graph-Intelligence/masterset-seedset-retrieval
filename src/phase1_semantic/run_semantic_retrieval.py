#!/usr/bin/env python3
"""Phase 1 -- Step 2: semantic retrieval for every query set.

For each query set, scores the whole candidate pool under each aggregation
strategy and writes the top-K ranked list. Seed papers are excluded from every
list.

Output: outputs/phase1_semantic/<slug>.json

    python -m src.phase1_semantic.run_semantic_retrieval
    python -m src.phase1_semantic.run_semantic_retrieval --topic "LLM Memory"
    python -m src.phase1_semantic.run_semantic_retrieval --strategy mean --top-k 100
"""
import argparse
import sys
import time

from ..common import config
from ..common.corpus import load_metadata
from ..common.embeddings import (load_corpus_embeddings, load_seed_sources,
                                 resolve_seed_vectors)
from ..common.io_utils import (ensure_dir, ranked_list_to_records,
                               write_json, write_manifest)
from ..common.seeds import load_query_sets, select
from .aggregate import DISPLAY_NAMES, STRATEGIES, rank


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topic", action="append", dest="topics",
                        help="Restrict to this topic (repeatable).")
    parser.add_argument("--strategy", action="append", dest="strategies",
                        choices=sorted(STRATEGIES), help="Restrict to this strategy (repeatable).")
    parser.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K,
                        help=f"Results per list (default {config.DEFAULT_TOP_K}).")
    parser.add_argument("--depth", type=int, default=config.DEFAULT_RETRIEVAL_DEPTH,
                        help="Partial-sort depth before seed removal.")
    parser.add_argument("--soft-and-alpha", type=float, default=0.5)
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--allow-encode-fallback", action="store_true",
                        help="Encode seeds missing from the pool instead of failing.")
    args = parser.parse_args()

    print(config.describe())
    print()

    query_sets = select(load_query_sets(), args.topics)
    strategy_names = args.strategies or list(STRATEGIES)

    seed_sources = load_seed_sources()
    corpus = seed_sources[0][1]          # the pool; what we actually rank
    metadata = load_metadata()
    ensure_dir(config.PHASE1_DIR)

    kwargs = {"soft_and": {"alpha": args.soft_and_alpha},
              "cluster": {"n_clusters": args.clusters}}

    print(f"\nQuery sets: {len(query_sets)}   strategies: {', '.join(strategy_names)}")
    print(f"top_k={args.top_k}  depth={args.depth}\n")

    for qs in query_sets:
        print(f"{'=' * 72}\n{qs.topic}  ({len(qs)} seeds)\n{'=' * 72}")
        started = time.time()

        try:
            seed_embs, resolved_ids, provenance = resolve_seed_vectors(
                seed_sources, qs.paper_ids, qs.texts,
                allow_encode_fallback=args.allow_encode_fallback)
        except KeyError as exc:
            print(f"  SKIPPED: {exc}\n")
            continue

        from collections import Counter
        by_source = Counter(provenance.values())
        if set(by_source) - {"pool"}:
            # A seed sourced from eval is not in the retrieval pool, and so is
            # also not a node in the citation graph -- PPR cannot use it.
            print(f"  seed vector sources: {dict(by_source)}")
        encoded_ids = [pid for pid, s in provenance.items() if s == "encoded"]

        # Excluded from results even if a seed was encoded rather than looked
        # up -- the paper is still a seed and must not be returned.
        exclude = set(qs.paper_ids)

        results, per_strategy_ids = {}, {}
        for name in strategy_names:
            scores = STRATEGIES[name](seed_embs, corpus.matrix, **kwargs.get(name, {}))
            ranked = rank(scores, corpus.paper_ids, exclude_ids=exclude,
                          top_k=args.top_k, depth=args.depth)
            results[name] = ranked_list_to_records(ranked, metadata)
            per_strategy_ids[name] = {pid for pid, _ in ranked}

            leaked = per_strategy_ids[name] & exclude
            assert not leaked, f"seed leaked into {name} results: {leaked}"

            print(f"  {DISPLAY_NAMES[name]:<10} top-3:")
            for rec in results[name][:3]:
                print(f"      {rec['rank']}. [{rec['score']:.4f}] {rec['title'][:64]}")

        write_json(config.PHASE1_DIR / f"{qs.slug}.json", {
            "topic": qs.topic,
            "slug": qs.slug,
            "n_seeds": len(qs),
            "seed_paper_ids": qs.paper_ids,
            "seeds_encoded_from_text": encoded_ids,
            "seed_vector_sources": provenance,
            "top_k": args.top_k,
            "strategies": results,
        })
        print(f"  -> {qs.slug}.json  ({time.time() - started:.1f}s)\n")

    write_manifest(config.PHASE1_DIR / "manifest.json", "phase1_semantic", {
        "corpus_parquet": str(config.CORPUS_PARQUET),
        "embeddings_path": str(config.EMBEDDINGS_PATH),
        "seeds_json": str(config.SEEDS_JSON),
        "pool_size": len(corpus),
        "embedding_dim": corpus.dim,
        "strategies": strategy_names,
        "top_k": args.top_k,
        "depth": args.depth,
        "soft_and_alpha": args.soft_and_alpha,
        "clusters": args.clusters,
        "seeds_excluded_from_results": True,
        "seed_vectors": "looked up from embedding sources",
        "embedding_sources": [str(config.EMBEDDINGS_PATH)]
                             + ([str(config.SEED_EMBEDDINGS_PATH)]
                                if len(seed_sources) > 1 else []),
        "topics": [qs.topic for qs in query_sets],
    })
    print(f"Done. Results in {config.PHASE1_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
