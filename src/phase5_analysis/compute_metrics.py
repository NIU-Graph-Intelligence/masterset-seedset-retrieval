#!/usr/bin/env python3
"""Phase 5 -- Step 2: consolidate judge metrics into one comparison table.

Output: outputs/phase5_analysis/metrics.json

    python -m src.phase5_analysis.compute_metrics
"""
import argparse
import sys

from ..common import config
from ..common.io_utils import ensure_dir, read_json, write_json, write_manifest
from ..common.seeds import load_query_sets, select


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topic", action="append", dest="topics")
    args = parser.parse_args()

    query_sets = select(load_query_sets(), args.topics)
    ensure_dir(config.PHASE5_DIR)

    rows, per_query = [], []
    for qs in query_sets:
        path = config.PHASE4_DIR / f"{qs.slug}.json"
        if not path.exists():
            print(f"[skip] {qs.topic}: not judged yet")
            continue

        data = read_json(path)
        print(f"{'=' * 78}\n{qs.topic}\n{'=' * 78}")
        print(f"  {'method':<22} {'avg@10':>7} {'avg@50':>7} {'P@10':>6} {'P@50':>6} {'n':>4}")

        entry = {"topic": qs.topic, "slug": qs.slug, "methods": {}}
        for name, block in data["methods"].items():
            m = block["metrics"]
            entry["methods"][name] = m
            rows.append({
                "topic": qs.topic, "slug": qs.slug, "method": name,
                "avg_at_10": m["avg_relevance_at_10"],
                "avg_at_50": m["avg_relevance_at_50"],
                "p_at_10": m["precision_at_10"],
                "p_at_50": m["precision_at_50"],
                "n_papers": m["n_papers"],
                "distribution": m["distribution"],
            })
            print(f"  {name:<22} {m['avg_relevance_at_10']:>7.2f} "
                  f"{m['avg_relevance_at_50']:>7.2f} {m['precision_at_10']:>6.2f} "
                  f"{m['precision_at_50']:>6.2f} {m['n_papers']:>4}")

        # Winner per metric, so the paper's claims are computed not eyeballed.
        entry["best"] = {}
        for metric in ["avg_relevance_at_10", "avg_relevance_at_50",
                       "precision_at_10", "precision_at_50"]:
            best = max(entry["methods"].items(), key=lambda kv: kv[1][metric])
            entry["best"][metric] = {"method": best[0], "value": best[1][metric]}
        print(f"  best avg@10: {entry['best']['avg_relevance_at_10']['method']}"
              f"   best P@10: {entry['best']['precision_at_10']['method']}\n")
        per_query.append(entry)

    write_json(config.PHASE5_DIR / "metrics.json",
               {"queries": per_query, "flat_rows": rows})
    write_manifest(config.PHASE5_DIR / "manifest_metrics.json",
                   "phase5_metrics", {"n_queries": len(per_query),
                                      "n_rows": len(rows)})
    print(f"Done -> {config.PHASE5_DIR / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
