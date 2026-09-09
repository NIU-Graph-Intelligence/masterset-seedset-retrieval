#!/usr/bin/env python3
"""Phase 4 -- companion: measure judge self-consistency.

Judges the same papers several times and reports exact agreement. The number
belongs in the paper: it is what justifies treating a single judge pass as a
reliable measurement.

The judge cache is bypassed here by design -- a cached judgment would return
the identical answer and report perfect agreement by construction.

Output: outputs/phase4_judge/consistency.json

    python -m src.phase4_judge.run_consistency --topic "Neural Algorithmic Reasoning"
"""
import argparse
import sys
from collections import Counter

from ..common import config
from ..common.corpus import load_metadata
from ..common.io_utils import ensure_dir, read_json, write_json, write_manifest
from ..common.metrics import exact_agreement
from ..common.seeds import load_query_sets, select
from .judge_client import OllamaJudge


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topic", action="append", dest="topics",
                        help="Default: the first query set.")
    parser.add_argument("--method", default="semantic_mean",
                        help="Which judged list to sample from.")
    parser.add_argument("--n-papers", type=int, default=20)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    print(config.describe())
    print()

    query_sets = select(load_query_sets(), args.topics)
    if not args.topics:
        query_sets = query_sets[:1]

    metadata = load_metadata()
    ensure_dir(config.PHASE4_DIR)
    judge = OllamaJudge(temperature=args.temperature)

    report = []
    for qs in query_sets:
        judged_path = config.PHASE4_DIR / f"{qs.slug}.json"
        if not judged_path.exists():
            print(f"[skip] {qs.topic}: run run_judge.py first")
            continue

        data = read_json(judged_path)
        if args.method not in data["methods"]:
            print(f"[skip] {qs.topic}: method {args.method!r} not judged. "
                  f"Available: {sorted(data['methods'])}")
            continue

        papers = data["methods"][args.method]["papers"][:args.n_papers]
        print(f"{'=' * 72}\n{qs.topic} -- {len(papers)} papers x {args.n_runs} runs\n{'=' * 72}")

        per_paper, runs = [], [[] for _ in range(args.n_runs)]
        for i, paper in enumerate(papers, start=1):
            pid = paper["paper_id"]
            meta = metadata.get(pid, {})
            title = meta.get("title", paper.get("title", ""))

            scores = []
            for r in range(args.n_runs):
                judgment = judge.judge(qs.topic, qs.titles, title,
                                       meta.get("abstract", ""))
                scores.append(judgment.score)
                runs[r].append(judgment.score)

            consistent = len(set(scores)) == 1
            per_paper.append({"paper_id": pid, "title": title,
                              "scores": scores, "consistent": consistent})
            flag = " " if consistent else "*"
            print(f"  {flag}[{i:2d}/{len(papers)}] {scores}  {title[:52]}")

        agreement = exact_agreement(runs)
        spread = Counter(max(p["scores"]) - min(p["scores"]) for p in per_paper)
        n_inconsistent = sum(1 for p in per_paper if not p["consistent"])

        print(f"\n  Exact agreement: {agreement:.1%} "
              f"({len(per_paper) - n_inconsistent}/{len(per_paper)})")
        print(f"  Score spread distribution: {dict(sorted(spread.items()))}")
        if n_inconsistent:
            print("  Papers that varied:")
            for p in per_paper:
                if not p["consistent"]:
                    print(f"      {p['scores']}  {p['title'][:58]}")

        report.append({
            "topic": qs.topic, "slug": qs.slug, "method": args.method,
            "n_papers": len(per_paper), "n_runs": args.n_runs,
            "exact_agreement": agreement,
            "n_inconsistent": n_inconsistent,
            "max_spread_distribution": {str(k): v for k, v in sorted(spread.items())},
            "papers": per_paper,
        })

    write_json(config.PHASE4_DIR / "consistency.json", {
        "model": judge.model, "temperature": args.temperature, "reports": report,
    })
    write_manifest(config.PHASE4_DIR / "manifest_consistency.json",
                   "phase4_judge_consistency", {
                       "model": judge.model, "temperature": args.temperature,
                       "n_papers": args.n_papers, "n_runs": args.n_runs,
                       "method": args.method, "cache_bypassed": True,
                   })
    print(f"\nDone -> {config.PHASE4_DIR / 'consistency.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
