#!/usr/bin/env python3
"""Search the candidate pool for papers to use as seeds.

Only returns papers that are actually in the retrieval pool, so anything this
prints is guaranteed to resolve in every phase. Use it to pick replacements for
seeds that failed validation.

    # find candidates by keyword (all terms must appear in the title)
    python scripts/find_papers.py "in-context learning"
    python scripts/find_papers.py in-context demonstrations --venue acl emnlp

    # search abstracts too, and require a well-connected paper
    python scripts/find_papers.py "in-context" --in-abstract --min-in-degree 20

    # emit ready-to-paste seeds.json entries
    python scripts/find_papers.py "in-context learning" --json --limit 5
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl  # noqa: E402

from src.common import config  # noqa: E402


def load_graph():
    if not config.GRAPH_PATH.exists():
        return None
    with open(config.GRAPH_PATH, "rb") as fh:
        return pickle.load(fh)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("terms", nargs="+",
                        help="All terms must appear (case-insensitive).")
    parser.add_argument("--in-abstract", action="store_true",
                        help="Match against title + abstract instead of title only.")
    parser.add_argument("--venue", nargs="+", help="Restrict to these venues.")
    parser.add_argument("--year-min", type=int)
    parser.add_argument("--year-max", type=int)
    parser.add_argument("--min-in-degree", type=int, default=0,
                        help="Require at least this many in-pool citations. "
                             "A well-cited seed gives PPR more to work with.")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--json", action="store_true",
                        help="Print seeds.json entries instead of a table.")
    parser.add_argument("--sort", choices=["in_degree", "year", "title"],
                        default="in_degree")
    args = parser.parse_args()

    df = pl.read_parquet(config.CORPUS_PARQUET,
                         columns=["paper_id", "title", "abstract", "year", "venue"])

    haystack = pl.col("title")
    if args.in_abstract:
        haystack = pl.col("title") + " " + pl.col("abstract").fill_null("")

    expr = pl.lit(True)
    for term in args.terms:
        expr = expr & haystack.str.to_lowercase().str.contains(term.lower(), literal=True)
    if args.venue:
        expr = expr & pl.col("venue").is_in([v.lower() for v in args.venue])
    if args.year_min is not None:
        expr = expr & (pl.col("year").cast(pl.Int32, strict=False) >= args.year_min)
    if args.year_max is not None:
        expr = expr & (pl.col("year").cast(pl.Int32, strict=False) <= args.year_max)

    hits = df.filter(expr)
    if hits.height == 0:
        print(f"No papers in the pool match: {' AND '.join(args.terms)}")
        print("Try fewer terms, or --in-abstract.")
        return 1

    graph = load_graph()
    rows = []
    for row in hits.iter_rows(named=True):
        pid = row["paper_id"]
        in_deg = graph.in_degree(pid) if graph and pid in graph else 0
        out_deg = graph.out_degree(pid) if graph and pid in graph else 0
        if in_deg < args.min_in_degree:
            continue
        rows.append({**row, "in_degree": in_deg, "out_degree": out_deg})

    if not rows:
        print(f"{hits.height} title match(es), but none with in-degree >= "
              f"{args.min_in_degree}. Lower --min-in-degree.")
        return 1

    key = {"in_degree": lambda r: -r["in_degree"],
           "year": lambda r: str(r["year"]),
           "title": lambda r: r["title"]}[args.sort]
    rows.sort(key=key)
    rows = rows[:args.limit]

    if args.json:
        entries = [{"paper_id": r["paper_id"], "title": r["title"],
                    "abstract": r["abstract"] or "", "year": str(r["year"] or ""),
                    "venue": r["venue"] or ""} for r in rows]
        print(json.dumps(entries, indent=2, ensure_ascii=False))
        return 0

    print(f"{len(rows)} of {hits.height} match(es) in the pool "
          f"({config.CORPUS_PARQUET.name})\n")
    print(f"{'in':>5} {'out':>5} {'year':>5} {'venue':<8} title")
    print("-" * 100)
    for r in rows:
        print(f"{r['in_degree']:>5} {r['out_degree']:>5} {str(r['year'] or ''):>5} "
              f"{(r['venue'] or ''):<8} {r['title'][:70]}")
        print(f"{'':>5} {'':>5} {'':>5} {'':<8} {r['paper_id']}")
    print("\nRe-run with --json to get seeds.json entries for these papers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
