#!/usr/bin/env python3
"""Discover what is actually on disk, and print the .env lines that match.

Run this once on a new machine, before anything else. It answers the questions
the rest of the pipeline depends on:

  - which parquet files exist, and how many papers each holds
  - which one has a usable references column (and what the cited-id key is)
  - where the SPECTER2 embeddings are, and whether they cover the pool
  - whether seeds.json resolves

    python scripts/inspect_data.py
    python scripts/inspect_data.py --search-root /home/ratul/mustcite
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl  # noqa: E402

from src.common import config  # noqa: E402

CANDIDATE_REF_KEYS = ["matched_paper_id", "paper_id", "cited_paper_id",
                      "corpus_id", "id", "reference_id"]


def human(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def schema_of(path: Path):
    try:
        return dict(pl.read_parquet_schema(path))
    except Exception:
        return dict(pl.scan_parquet(path).collect_schema())


def probe_references(path: Path, column: str):
    """Look at real rows to work out how the reference column is shaped."""
    try:
        sample = pl.read_parquet(path, columns=["paper_id", column], n_rows=200)
    except Exception as exc:
        return {"error": str(exc)}

    for value in sample[column].to_list():
        if value is None:
            continue
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return {"shape": "string (not JSON)"}
        if isinstance(value, (list, tuple)) and value:
            first = value[0]
            if isinstance(first, dict):
                keys = sorted(first.keys())
                suggested = next((k for k in CANDIDATE_REF_KEYS if k in keys), None)
                return {"shape": "list[dict]", "keys": keys,
                        "suggested_id_key": suggested}
            if isinstance(first, str):
                return {"shape": "list[str]", "suggested_id_key": None}
    return {"shape": "empty or all-null in first 200 rows"}


def find_files(roots, patterns, limit=40):
    hits = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for pattern in patterns:
            for path in sorted(root.rglob(pattern)):
                if path.is_file():
                    hits.append(path)
                    if len(hits) >= limit:
                        return hits
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--search-root", action="append", dest="roots",
                        help="Extra directory to scan for parquet/.pt files.")
    args = parser.parse_args()

    print(config.describe())
    print()

    roots = [config.DATA_DIR, config.REPO_ROOT]
    if config.EMBEDDINGS_DIR.exists():
        roots.append(config.EMBEDDINGS_DIR)
    roots += [Path(r) for r in (args.roots or [])]

    # ---------------------------------------------------------------- parquet
    print("=" * 78)
    print("PARQUET FILES")
    print("=" * 78)
    parquets = find_files(roots, ["*.parquet"])
    if not parquets:
        print(f"  none found under: {', '.join(str(r) for r in roots)}")
    parquet_rows = {}
    for path in parquets:
        schema = schema_of(path)
        try:
            n_rows = pl.scan_parquet(path).select(pl.len()).collect().item()
        except Exception:
            n_rows = "?"
        has_id = "paper_id" in schema
        ref_cols = [c for c in schema
                    if "ref" in c.lower() or "citation" in c.lower()]

        print(f"\n  {path}")
        print(f"      {human(path.stat().st_size)}   rows={n_rows:,}"
              if isinstance(n_rows, int) else
              f"      {human(path.stat().st_size)}   rows={n_rows}")
        print(f"      columns ({len(schema)}): {', '.join(list(schema)[:12])}"
              + (" ..." if len(schema) > 12 else ""))
        if not has_id:
            print("      [!] no 'paper_id' column -- cannot be the corpus pool")
            continue
        if ref_cols:
            for col in ref_cols:
                info = probe_references(path, col)
                print(f"      references column {col!r}: {info}")
                if info.get("suggested_id_key") and isinstance(n_rows, int):
                    parquet_rows[path] = (n_rows, col, info["suggested_id_key"])
        else:
            print("      [!] no reference-like column -- graph cannot be built "
                  "from this file")

    # ------------------------------------------------------------- embeddings
    print("\n" + "=" * 78)
    print("EMBEDDING FILES")
    print("=" * 78)
    pts = find_files(roots, ["*.pt", "*.npz", "*embeddings*"], limit=25)
    pts = [p for p in pts if p.suffix in {".pt", ".npz"}]
    if not pts:
        print("  none found. Point EMBEDDINGS_DIR at the SPECTER2 output "
              "directory,\n  or set --search-root to scan somewhere else.")
    emb_counts = {}
    for path in pts:
        n = None
        try:
            import torch
            blob = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(blob, dict) and "paper_ids" in blob:
                n = len(blob["paper_ids"])
                emb_counts[path] = n
                dim = blob.get("embedding_dim", "?")
                print(f"  {path}")
                print(f"      {human(path.stat().st_size)}   vectors={n:,}   dim={dim}")
                continue
        except Exception as exc:
            print(f"  {path}   ({human(path.stat().st_size)})   [unreadable: {exc}]")
            continue
        print(f"  {path}   ({human(path.stat().st_size)})   [no 'paper_ids' key]")

    # The corpus parquet must describe the same papers as the pool embeddings.
    # Matching on row count is what catches an all_papers/candidate_pool mixup.
    best = None
    if emb_counts and parquet_rows:
        print("\n  Matching parquet row counts against embedding vector counts:")
        for emb_path, n_vec in emb_counts.items():
            hits = [p for p, (n, _, _) in parquet_rows.items() if n == n_vec]
            if hits:
                print(f"      {emb_path.name} ({n_vec:,}) == {hits[0].name}")
                if best is None or "candidate" in hits[0].name or "pool" in hits[0].name:
                    n, col, key = parquet_rows[hits[0]]
                    best = (hits[0], n, col, key, emb_path)
            else:
                print(f"      {emb_path.name} ({n_vec:,}) -- no parquet with "
                      f"this row count")
        if best is None:
            print("      [!] no parquet matches any embedding file. Retrieval "
                  "would silently drop papers.")

    # ------------------------------------------------------------------ seeds
    print("\n" + "=" * 78)
    print("SEEDS")
    print("=" * 78)
    if config.SEEDS_JSON.exists():
        try:
            from src.common.seeds import load_query_sets
            query_sets = load_query_sets()
            total = sum(len(q) for q in query_sets)
            print(f"  {config.SEEDS_JSON}")
            print(f"  {len(query_sets)} query sets, {total} seed papers")
            for q in query_sets:
                print(f"      {len(q):>2} seeds  {q.topic}")
        except Exception as exc:
            print(f"  [!] {config.SEEDS_JSON} failed to parse: {exc}")
    else:
        print(f"  [!] not found: {config.SEEDS_JSON}")

    # ------------------------------------------------------------ suggestions
    print("\n" + "=" * 78)
    print("SUGGESTED .env")
    print("=" * 78)
    print(f"DATA_DIR={config.DATA_DIR}")
    print(f"OUTPUT_DIR={config.OUTPUT_DIR}")
    if best:
        path, n_rows, ref_col, ref_key, emb_path = best
        rel = path.relative_to(config.DATA_DIR) if config.DATA_DIR in path.parents else path
        print(f"CORPUS_PARQUET={rel}   # {n_rows:,} papers, matches {emb_path.name}")
        print(f"REFERENCES_COLUMN={ref_col}")
        print(f"REFERENCE_ID_KEY={ref_key}")
        print(f"EMBEDDINGS_DIR={emb_path.parent}")
        print(f"EMBEDDINGS_FILE={emb_path.name}")
        others = [p.name for p in emb_counts if p != emb_path]
        if others:
            print(f"SEED_EMBEDDINGS_FILE={others[0]}"
                  f"   # seeds outside the pool are looked up here")
    else:
        print("CORPUS_PARQUET=<no parquet matched an embedding file>")
        print("EMBEDDINGS_DIR=<path to the SPECTER2 embeddings directory>")
        print("EMBEDDINGS_FILE=candidates_embeddings.pt")
    print("SEEDS_JSON=data/seeds.json")
    print("OLLAMA_MODEL=qwen2.5:7b-instruct")
    print("\nPaste the lines you agree with into .env, then re-run this script "
          "to confirm nothing is [MISSING].")
    return 0


if __name__ == "__main__":
    sys.exit(main())
