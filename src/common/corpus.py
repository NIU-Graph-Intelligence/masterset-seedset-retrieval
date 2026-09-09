"""Access to the candidate-pool parquet.

Loaded lazily and cached, because several scripts in a phase may each want
paper metadata and the file is large.
"""
from functools import lru_cache
from pathlib import Path
from typing import Dict

import polars as pl

from . import config


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Corpus parquet not found: {path}\n"
            f"Set DATA_DIR / CORPUS_PARQUET in .env."
        )
    return path


@lru_cache(maxsize=1)
def load_metadata(parquet_path: str = None) -> Dict[str, dict]:
    """paper_id -> {"title", "abstract", "year", "venue"} for the whole pool."""
    path = _require(Path(parquet_path) if parquet_path else config.CORPUS_PARQUET)

    available = pl.read_parquet_schema(path).keys()
    columns = [c for c in ["paper_id", "title", "abstract", "year", "venue"]
               if c in available]
    if "paper_id" not in columns:
        raise ValueError(f"{path} has no 'paper_id' column. Found: {sorted(available)}")

    df = pl.read_parquet(path, columns=columns)
    meta = {}
    for row in df.iter_rows(named=True):
        meta[row["paper_id"]] = {
            "title": row.get("title") or "",
            "abstract": row.get("abstract") or "",
            "year": str(row.get("year") or ""),
            "venue": row.get("venue") or "",
        }
    return meta


@lru_cache(maxsize=1)
def load_paper_ids(parquet_path: str = None) -> frozenset:
    path = _require(Path(parquet_path) if parquet_path else config.CORPUS_PARQUET)
    df = pl.read_parquet(path, columns=["paper_id"])
    return frozenset(df["paper_id"].to_list())


def titles_for(paper_ids) -> Dict[str, str]:
    meta = load_metadata()
    return {pid: meta.get(pid, {}).get("title", "") for pid in paper_ids}
