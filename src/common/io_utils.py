"""Filesystem helpers shared by every phase."""
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(text: str, max_len: int = 60) -> str:
    """Turn a topic name into a stable, filesystem-safe identifier.

    'Mixed Set 1: RL + Formal Language' -> 'mixed_set_1_rl_formal_language'

    Used to name every per-query artifact, so a given topic maps to the same
    filename in every phase and the phases can find each other's output.
    """
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len]


def write_json(path: Path, obj: Any, indent: int = 2) -> Path:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, ensure_ascii=False)
    return path


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def append_jsonl(path: Path, record: dict) -> None:
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_manifest(path: Path, phase: str, params: dict) -> Path:
    """Record the parameters a phase ran with.

    Every phase writes one of these next to its results. When a number in the
    paper is questioned, the manifest says exactly how it was produced.
    """
    return write_json(path, {
        "phase": phase,
        "generated_at": utc_now(),
        "params": params,
    })


def ranked_list_to_records(ranked, pid_to_meta=None) -> list:
    """[(paper_id, score), ...] -> [{"rank", "paper_id", "title", "score"}, ...]"""
    records = []
    for i, (pid, score) in enumerate(ranked, start=1):
        meta = (pid_to_meta or {}).get(pid, {})
        records.append({
            "rank": i,
            "paper_id": pid,
            "title": meta.get("title", ""),
            "score": float(score),
        })
    return records


def records_to_ids(records) -> list:
    """Inverse helper: pull the ordered paper_id list out of saved records."""
    return [r["paper_id"] for r in records]
