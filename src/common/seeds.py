"""Loading and validating the seed query sets.

The single source of truth for queries is data/seeds.json. No script in this
repository may define seed papers inline -- that was the defect that let three
copies of the same query drift apart, each producing different embeddings.

Expected schema:

    {
      "SEED_SETS": [
        {
          "topic": "Neural Algorithmic Reasoning",
          "papers": [
            {"paper_id": "...", "title": "...", "abstract": "...",
             "year": "2025", "venue": "neurips"},
            ...
          ]
        },
        ...
      ]
    }
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from . import config
from .io_utils import read_json, slugify


@dataclass(frozen=True)
class SeedPaper:
    paper_id: str
    title: str
    abstract: str = ""
    year: str = ""
    venue: str = ""


@dataclass(frozen=True)
class QuerySet:
    topic: str
    papers: List[SeedPaper] = field(default_factory=list)

    @property
    def slug(self) -> str:
        return slugify(self.topic)

    @property
    def paper_ids(self) -> List[str]:
        return [p.paper_id for p in self.papers]

    @property
    def titles(self) -> List[str]:
        return [p.title for p in self.papers]

    @property
    def texts(self):
        """(title, abstract) pairs -- only needed by the encode-from-text fallback."""
        return [(p.title, p.abstract) for p in self.papers]

    def __len__(self) -> int:
        return len(self.papers)


def load_query_sets(path: Path = None) -> List[QuerySet]:
    path = path or config.SEEDS_JSON
    if not path.exists():
        raise FileNotFoundError(
            f"Seeds file not found: {path}\n"
            f"Set SEEDS_JSON in .env, or place seeds.json in {config.DATA_DIR}."
        )

    raw = read_json(path)
    if isinstance(raw, dict):
        sets_raw = raw.get("SEED_SETS", raw.get("seed_sets"))
    else:
        sets_raw = raw
    if sets_raw is None:
        raise ValueError(f"{path} has no 'SEED_SETS' key.")

    query_sets = []
    for entry in sets_raw:
        topic = entry.get("topic")
        if not topic:
            raise ValueError(f"A query set in {path} is missing 'topic'.")
        papers = []
        for p in entry.get("papers", []):
            pid = p.get("paper_id")
            if not pid:
                raise ValueError(
                    f"Seed in '{topic}' is missing paper_id: {p.get('title', '?')!r}"
                )
            papers.append(SeedPaper(
                paper_id=pid,
                title=p.get("title", ""),
                abstract=p.get("abstract") or "",
                year=str(p.get("year", "")),
                venue=p.get("venue", ""),
            ))
        if not papers:
            raise ValueError(f"Query set '{topic}' has no papers.")
        query_sets.append(QuerySet(topic=topic, papers=papers))

    _assert_unique_slugs(query_sets)
    return query_sets


def _assert_unique_slugs(query_sets) -> None:
    """Two topics that slugify identically would overwrite each other's files."""
    seen = {}
    for qs in query_sets:
        if qs.slug in seen:
            raise ValueError(
                f"Topics {seen[qs.slug]!r} and {qs.topic!r} both slugify to "
                f"{qs.slug!r}; rename one of them in seeds.json."
            )
        seen[qs.slug] = qs.topic


def select(query_sets, topics=None):
    """Filter by topic name or slug. Used by every --topic CLI flag."""
    if not topics:
        return query_sets
    wanted = {slugify(t) for t in topics}
    chosen = [qs for qs in query_sets if qs.slug in wanted]
    missing = wanted - {qs.slug for qs in chosen}
    if missing:
        available = ", ".join(sorted(qs.slug for qs in query_sets))
        raise ValueError(f"Unknown topic(s): {sorted(missing)}.\nAvailable: {available}")
    return chosen
