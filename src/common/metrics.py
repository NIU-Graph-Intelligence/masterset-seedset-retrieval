"""Metric definitions shared by the analysis phases.

Kept in one place so that a metric reported in the paper has exactly one
implementation behind it.
"""
from typing import Dict, Iterable, List, Sequence


def jaccard(a: Iterable, b: Iterable) -> float:
    """|A n B| / |A u B|. Two empty sets are defined as identical."""
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def overlap_count(a: Iterable, b: Iterable) -> int:
    return len(set(a) & set(b))


def average_relevance(scores: Sequence[int], k: int = None) -> float:
    """Mean judge score over the top k, ignoring unparseable judgments (0)."""
    window = list(scores[:k]) if k else list(scores)
    valid = [s for s in window if s and s > 0]
    return round(sum(valid) / len(valid), 3) if valid else 0.0


def precision_at_k(scores: Sequence[int], k: int, threshold: int = 4) -> float:
    """Fraction of the top k scored >= threshold.

    Denominator is the number of papers actually present (not k), so a short
    list is not silently penalised.
    """
    window = list(scores[:k])
    if not window:
        return 0.0
    hits = sum(1 for s in window if s and s >= threshold)
    return round(hits / len(window), 3)


def relevance_distribution(scores: Sequence[int]) -> Dict[str, int]:
    dist = {str(i): 0 for i in range(1, 6)}
    dist["unparsed"] = 0
    for s in scores:
        key = str(s)
        if key in dist:
            dist[key] += 1
        else:
            dist["unparsed"] += 1
    return dist


def exact_agreement(runs: Sequence[Sequence[int]]) -> float:
    """Fraction of items where every repeated judgment gave the same score."""
    if not runs or not runs[0]:
        return 0.0
    n_items = len(runs[0])
    agree = sum(1 for i in range(n_items) if len({r[i] for r in runs}) == 1)
    return round(agree / n_items, 3)


def summarise(judged: List[dict], ks: Sequence[int] = (10, 20, 50)) -> dict:
    """Standard metric block for one ranked, judged list."""
    scores = [r.get("relevance_score", 0) for r in judged]
    block = {"n_papers": len(scores)}
    for k in ks:
        block[f"avg_relevance_at_{k}"] = average_relevance(scores, k)
        block[f"precision_at_{k}"] = precision_at_k(scores, k)
    block["avg_relevance_all"] = average_relevance(scores)
    block["distribution"] = relevance_distribution(scores)
    return block
