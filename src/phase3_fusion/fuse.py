"""Rank fusion primitives. Pure functions, no I/O."""
from typing import Dict, List, Sequence, Tuple


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    k: int = 60,
    weights: Sequence[float] = None,
    missing_rank: int = None,
) -> List[Tuple[str, float]]:
    """Combine ranked id lists by summing 1 / (k + rank).

    RRF uses only rank position, never the underlying score, which is what
    makes it safe to combine a cosine similarity with a PageRank mass -- two
    quantities on entirely incomparable scales.

    ``k`` damps the influence of the very top ranks; the standard value is 60.

    ``missing_rank`` is the rank assigned to a document absent from a list. It
    defaults to (longest list length + 1), so absence is treated as "just past
    the end of what this ranker returned" rather than an arbitrary constant. A
    fixed large constant makes the penalty depend on retrieval depth, which
    silently changes fusion behaviour when depth changes.
    """
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights and ranked_lists must be the same length")

    if missing_rank is None:
        missing_rank = max((len(lst) for lst in ranked_lists), default=0) + 1

    rank_maps = [{pid: i for i, pid in enumerate(lst, start=1)} for lst in ranked_lists]

    everything = set()
    for lst in ranked_lists:
        everything.update(lst)

    fused: Dict[str, float] = {}
    for pid in everything:
        fused[pid] = sum(
            w / (k + rank_map.get(pid, missing_rank))
            for rank_map, w in zip(rank_maps, weights)
        )

    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


def contribution_breakdown(
    fused: Sequence[Tuple[str, float]],
    ranked_lists: Sequence[Sequence[str]],
    names: Sequence[str],
) -> Dict[str, int]:
    """How many of the fused results each input list contributed.

    'unique_to_<name>' counts papers only that ranker found -- the direct
    measure of whether a signal is actually adding anything, or just
    duplicating a view the other ranker already had.
    """
    fused_ids = {pid for pid, _ in fused}
    sets = [set(lst) for lst in ranked_lists]

    breakdown = {}
    for name, s in zip(names, sets):
        breakdown[f"from_{name}"] = len(fused_ids & s)
    for i, (name, s) in enumerate(zip(names, sets)):
        others = set().union(*[t for j, t in enumerate(sets) if j != i]) if len(sets) > 1 else set()
        breakdown[f"unique_to_{name}"] = len((fused_ids & s) - others)
    if len(sets) > 1:
        breakdown["in_all_inputs"] = len(fused_ids & set.intersection(*sets))
    return breakdown
