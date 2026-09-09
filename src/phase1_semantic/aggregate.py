"""Set-to-score aggregation strategies.

Pure functions: (seed_matrix, corpus_matrix) -> score per corpus paper.
No I/O, no configuration, no side effects, so they can be unit-tested and
reasoned about independently of the pipeline.

All inputs are assumed L2-normalised, so a dot product is cosine similarity.
"""
from typing import Callable, Dict

import numpy as np


def _similarity_matrix(seed_embs: np.ndarray, corpus_embs: np.ndarray) -> np.ndarray:
    """(n_seeds, n_corpus) cosine similarities."""
    return seed_embs @ corpus_embs.T


def aggregate_mean(seed_embs: np.ndarray, corpus_embs: np.ndarray) -> np.ndarray:
    """Mean similarity across seeds.

    Computed as the mean of the similarity matrix rather than by averaging the
    seed vectors first. The two give the same *ranking* (cosine is linear in
    the query), but only this form yields scores that are literally the mean
    cosine similarity -- which is what the paper reports.
    """
    return _similarity_matrix(seed_embs, corpus_embs).mean(axis=0)


def aggregate_max(seed_embs: np.ndarray, corpus_embs: np.ndarray) -> np.ndarray:
    """Highest similarity to any single seed. Rewards niche, seed-specific hits."""
    return _similarity_matrix(seed_embs, corpus_embs).max(axis=0)


def aggregate_soft_and(seed_embs: np.ndarray, corpus_embs: np.ndarray,
                       alpha: float = 0.5) -> np.ndarray:
    """alpha * mean + (1 - alpha) * min.

    The min term penalises candidates that match only part of the seed set,
    which is what makes this the balanced choice for diverse queries.
    """
    sims = _similarity_matrix(seed_embs, corpus_embs)
    return alpha * sims.mean(axis=0) + (1.0 - alpha) * sims.min(axis=0)


def aggregate_cluster(seed_embs: np.ndarray, corpus_embs: np.ndarray,
                      n_clusters: int = 3, random_state: int = 42) -> np.ndarray:
    """Cluster seeds, take max within each cluster, average across clusters.

    Gives every sub-theme of the seed set equal weight, so a minority sub-topic
    is not drowned out by a majority one. Falls back to mean when there are too
    few seeds to form the requested number of clusters.
    """
    from sklearn.cluster import KMeans

    k = min(n_clusters, len(seed_embs))
    if k < 2:
        return aggregate_mean(seed_embs, corpus_embs)

    labels = KMeans(n_clusters=k, random_state=random_state, n_init=10) \
        .fit_predict(seed_embs)

    per_cluster = []
    for label in range(k):
        members = seed_embs[labels == label]
        if len(members) == 0:
            continue
        per_cluster.append(_similarity_matrix(members, corpus_embs).max(axis=0))

    return np.mean(per_cluster, axis=0)


STRATEGIES: Dict[str, Callable] = {
    "mean": aggregate_mean,
    "max": aggregate_max,
    "soft_and": aggregate_soft_and,
    "cluster": aggregate_cluster,
}

DISPLAY_NAMES = {
    "mean": "Mean",
    "max": "Max",
    "soft_and": "Soft-AND",
    "cluster": "Cluster",
}


def rank(
    scores: np.ndarray,
    paper_ids,
    exclude_ids=None,
    top_k: int = 50,
    depth: int = 200,
):
    """Score array -> ranked [(paper_id, score)], seeds removed.

    Seed removal happens BEFORE truncation, so the returned list holds top_k
    genuine results. A seed always scores 1.0 against itself, so leaving seeds
    in would put them at the head of every list and inflate every metric.

    ``depth`` bounds the partial sort; it only needs to exceed
    top_k + len(exclude_ids).
    """
    exclude = set(exclude_ids or ())
    depth = max(depth, top_k + len(exclude))
    depth = min(depth, len(scores))

    # argpartition is O(n); a full argsort over ~150k papers per strategy per
    # query is wasted work.
    candidate_idx = np.argpartition(-scores, depth - 1)[:depth]
    candidate_idx = candidate_idx[np.argsort(-scores[candidate_idx])]

    ranked = []
    for idx in candidate_idx:
        pid = paper_ids[idx]
        if pid in exclude:
            continue
        ranked.append((pid, float(scores[idx])))
        if len(ranked) >= top_k:
            break
    return ranked
