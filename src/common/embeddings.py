"""Corpus embedding matrix, and seed vector lookup.

Design note -- why seeds are LOOKED UP rather than re-encoded
------------------------------------------------------------
Every seed paper carries a paper_id that exists in the candidate pool, so its
SPECTER2 vector is already sitting in the precomputed matrix. Re-encoding the
seed from its title and abstract produces a *different* vector than the one
the corpus was indexed with, because the corpus encoder saw a specific text
format, truncation length, and adapter revision.

Comparing a freshly-encoded query against a differently-encoded index is a
silent source of error. Lookup is also faster and drops the transformers /
adapters dependency from the retrieval path entirely.

``encode_texts`` remains available as an explicit fallback for seeds that are
genuinely absent from the pool, and every use of it is logged and reported.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import config


@dataclass
class CorpusEmbeddings:
    matrix: np.ndarray          # (n_papers, dim), L2-normalised, float32
    paper_ids: List[str]
    index_of: Dict[str, int]

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1])

    def __len__(self) -> int:
        return len(self.paper_ids)

    def vector_for(self, paper_id: str):
        idx = self.index_of.get(paper_id)
        return None if idx is None else self.matrix[idx]

    def contains(self, paper_id: str) -> bool:
        return paper_id in self.index_of


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def load_corpus_embeddings(path: Path = None, verbose: bool = True) -> CorpusEmbeddings:
    import torch

    path = path or config.EMBEDDINGS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {path}\n"
            f"Set EMBEDDINGS_DIR / EMBEDDINGS_FILE in .env."
        )

    if verbose:
        print(f"Loading corpus embeddings from {path} ...")
    blob = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(blob, dict) or "embeddings" not in blob or "paper_ids" not in blob:
        raise ValueError(
            f"{path} must be a dict with keys 'embeddings' and 'paper_ids'. "
            f"Got: {type(blob).__name__}"
        )

    raw = blob["embeddings"]
    matrix = raw.numpy() if hasattr(raw, "numpy") else np.asarray(raw)
    matrix = l2_normalise(matrix)
    paper_ids = list(blob["paper_ids"])

    if len(paper_ids) != matrix.shape[0]:
        raise ValueError(
            f"Row count mismatch in {path}: {matrix.shape[0]} vectors "
            f"vs {len(paper_ids)} paper_ids."
        )

    index_of = {pid: i for i, pid in enumerate(paper_ids)}
    if verbose:
        print(f"  {len(paper_ids):,} papers, dim={matrix.shape[1]}")
    return CorpusEmbeddings(matrix=matrix, paper_ids=paper_ids, index_of=index_of)


def load_seed_sources(verbose: bool = True):
    """Embedding sources for SEED lookup, in priority order.

    The candidate pool comes first, then the eval file if configured. Under the
    CoreML-168K setting the pool is cut at year <= 2025 while eval queries are
    2026 papers, so a recent seed may legitimately have a vector only in eval.
    Both files come from the same encoder, so the vectors are comparable.

    This affects SEED lookup only. Retrieval always ranks the pool.
    """
    sources = [("pool", load_corpus_embeddings(config.EMBEDDINGS_PATH, verbose))]

    secondary = config.SEED_EMBEDDINGS_PATH
    if secondary and secondary.exists() and secondary != config.EMBEDDINGS_PATH:
        if verbose:
            print(f"Loading secondary seed embeddings from {secondary} ...")
        extra = load_corpus_embeddings(secondary, verbose)
        if extra.dim != sources[0][1].dim:
            raise ValueError(
                f"Dimension mismatch: {config.EMBEDDINGS_PATH.name} has dim "
                f"{sources[0][1].dim}, {secondary.name} has dim {extra.dim}. "
                f"They must come from the same encoder."
            )
        sources.append(("eval", extra))
    elif secondary and verbose:
        print(f"[note] secondary seed embeddings not found: {secondary}")

    return sources


def resolve_seed_vectors(
    sources,
    seed_ids: Sequence[str],
    seed_texts: Sequence[Tuple[str, str]] = None,
    allow_encode_fallback: bool = False,
    verbose: bool = True,
):
    """Assemble the (n_seeds, dim) query matrix from the available sources.

    Returns (matrix, resolved_ids, provenance) where provenance maps each
    paper_id to the source that supplied its vector: a source name, or
    "encoded" when it was generated from text.
    """
    rows, resolved, provenance, missing = [], [], {}, []

    for i, pid in enumerate(seed_ids):
        for name, corpus in sources:
            vec = corpus.vector_for(pid)
            if vec is not None:
                rows.append(vec)
                resolved.append(pid)
                provenance[pid] = name
                break
        else:
            missing.append((i, pid))

    if missing:
        if not allow_encode_fallback:
            detail = "\n".join(f"    - {pid}" for _, pid in missing)
            names = ", ".join(n for n, _ in sources)
            raise KeyError(
                f"{len(missing)} seed paper(s) have no vector in any embedding "
                f"source ({names}):\n{detail}\n"
                f"Run validate_seeds.py to inspect, then either fix the "
                f"paper_ids in seeds.json or pass --allow-encode-fallback."
            )
        if seed_texts is None:
            raise ValueError("allow_encode_fallback=True requires seed_texts.")
        if verbose:
            print(f"  Encoding {len(missing)} seed(s) from text")
        encoded_matrix = encode_texts([seed_texts[i] for i, _ in missing],
                                      verbose=verbose)
        for (idx, pid), vec in zip(missing, encoded_matrix):
            rows.append(vec)
            resolved.append(pid)
            provenance[pid] = "encoded"

    if not rows:
        raise ValueError("No seed vectors could be resolved.")

    return l2_normalise(np.vstack(rows)), resolved, provenance


def seed_matrix(corpus, seed_ids, seed_texts=None,
                allow_encode_fallback=False, verbose=True):
    """Backwards-compatible single-source wrapper around resolve_seed_vectors."""
    matrix, resolved, provenance = resolve_seed_vectors(
        [("pool", corpus)], seed_ids, seed_texts, allow_encode_fallback, verbose)
    encoded = [pid for pid, src in provenance.items() if src == "encoded"]
    return matrix, resolved, encoded


def encode_texts(texts: Sequence[Tuple[str, str]], verbose: bool = True) -> np.ndarray:
    """Encode (title, abstract) pairs with SPECTER2. Fallback path only."""
    import torch
    from transformers import AutoTokenizer
    from adapters import AutoAdapterModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"  Loading SPECTER2 ({config.SPECTER2_BASE}) on {device} ...")

    tokenizer = AutoTokenizer.from_pretrained(config.SPECTER2_BASE)
    model = AutoAdapterModel.from_pretrained(config.SPECTER2_BASE)
    model.load_adapter(config.SPECTER2_ADAPTER, source="hf",
                       load_as="proximity", set_active=True)
    model = model.to(device).eval()

    vectors = []
    with torch.no_grad():
        for title, abstract in texts:
            # Must match the corpus generation script byte for byte:
            #     f"{title} [SEP] {abstract}"   (spaces around the separator)
            text = f"{title} [SEP] {abstract}"
            inputs = tokenizer(text, padding=True, truncation=True, max_length=512,
                               return_tensors="pt",
                               return_token_type_ids=False).to(device)
            outputs = model(**inputs)
            vectors.append(outputs.last_hidden_state[:, 0, :].cpu().numpy()[0])

    return l2_normalise(np.vstack(vectors))
