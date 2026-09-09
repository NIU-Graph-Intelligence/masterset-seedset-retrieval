"""Central configuration.

This is the ONLY module in the repository that touches absolute paths or
environment variables. Every other script imports its paths from here, so
moving the project to a new machine requires editing .env and nothing else.

Values in .env may be absolute, or relative to the repository root.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# src/common/config.py -> src/common -> src -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(REPO_ROOT / ".env")


def _path(env_key: str, default: str, base: Path = None) -> Path:
    """Resolve an env value to a Path.

    Absolute values are used as-is. Relative values are resolved against
    ``base`` (the repo root unless stated otherwise).
    """
    base = base or REPO_ROOT
    raw = os.getenv(env_key, default)
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (base / p)


def _str(env_key: str, default: str) -> str:
    return os.getenv(env_key, default)


# --- Directories ------------------------------------------------------------
DATA_DIR = _path("DATA_DIR", "data")
OUTPUT_DIR = _path("OUTPUT_DIR", "outputs")
EMBEDDINGS_DIR = _path("EMBEDDINGS_DIR", "data/embeddings")

# --- Input files ------------------------------------------------------------
# The corpus parquet lives in the external dataset directory.
CORPUS_PARQUET = _path("CORPUS_PARQUET", "candidate_pool_v7.0.parquet", base=DATA_DIR)

# seeds.json is version-controlled alongside the code, so it resolves against
# the REPO, not DATA_DIR. DATA_DIR usually points at an external dataset
# directory that is shared across projects and has no business holding this
# project's query definitions.
SEEDS_JSON = _path("SEEDS_JSON", "data/seeds.json", base=REPO_ROOT)

# Embeddings for the candidate pool. This file and CORPUS_PARQUET must describe
# the SAME set of papers -- retrieving from a pool whose vectors were built from
# a different split silently drops candidates.
# EMBEDDINGS_PATH (a full path) overrides EMBEDDINGS_FILE (relative to
# EMBEDDINGS_DIR); either may be used.
if os.getenv("EMBEDDINGS_PATH"):
    EMBEDDINGS_PATH = _path("EMBEDDINGS_PATH", "", base=EMBEDDINGS_DIR)
else:
    EMBEDDINGS_PATH = _path("EMBEDDINGS_FILE", "candidates_embeddings.pt",
                            base=EMBEDDINGS_DIR)

# Optional secondary source for SEED vectors only.
#
# Under the CoreML-168K setting the candidate pool is cut at year <= 2025 while
# evaluation queries are 2026 papers, so a recent seed can legitimately have a
# vector in the eval file and none in the pool file. Seeds are looked up here
# when the pool does not contain them; the retrieval pool itself is unaffected.
# Set to an empty value to disable.
_seed_emb = os.getenv("SEED_EMBEDDINGS_FILE", "eval_embeddings.pt")
SEED_EMBEDDINGS_PATH = (_path("SEED_EMBEDDINGS_FILE", _seed_emb, base=EMBEDDINGS_DIR)
                        if _seed_emb.strip() else None)

# --- Per-phase output directories -------------------------------------------
PHASE1_DIR = OUTPUT_DIR / "phase1_semantic"
PHASE2_DIR = OUTPUT_DIR / "phase2_graph"
PHASE3_DIR = OUTPUT_DIR / "phase3_fusion"
PHASE4_DIR = OUTPUT_DIR / "phase4_judge"
PHASE5_DIR = OUTPUT_DIR / "phase5_analysis"

# Artifacts shared across phases
GRAPH_PATH = PHASE2_DIR / "citation_graph.pkl"
GRAPH_STATS_PATH = PHASE2_DIR / "graph_stats.json"
SEED_VALIDATION_PATH = PHASE1_DIR / "seed_validation.json"
JUDGE_CACHE_PATH = PHASE4_DIR / "judge_cache.jsonl"

# --- Model identifiers ------------------------------------------------------
SPECTER2_BASE = _str("SPECTER2_BASE", "allenai/specter2_base")
SPECTER2_ADAPTER = _str("SPECTER2_ADAPTER", "allenai/specter2")

# --- Corpus schema ----------------------------------------------------------
REFERENCES_COLUMN = _str("REFERENCES_COLUMN", "references")
REFERENCE_ID_KEY = _str("REFERENCE_ID_KEY", "matched_paper_id")

# --- LLM judge --------------------------------------------------------------
OLLAMA_MODEL = _str("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_HOST = _str("OLLAMA_HOST", "http://localhost:11434")

# --- Retrieval defaults -----------------------------------------------------
# Depth pulled from each ranker before seed removal and truncation. Retrieving
# deeper than TOP_K and truncating afterwards is what keeps the final list at
# exactly TOP_K even after seeds are dropped.
DEFAULT_RETRIEVAL_DEPTH = 200
DEFAULT_TOP_K = 50

# PageRank DAMPING factor -- the probability of following an edge.
# Teleport probability back to the seeds is (1 - damping).
# NOTE: networkx names this argument `alpha`. The paper reports the teleport
# probability (0.15), which corresponds to damping = 0.85.
DEFAULT_DAMPING = 0.85

# Reciprocal Rank Fusion smoothing constant.
DEFAULT_RRF_K = 60


def describe() -> str:
    """Human-readable dump of resolved paths, for run logs."""
    lines = ["Resolved configuration:"]
    for name in [
        "REPO_ROOT", "DATA_DIR", "OUTPUT_DIR", "EMBEDDINGS_DIR",
        "CORPUS_PARQUET", "SEEDS_JSON", "EMBEDDINGS_PATH",
        "SEED_EMBEDDINGS_PATH",
    ]:
        value = globals()[name]
        if value is None:
            lines.append(f"  {name:<21} = (disabled)")
            continue
        exists = "" if not isinstance(value, Path) else ("" if value.exists() else "   [MISSING]")
        lines.append(f"  {name:<21} = {value}{exists}")
    lines.append(f"  {'OLLAMA_MODEL':<21} = {OLLAMA_MODEL}")
    return "\n".join(lines)
