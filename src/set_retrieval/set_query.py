import os
import json
import torch
import numpy as np
import polars as pl
from pathlib import Path
from dotenv import load_dotenv
from sklearn.cluster import KMeans

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))

# using specter2 embeddings from weeks 1-2
EMBED_DIR = Path("C:/Users/Jagan/OneDrive/Desktop/MasterSet/output/dense/SPECTER2-pretrained/embeddings/")
TRAIN_PARQUET = DATA_DIR / "train_v2.0.parquet"
TOP_K = 100


def load_embeddings():
    print("Loading train embeddings...")
    data = torch.load(EMBED_DIR / "train_embeddings.pt", map_location="cpu", weights_only=False)
    embs = data["embeddings"].numpy().astype("float32")
    pids = data["paper_ids"]
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms
    print(f"  Loaded {len(pids)} papers")
    return embs, pids


def load_titles():
    df = pl.read_parquet(TRAIN_PARQUET, columns=["paper_id", "title"])
    return {row["paper_id"]: row["title"] for row in df.iter_rows(named=True)}


def get_seed_embeddings(seed_ids, train_embs, train_pids):
    pid_to_idx = {pid: i for i, pid in enumerate(train_pids)}
    found = []
    for sid in seed_ids:
        if sid in pid_to_idx:
            found.append(train_embs[pid_to_idx[sid]])
        else:
            print(f"  Warning: seed {sid} not found")
    return np.array(found, dtype="float32") if found else None


# strategy 1: mean
def aggregate_mean(seed_embs, train_embs):
    mean_vec = seed_embs.mean(axis=0)
    mean_vec /= (np.linalg.norm(mean_vec) + 1e-9)
    return train_embs @ mean_vec


# strategy 2: max
def aggregate_max(seed_embs, train_embs):
    sim_matrix = seed_embs @ train_embs.T
    return sim_matrix.max(axis=0)


# strategy 3: soft-and 
def aggregate_soft_and(seed_embs, train_embs, alpha=0.5):
    sim_matrix = seed_embs @ train_embs.T
    mean_scores = sim_matrix.mean(axis=0)
    min_scores = sim_matrix.min(axis=0)
    return alpha * mean_scores + (1 - alpha) * min_scores


# strategy 4: cluster then aggregate
def aggregate_cluster(seed_embs, train_embs):
    if len(seed_embs) < 3:
        print("  Not enough seeds to cluster, using mean")
        return aggregate_mean(seed_embs, train_embs)

    n_clusters = min(3, len(seed_embs))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(seed_embs)

    cluster_scores = []
    for label in range(n_clusters):
        cluster_embs = seed_embs[kmeans.labels_ == label]
        sim_matrix = cluster_embs @ train_embs.T
        cluster_scores.append(sim_matrix.max(axis=0))

    return np.mean(cluster_scores, axis=0)


def retrieve(seed_ids, strategy_fn, train_embs, train_pids, k=TOP_K):
    seed_embs = get_seed_embeddings(seed_ids, train_embs, train_pids)
    if seed_embs is None:
        return []
    scores = strategy_fn(seed_embs, train_embs)
    seed_set = set(seed_ids)
    results = []
    for idx in np.argsort(-scores):
        pid = train_pids[idx]
        if pid not in seed_set:
            results.append((pid, float(scores[idx])))
        if len(results) >= k:
            break
    return results


def print_results(results, pid_to_title, top_n=20):
    for rank, (pid, score) in enumerate(results[:top_n], 1):
        title = pid_to_title.get(pid, "[unknown]")
        title_short = title[:80] + "..." if len(title) > 80 else title
        print(f"  {rank:2d}. [{score:.4f}] {title_short}")


if __name__ == "__main__":
    train_embs, train_pids = load_embeddings()
    pid_to_title = load_titles()

  
    query_name = "Query 1: Neural Algorithmic Reasoning"
    seed_ids = [
        "0133e9c0-f893-5504-b8c1-b7b05d869d95",
        "5bf0c02f-8ed2-5e97-9161-541558feab35",
        "dfbeee5c-e0e2-5942-9441-280635e57976",
        "1bc7f6e0-b0ae-5038-8787-5c119e4af13f",
        "b72c39fd-bd6f-5725-95df-9a2039c6c3a3",
        "e5b7c941-e9e9-5906-b170-68c1e3e27ad2",
        "8a7ccafe-8c36-5c4a-b729-470a5c679190",
        "a161b3e2-3e20-56a8-a30b-a1e2686fd7cb",
        "5a59220e-42bf-52b8-b824-ff2dd7004d1f",
        "1d2cc124-9fa8-5a19-b0d0-5a331a65f35",
    ]

    print(f"\n{query_name} ({len(seed_ids)} seeds)")
    pid_set = set(train_pids)
    found = sum(1 for s in seed_ids if s in pid_set)
    print(f"Seeds found in train set: {found}/{len(seed_ids)}")

    print("\n-- Mean --")
    results = retrieve(seed_ids, aggregate_mean, train_embs, train_pids)
    print_results(results, pid_to_title)