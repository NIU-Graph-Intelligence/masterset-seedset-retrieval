import os
import torch
import numpy as np
import polars as pl
from pathlib import Path
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))

EMBED_DIR = Path("C:/Users/Jagan/OneDrive/Desktop/MasterSet/output/dense/SPECTER2-pretrained/embeddings/")
TRAIN_PARQUET = DATA_DIR / "train_v2.0.parquet"
BASE_MODEL = "allenai/specter2_base"
ADAPTER_NAME = "allenai/specter2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def load_specter2():
    print("Loading SPECTER2 model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoAdapterModel.from_pretrained(BASE_MODEL)
    model.load_adapter(ADAPTER_NAME, source="hf", load_as="proximity", set_active=True)
    model = model.to(DEVICE)
    model.eval()
    print("  Model loaded")
    return tokenizer, model


def embed_seeds(seed_texts, tokenizer, model):
    # seed_texts is a list of (title, abstract) tuples
    embs = []
    with torch.no_grad():
        for title, abstract in seed_texts:
            text = f"{title} [SEP] {abstract}"
            inputs = tokenizer(text, padding=True, truncation=True,
                               max_length=512, return_tensors="pt",
                               return_token_type_ids=False).to(DEVICE)
            outputs = model(**inputs)
            cls_emb = outputs[0][:, 0, :].cpu().numpy()
            embs.append(cls_emb[0])
    embs = np.array(embs, dtype="float32")
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embs / norms


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


def retrieve(seed_embs, strategy_fn, train_embs, train_pids, k=TOP_K):
    scores = strategy_fn(seed_embs, train_embs)
    results = []
    for idx in np.argsort(-scores):
        pid = train_pids[idx]
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
    tokenizer, model = load_specter2()

    # query 1 seeds
    query1_seeds = [
        ("Tropical Attention: Neural Algorithmic Reasoning for Combinatorial Algorithms",
         "We introduce Tropical Attention, an attention mechanism grounded in tropical geometry that lifts the attention kernel into tropical projective space, where reasoning is piecewise-linear and 1-Lipschitz, thus preserving the polyhedral decision structure inherent to combinatorial reasoning."),
        ("Primal-Dual Neural Algorithmic Reasoning",
         "We introduce a general NAR framework grounded in the primal-dual paradigm, a classical method for designing efficient approximation algorithms. By leveraging a bipartite representation between primal and dual variables, we establish an alignment between primal-dual algorithms and Graph Neural Networks."),
        ("Discrete Neural Algorithmic Reasoning",
         "Neural algorithmic reasoning aims to capture computations with neural networks by training models to imitate the execution of classical algorithms. We propose to force neural reasoners to maintain the execution trajectory as a combination of finite predefined states."),
        ("Understanding Transformer Reasoning Capabilities via Graph Algorithms",
         "We investigate transformer scaling regimes able to perfectly solve different classes of algorithmic problems. Our novel representational hierarchy separates 9 algorithmic reasoning problems into classes solvable by transformers in different realistic parameter scaling regimes."),
        ("Transformers Can Do Arithmetic with the Right Embeddings",
         "The poor performance of transformers on arithmetic tasks stems from their inability to keep track of the exact position of each digit. We mend this by adding an embedding to each digit that encodes its position relative to the start of the number."),
        ("PUZZLES: A Benchmark for Neural Algorithmic Reasoning",
         "We introduce PUZZLES, a benchmark based on Simon Tatham's Portable Puzzle Collection, aimed at fostering progress in algorithmic and logical reasoning in RL. PUZZLES contains 40 diverse logic puzzles of adjustable sizes and varying levels of complexity."),
        ("Open-Book Neural Algorithmic Reasoning",
         "We propose a novel open-book learning framework where the network can access and utilize all instances in the training dataset when reasoning for a given instance."),
        ("Deep Equilibrium Algorithmic Reasoning",
         "We study neurally solving algorithms from a different perspective: since the algorithm's solution is often an equilibrium, it is possible to find the solution directly by solving an equilibrium equation."),
        ("Simulation of Graph Algorithms with Looped Transformers",
         "We study the ability of transformer networks to simulate algorithms on graphs. We prove by construction that this architecture can simulate Dijkstra's shortest path, Breadth- and Depth-First Search, and Kosaraju's strongly connected components."),
        ("On the Markov Property of Neural Algorithmic Reasoning: Analyses and Methods",
         "We present ForgetNet, which does not use historical embeddings and thus is consistent with the Markov nature of algorithmic reasoning tasks."),
    ]

    print("\nQuery 1: Neural Algorithmic Reasoning (10 seeds)")
    seed_embs = embed_seeds(query1_seeds, tokenizer, model)

    print("\n-- Mean --")
    results = retrieve(seed_embs, aggregate_mean, train_embs, train_pids)
    print_results(results, pid_to_title)