import os
import json
import time
import polars as pl
from pathlib import Path
from dotenv import load_dotenv
import ollama
import re

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path("output")
RESULTS_DIR = OUTPUT_DIR / "judge_results"
TRAIN_PARQUET = DATA_DIR / "train_v1.2.parquet"


JUDGE_PROMPT = """You are an expert in AI/ML research. You will be given a research topic defined by a set of seed papers, and a candidate paper. Your job is to judge how relevant the candidate paper is to the research topic.

Research Topic: {topic_name}

Seed Papers (these define the topic):
{seed_titles}

Candidate Paper:
Title: {candidate_title}
Abstract: {candidate_abstract}

How relevant is this candidate paper to the research topic defined by the seed papers?

Rate on a scale of 1-5:
5 = Directly on-topic — could be a seed paper itself
4 = Closely related — addresses the same research question or method
3 = Somewhat related — shares methods or problem domain
2 = Loosely related — tangential connection
1 = Not relevant

Respond with ONLY the number (1-5) followed by a period and a one-sentence justification.
Example: 4. This paper directly addresses neural algorithmic reasoning using graph neural networks.
"""


def load_abstracts():
    print("Loading paper abstracts...")
    df = pl.read_parquet(TRAIN_PARQUET, columns=["paper_id", "title", "abstract"])
    return {row["paper_id"]: {"title": row["title"], "abstract": row["abstract"] or ""}
            for row in df.iter_rows(named=True)}


def judge_paper(topic_name, seed_titles, candidate_title, candidate_abstract):
    seed_list = "\n".join(f"- {t}" for t in seed_titles)

    prompt = JUDGE_PROMPT.format(
        topic_name=topic_name,
        seed_titles=seed_list,
        candidate_title=candidate_title,
        candidate_abstract=candidate_abstract[:500]
    )

    response = ollama.chat(
        model="qwen2.5:7b-instruct",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    reply = response["message"]["content"].strip()

    try:
        match = re.search(r'\b([1-5])\b', reply)

        if match:
            score = int(match.group(1))
            justification = reply[match.end():].strip()

            if justification.startswith("."):
                justification = justification[1:].strip()
        else:
            score = 0
            justification = reply

    except Exception:
        score = 0
        justification = reply

    return score, justification


def average_relevance(results, top_k=None):
    scores = [r["relevance_score"] for r in results[:top_k] if r["relevance_score"] > 0]
    return round(sum(scores) / len(scores), 3) if scores else 0


def precision_at_k(results, k, threshold=4):
    top = results[:k]
    relevant = sum(1 for r in top if r["relevance_score"] >= threshold)
    return round(relevant / k, 3)


def relevance_distribution(results):
    dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for r in results:
        s = r["relevance_score"]
        if s in dist:
            dist[s] += 1
    return dist


if __name__ == "__main__":
    topic_name = "Neural Algorithmic Reasoning"
    seed_titles = [
        "Tropical Attention: Neural Algorithmic Reasoning for Combinatorial Algorithms",
        "Primal-Dual Neural Algorithmic Reasoning",
        "Discrete Neural Algorithmic Reasoning",
        "Open-Book Neural Algorithmic Reasoning",
        "Deep Equilibrium Algorithmic Reasoning",
    ]

    test_cases = [
        ("The CLRS Algorithmic Reasoning Benchmark",
         "We introduce a benchmark for neural algorithmic reasoning based on 30 algorithms from CLRS."),
        ("Graph Neural Networks are Dynamic Programmers",
         "We show that GNNs can simulate dynamic programming algorithms and prove expressiveness results."),
        ("Language Models are Few-Shot Learners",
         "We train GPT-3, a large language model with 175B parameters that achieves few-shot performance."),
        ("Attention is Turing-Complete",
         "We prove that transformers with hard attention are Turing complete under certain conditions."),
        ("Decoupled Weight Decay Regularization",
         "We propose AdamW, decoupling weight decay from the gradient update in Adam optimizer."),
    ]

    print(f"Testing judge on 5 papers...\n")
    for title, abstract in test_cases:
        score, justification = judge_paper(topic_name, seed_titles, title, abstract)
        print(f"Score: {score} | {title[:60]}")
        print(f"  → {justification}\n")
        time.sleep(1)