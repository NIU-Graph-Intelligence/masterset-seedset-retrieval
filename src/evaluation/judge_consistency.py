import json
import time
from pathlib import Path

from llm_judge import load_abstracts, judge_paper


OUTPUT_DIR = Path("output")
RESULTS_DIR = OUTPUT_DIR / "judge_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_PAPERS = 20
NUM_RUNS = 3


if __name__ == "__main__":
    topic_name = "Neural Algorithmic Reasoning"

    seed_titles = [
        "Tropical Attention: Neural Algorithmic Reasoning for Combinatorial Algorithms",
        "Primal-Dual Neural Algorithmic Reasoning",
        "Discrete Neural Algorithmic Reasoning",
        "Open-Book Neural Algorithmic Reasoning",
        "Deep Equilibrium Algorithmic Reasoning",
    ]

    # Use the first 20 papers from the semantic evaluation
    semantic_file = RESULTS_DIR / "query_1_neural_algorithmic_reasoning_judged.json"

    with open(semantic_file, "r") as f:
        data = json.load(f)

    papers = data["semantic"]["papers"][:NUM_PAPERS]

    paper_map = load_abstracts()

    results = []

    print("=" * 70)
    print("LLM JUDGE CONSISTENCY CHECK")
    print("=" * 70)
    print(f"Testing {NUM_PAPERS} papers x {NUM_RUNS} runs = {NUM_PAPERS * NUM_RUNS} judgments\n")

    for i, paper in enumerate(papers):
        pid = paper["paper_id"]
        info = paper_map.get(pid, {})

        title = info.get("title", paper["title"])
        abstract = info.get("abstract", "")

        scores = []
        justifications = []

        print(f"[{i + 1:2d}/{NUM_PAPERS}] {title[:70]}")

        for run in range(NUM_RUNS):
            score, justification = judge_paper(
                topic_name,
                seed_titles,
                title,
                abstract
            )

            scores.append(score)
            justifications.append(justification)

            print(f"    Run {run + 1}: Score={score}")

        results.append({
            "paper_id": pid,
            "title": title,
            "scores": scores,
            "justifications": justifications,
            "all_agree": len(set(scores)) == 1,
            "score_range": max(scores) - min(scores)
        })

    # Calculate consistency statistics
    valid_results = [r for r in results if all(s > 0 for s in r["scores"])]

    exact_agreement = sum(r["all_agree"] for r in valid_results)

    total_scores = sum(len(r["scores"]) for r in valid_results)
    score_range_zero = sum(
        1 for r in valid_results if r["score_range"] == 0
    )

    avg_range = (
        sum(r["score_range"] for r in valid_results) / len(valid_results)
        if valid_results else 0
    )

    consistency_rate = (
        exact_agreement / len(valid_results)
        if valid_results else 0
    )

    summary = {
        "num_papers": NUM_PAPERS,
        "num_runs_per_paper": NUM_RUNS,
        "total_judgments": NUM_PAPERS * NUM_RUNS,
        "valid_papers": len(valid_results),
        "exact_agreement_papers": exact_agreement,
        "exact_agreement_rate": round(consistency_rate, 3),
        "average_score_range": round(avg_range, 3),
        "results": results
    }

    output_file = RESULTS_DIR / "judge_consistency.json"

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("CONSISTENCY SUMMARY")
    print("=" * 70)
    print(f"Papers tested:              {NUM_PAPERS}")
    print(f"Runs per paper:             {NUM_RUNS}")
    print(f"Total judgments:            {NUM_PAPERS * NUM_RUNS}")
    print(f"Exact agreement:            {exact_agreement}/{len(valid_results)}")
    print(f"Exact agreement rate:       {consistency_rate:.1%}")
    print(f"Average score range:        {avg_range:.3f}")
    print(f"\nSaved to: {output_file}")