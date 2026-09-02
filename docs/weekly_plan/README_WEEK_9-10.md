# Weeks 9–10: Demo & Final Report

## Before You Start

Great work getting the LLM judge running locally with Qwen 2.5 7B — the consistency results look strong. A few things from earlier phases to wrap up before the final push. These will also make the final report stronger.

### 1. Complete the Phase 3 follow-ups

If you haven't already:
- **Fuse PPR with Soft-AND and Max** (not just Mean). In weeks 3-4 you found that different strategies work better for different query types. It's worth checking if fusing PPR with Soft-AND fixes the Q3 problem where fusion hurt performance.
- **Test alpha sensitivity on Q1 and Q2** (you only tested Q3).
- **Try RRF with k=20 and k=100** on one query to see if it matters.
- **Create a mixed-topic set with seeds that are in the graph** so PPR can actually run on it.

### 2. Evaluate other fusion combinations with the LLM judge

You evaluated Semantic (Mean) vs Graph vs Fused (Mean+PPR). Now also evaluate:
- Fused (Soft-AND + PPR)
- Fused (Max + PPR)

This tells us which semantic strategy works best when combined with graph signal. Just re-run the judge on the new fused results.

### 3. Discuss relevance distributions in your observations

Your Phase 4 data shows that Graph (PPR) produced zero score-5 papers for Q1 and Q2, while Semantic produced some. That's a meaningful finding worth calling out — PPR finds related-but-not-directly-on-topic papers, while semantic finds the closest matches.

---

## What You Need to Build

### Part 1: Demo Script (Days 1–4)

Build a single script that runs the full pipeline end-to-end. A user gives a set of seed papers, and the system returns a ranked list with relevance scores.

Create `src/demo/run_pipeline.py`:

```python
import argparse
import json

def run_demo(seed_file, model="specter2", strategy="mean",
             use_graph=True, use_judge=True, top_k=20):
    """
    Full pipeline:
    1. Load seed papers from JSON file
    2. Embed seeds using SPECTER2
    3. Run semantic retrieval with chosen aggregation strategy
    4. (Optional) Run PPR on citation graph
    5. (Optional) Fuse with RRF
    6. (Optional) Score results with LLM judge
    7. Print and save ranked results
    """
    
    # Step 1: Load seeds
    with open(seed_file) as f:
        seeds = json.load(f)
    print(f"Loaded {len(seeds)} seed papers")
    
    # Step 2-3: Semantic retrieval
    # ... (reuse your existing functions)
    
    # Step 4: Graph retrieval (if enabled)
    # ... (reuse your PPR code)
    
    # Step 5: Fusion (if both signals available)
    # ... (reuse your RRF code)
    
    # Step 6: LLM judge (if enabled)
    # ... (reuse your judge code)
    
    # Step 7: Output
    # Print results as a clean table
    # Save to JSON


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", required=True, help="Path to seed papers JSON")
    parser.add_argument("--strategy", default="mean",
                        choices=["mean", "max", "soft-and", "cluster"])
    parser.add_argument("--no-graph", action="store_true",
                        help="Skip graph retrieval")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip LLM relevance scoring")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    
    run_demo(args.seeds, strategy=args.strategy,
             use_graph=not args.no_graph,
             use_judge=not args.no_judge,
             top_k=args.top_k)
```

**Seed file format** (JSON):

```json
{
    "topic": "Neural Algorithmic Reasoning",
    "papers": [
        {
            "title": "Tropical Attention: Neural Algorithmic Reasoning...",
            "abstract": "We introduce Tropical Attention..."
        },
        {
            "title": "Primal-Dual Neural Algorithmic Reasoning",
            "abstract": "We introduce a general NAR framework..."
        }
    ]
}
```

**The demo should:**
- Accept any set of seed papers (not just the hardcoded ones)
- Let the user choose the aggregation strategy
- Optionally skip graph or judge steps (useful for seeds not in the graph)
- Print a clean table: rank, title, retrieval score, relevance score (if judged)
- Save full results to a JSON file

### Part 2: Final Report (Days 5–10)

Write a report (3-5 pages) covering the entire project. Structure:

**1. Introduction (half a page)**
- What is the MasterSet benchmark?
- What problem are we solving? (Set-based citation retrieval)
- Why does it matter?

**2. Methods (1-1.5 pages)**
- Semantic retrieval: SPECTER2 embeddings, four aggregation strategies (Mean, Max, Soft-AND, Cluster). Briefly explain each.
- Graph retrieval: citation graph construction, Personalized PageRank. Explain how the seed set becomes the personalization vector.
- Fusion: Reciprocal Rank Fusion. Explain the formula.
- Evaluation: LLM-as-judge with Qwen 2.5 7B. Explain the 1-5 scoring rubric.

**3. Results (1-1.5 pages)**

Include these tables (pull numbers from your deliverables):

*Table 1: Aggregation strategy comparison (Phase 2)*
- Jaccard overlap between strategies for each query
- Key finding: topic cohesion determines strategy agreement

*Table 2: Semantic vs Graph vs Fused (Phase 3)*
- Jaccard overlap showing complementarity
- PPR unique finds per query

*Table 3: LLM judge evaluation (Phase 4)*
- Avg relevance @10, @50, Precision@10 for all methods × all queries
- Judge consistency stats

*Table 4: Relevance distribution*
- Score 1-5 counts for each method

**4. Discussion (half a page)**
- When does fusion help? (Broad topics — yes. Narrow topics — not always.)
- When does PPR fail? (Seeds not in graph. Papers with no citations.)
- What did the mixed-topic experiments show?
- Which strategy should you use when? (Mean for focused, Soft-AND for diverse, Fusion for broad)

**5. Limitations and Future Work (quarter page)**
- PPR depends on seeds being in the graph
- LLM judge may have biases (only one model tested)
- Could try learned fusion weights instead of fixed RRF
- Could try other embedding models beyond SPECTER2

**6. Conclusion (quarter page)**
- Summarize the main finding: dual-space retrieval (semantic + graph) with fusion outperforms either signal alone for broad research topics.

**Format:** Write it as a Google Doc or Word document. Include all tables and figures. Cite the relevant papers (ColBERT, SPECTER, the PPR paper).

---

## Deliverables by End of Week 10

1. **Phase 3 and 4 follow-ups completed** — other fusion strategies evaluated, distributions discussed.
2. **Demo script** — working end-to-end pipeline with command-line interface.
3. **Final report** — 3-5 pages covering all phases.
4. **Clean codebase** — all code in the repo, organized, commented. Final PR to main.
5. **All results** in the shared drive — JSONs, comparison tables, everything.

---

## Timeline

| Day | Task |
|-----|------|
| 1 | Complete any remaining follow-ups from earlier phases |
| 2-3 | Build the demo script, test with all query sets |
| 4 | Polish demo — clean output formatting, error handling |
| 5-6 | Write the report — methods and results sections |
| 7-8 | Write the report — discussion, intro, conclusion |
| 9 | Review everything, clean up the repo |
| 10 | Final meeting with Ratul and Dr. Zhang |

---

## Tips

- **The report matters most.** The demo is a nice deliverable, but the report is what Dr. Zhang will read. Spend more time on writing clearly than on making the demo fancy.
- **Use your own deliverables.** You've already written observations documents for each phase — reuse that content. Don't start from scratch.
- **Tables over text.** Put numbers in tables, not in paragraphs. Much easier to read.
- **Be honest about limitations.** The mixed set failure and the Q3 regression are not bad results — they're interesting findings. Write about them.