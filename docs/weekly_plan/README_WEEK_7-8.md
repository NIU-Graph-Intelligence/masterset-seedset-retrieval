# Weeks 7–8: LLM-as-Judge Evaluation & Phase 3 Follow-ups

## Before You Start

Great work on the graph retrieval and fusion pipeline — the results are looking solid. Before moving into the new phase, there are a few things from weeks 5–6 worth finishing up. These shouldn't take more than a couple of days.

### 1. Confirm you're using v1.2 data

Make sure all your results (embeddings, set retrieval, graph, fusion) are based on the v1.2 parquet files shared in the drive. If not, re-run with v1.2 — the code is the same, just swap the input paths.

### 2. Try other aggregation strategies in fusion

You fused PPR with SPECTER2 Mean only. In weeks 3–4, you found that different strategies work better for different query types (Mean for focused topics, Soft-AND for diverse seeds). Try fusing PPR with at least Soft-AND and Max as well, and compare. Does the choice of semantic strategy matter when combined with graph signal?

### 3. Test alpha on more queries

You tested alpha sensitivity (0.10, 0.15, 0.20) on Q3 only. Run the same test on Q1 and Q2. The graph structure might behave differently for those topics. Quick to run — just a few extra lines.

### 4. Try different RRF k values

You used k=60 (the default). Try k=20 and k=100 as well on one query. k controls how much weight goes to the top-ranked papers vs lower-ranked ones. See if it changes the fused results meaningfully.

### 5. Create a mixed set with seeds that are in the graph

Your mixed-topic set couldn't run PPR because none of the seeds were in the training data. Create a new mixed set using papers that are actually in the train set, so you can test the mixed-topic hypothesis with both semantic and graph signals.

Once these are done, move on to the main weeks 7–8 work below.

---

## Where We Are

So far, all your evaluation has been **qualitative** — you looked at the retrieved paper titles and judged "does this seem relevant to the topic?" That works for getting a feel for the results, but it doesn't scale and it's subjective.

Now we add **automated evaluation** using an LLM as a judge. The idea: give an LLM the seed set topic and a retrieved paper's title + abstract, and ask it to judge how relevant that paper is. This gives us a relevance score for every retrieved paper, which we can then use to compute proper metrics across all our strategies and fusion methods.

---

## Key Concepts

### LLM-as-Judge

Instead of manually reading every retrieved paper, we ask an LLM (like GPT-4 or Claude) to judge relevance. The LLM gets:

- A description of the seed set topic (e.g., "Neural Algorithmic Reasoning")
- The titles of the seed papers (so it knows what the topic looks like)
- A retrieved paper's title and abstract

It returns a relevance score, for example on a 1–5 scale:

| Score | Meaning |
|-------|---------|
| 5 | Directly on-topic — could be a seed paper itself |
| 4 | Closely related — addresses the same research question or method |
| 3 | Somewhat related — shares methods or problem domain |
| 2 | Loosely related — tangential connection |
| 1 | Not relevant |

### Why This Matters

Right now you can say "Mean's top-10 looks better than Max's top-10 for Q3" but you can't say *how much* better. With LLM judging, you can compute things like: "Mean's top-50 has an average relevance of 3.8 while Max's has 3.2." That's a real, comparable number.

---

## What You Need to Build

### Step 1: Design the Prompt (Day 1)

The prompt needs to be clear and consistent. Here's a starting template:

```python
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

Respond with ONLY the number (1-5) and a one-sentence justification.
"""
```

**Important:** Test this prompt on a few examples manually first. Pick 5 papers from your Q1 results that you think range from very relevant to not relevant, and see if the LLM agrees with your judgment.

### Step 2: Build the Evaluation Script (Days 2–3)

Create `src/evaluation/llm_judge.py`:

```python
import json
import time
from pathlib import Path

def judge_paper(topic_name, seed_titles, candidate_title, candidate_abstract, client):
    """Ask the LLM to judge relevance of one paper."""
    seed_list = "\n".join(f"- {t}" for t in seed_titles)
    
    prompt = JUDGE_PROMPT.format(
        topic_name=topic_name,
        seed_titles=seed_list,
        candidate_title=candidate_title,
        candidate_abstract=candidate_abstract
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or whatever model you have access to
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # deterministic
        max_tokens=100
    )
    
    reply = response.choices[0].message.content.strip()
    # Parse the score (first character should be 1-5)
    score = int(reply[0])
    justification = reply[2:].strip() if len(reply) > 2 else ""
    return score, justification


def evaluate_retrieval(topic_name, seed_titles, retrieved_papers, client, delay=0.5):
    """Judge all retrieved papers for one query."""
    results = []
    for pid, title, abstract, retrieval_score in retrieved_papers:
        score, justification = judge_paper(
            topic_name, seed_titles, title, abstract, client
        )
        results.append({
            "paper_id": pid,
            "title": title,
            "relevance_score": score,
            "justification": justification,
            "retrieval_score": retrieval_score
        })
        time.sleep(delay)  # rate limiting
    return results
```

**Which LLM to use:** Check with Ratul about API access. `gpt-4o-mini` is cheap and fast. If we have access to Claude, that works too. The key is to use the same model for all evaluations so scores are comparable.

### Step 3: Evaluate All Your Results (Days 4–5)

For each query set, evaluate the top 50 retrieved papers from:

1. **Semantic only** (SPECTER2 Mean)
2. **Graph only** (PPR)
3. **Fused** (RRF)

That's 3 methods × 3 queries × 50 papers = 450 LLM calls. At ~1 second each, this takes about 8 minutes per run. Very doable.

For each method, compute:

```python
def average_relevance(results, top_k=None):
    """Average relevance score across top-k papers."""
    scores = [r["relevance_score"] for r in results[:top_k]]
    return sum(scores) / len(scores)

def precision_at_k_by_threshold(results, k, threshold=4):
    """Fraction of top-k papers with relevance >= threshold."""
    top_k = results[:k]
    relevant = sum(1 for r in top_k if r["relevance_score"] >= threshold)
    return relevant / k

def relevance_distribution(results):
    """Count how many papers got each score."""
    from collections import Counter
    return Counter(r["relevance_score"] for r in results)
```

Metrics to report for each method:

| Metric | What It Tells You |
|--------|------------------|
| Average relevance (top 10) | Overall quality of the best results |
| Average relevance (top 50) | Quality across a broader set |
| Precision@10 (threshold=4) | Fraction of top 10 that are closely relevant |
| Relevance distribution | How many 5s, 4s, 3s, 2s, 1s in the top 50 |

### Step 4: Compare Methods Quantitatively (Days 6–7)

Now you can answer the key questions with numbers:

1. **Which retrieval method produces the most relevant results?** Compare average relevance across semantic, graph, and fused.
2. **Does fusion actually help?** If fused has higher average relevance than either signal alone, fusion adds value.
3. **Which aggregation strategy works best when fused?** If you did the follow-up from above (fusing PPR with Mean vs Soft-AND vs Max), compare their judged relevance.
4. **Where does each method fail?** Look at papers scored 1 or 2 — what kind of papers are getting through that shouldn't be?

### Step 5: Check Judge Consistency (Day 8)

LLM judges aren't perfect. Run a small consistency check:

1. Pick 20 papers across all queries
2. Judge each paper **3 times** (same prompt, temperature=0)
3. Check if the scores are the same every time

If the scores vary a lot, we may need to average multiple judgments or adjust the prompt. Also, manually check 10-15 judgments yourself — do you agree with the LLM's scores?

Save these consistency results — they're important for the final report.

---

## Deliverables by End of Week 8

1. **Phase 3 follow-ups completed** — fusion with other strategies, alpha on all queries, different k values, graph-compatible mixed set.
2. **LLM judge script** — takes a query set and a list of retrieved papers, returns relevance scores.
3. **Evaluation results** — average relevance, precision, and distribution for semantic-only, graph-only, and fused across all query sets.
4. **Comparison table** — one table showing which method wins on which metric.
5. **Judge consistency check** — results from the repeated evaluation.
6. **Observations document (half a page)** — which method actually produces the most relevant papers? Does fusion help? Where does the LLM judge disagree with your intuition?

As before: pull request to main, results to the shared drive.

---

## How This Connects to What Comes Next

**Weeks 9–10 (Demo + Final Report):** You'll package everything into a working demo (seed set in → fused ranked list + relevance scores out) and write the final report covering all four phases. The LLM judge scores from this phase become the main evaluation in the report.

---

## Setup

You'll need an API key for the LLM. Options:

```bash
pip install openai   # for GPT-4o-mini
# or
pip install anthropic  # for Claude
```

Store the API key in your `.env` file:
```
OPENAI_API_KEY=your_key_here
```

Check with Ratul about which API to use and how to get access.

---

## Tips

- **Start with a small test.** Judge 5 papers manually, then compare with the LLM's judgment. Make sure the prompt works before running 450 calls.
- **Save everything.** Save the raw LLM responses (score + justification) to JSON. You'll want to look at the justifications later.
- **Use temperature=0.** This makes the LLM give the same answer every time for the same input. Important for reproducibility.
- **Rate limiting matters.** Add a small delay between API calls. Most APIs have rate limits and you don't want to get blocked mid-evaluation.