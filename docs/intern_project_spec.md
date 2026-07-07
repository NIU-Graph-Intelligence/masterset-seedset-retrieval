# Summer Intern Project Specification

**Project Title:** Training-Free Dual-Space Retrieval for Seed-Based Literature Discovery

**Duration:** 10 weeks, 14 hours/week (~140 hours total)

**Direct Mentor:** Lei Zhang, Md Toyaha Rahman Ratul

**Final Deliverable:** Interactive web demo + experimental report with initial results, potentially leading to a co-authored paper submission (IEEE BigData 2026 High School Symposium).

---

## 1. Project Overview

### 1.1 The Problem We're Solving

Researchers spend an enormous amount of time on literature discovery. The typical workflow is broken in a specific way: a researcher has a handful of papers they know are central to their interest (call these "seed papers"), and they need to find related work — but not just any related work. They need papers that are *genuinely relevant in a methodological or problem-formulation sense*, not papers that merely share keywords or topics.

Existing tools have clear limitations:

- **Connected Papers** uses only the citation graph. It misses papers that are semantically related but not directly connected through citations.
- **Semantic Scholar's recommendations** rely on collaborative filtering signals and dense embeddings. They miss papers that are graph-proximate but semantically dissimilar at the surface level.
- **Both** treat the query as a single paper or a keyword search; neither truly handles a *set* of seed papers as a coherent query.

### 1.2 What We're Building

A **training-free dual-space retrieval system** that takes a set of seed papers as input and returns ranked candidate papers based on relevance in both:

1. **Semantic space**: similarity in the content of papers (using structured fields like problem, method, experiment, baseline)
2. **Graph space**: proximity in the citation graph (using Personalized PageRank and related measures)

The key design choices that make this project a research contribution rather than just an engineering exercise:

- **Training-free**: no model fine-tuning. We use off-the-shelf encoders and graph algorithms. This makes the system reproducible, domain-portable, and free of train/test leakage concerns.
- **Set query**: the query is a *set* of papers, not a single one. Set-level aggregation is an under-studied design space and is where most of our methodological novelty will come from.
- **LLM-as-judge for evaluation**: we use LLMs only at evaluation time to assess whether retrieved papers are genuinely relevant, comparing our method against baselines.

### 1.3 Why This Matters

If we get this right, we contribute to three research communities:

- **Information Retrieval (IR)**: a new task formulation (set-based scholarly discovery) and a careful study of set-level fusion strategies.
- **Data Mining (DM)**: an analysis of how citation graph proximity and semantic proximity differ and complement each other at scale.
- **NLP / Scientific Document Processing**: an evaluation methodology for literature discovery using LLM judges.

For you personally as an intern: if the work goes well, you can be co-author on a paper submission. Even if results are modest, you'll come out with a working system, real research experience, and a clear understanding of how an end-to-end retrieval research project is built.

---

## 2. Problem Formulation

### 2.1 Formal Definition

**Input:**
- A set of seed papers Q = {q₁, q₂, ..., qₖ} where each qᵢ is a paper in our database
- (Optional) A budget K specifying how many recommendations to return

**Output:**
- A ranked list of K candidate papers (c₁, c₂, ..., c_K) from the database, excluding Q itself
- Each candidate has a score reflecting its predicted relevance to Q

**Objective:**
- The top-ranked candidates should be papers that a human researcher would consider "highly relevant to the topic represented by Q", including papers that the researcher might not have known about but should know about.

### 2.2 What "Relevant" Means Here

This is the subtle part. Relevance is multi-dimensional:

- **Topical relevance**: same subfield, same problem area
- **Methodological connection**: similar methods, or methods that inherit from / extend / contrast with seed papers
- **Information gain**: the candidate adds something a researcher wouldn't already know from reading just the seeds
- **Temporal positioning**: foundational work that seeds build on, OR contemporary parallel work, OR follow-up extensions

We will operationalize these dimensions in the evaluation protocol (Section 5).

### 2.3 Concrete Example

Imagine the seed set is five papers about Mixture-of-Experts routing in large language models, ranging from 2017 to 2023.

A good system should return:
- Foundational MoE papers (Shazeer et al. 2017, etc.) even if their terminology differs
- Recent routing innovations the seeds may not all cite
- Parallel work on conditional computation that uses different terms
- Key baseline methods used in MoE evaluation

A naive system would return:
- The top-cited LLM papers in general (broad but not specifically relevant)
- Only papers directly cited by the seeds (misses parallel and recent work)

---

## 3. Technical Approach

### 3.1 System Architecture Overview

```
                    Seed Papers Q = {q1, ..., qk}
                              │
                              ▼
            ┌─────────────────┴─────────────────┐
            │                                   │
            ▼                                   ▼
    Semantic Space Retrieval          Graph Space Retrieval
    (Late Interaction over            (Personalized PageRank
     structured fields)                from seed set)
            │                                   │
            ▼                                   ▼
       Per-candidate                       Per-candidate
       semantic score                       graph score
            │                                   │
            └─────────────┬─────────────────────┘
                          ▼
                  Rank Fusion (RRF)
                          │
                          ▼
                Final Ranked List
```

### 3.2 Semantic Space (Already Partially Built)

The current group infrastructure already provides:

- All papers (except few recent conference years) in the database have structured summaries (topics, contributions, methods, and extended_summary) generated offline by an LLM.
- Each field has been independently encoded into vectors. Each paper is represented as multiple vectors, not a single vector.
- A retrieval pipeline using late interaction (MaxSim over field vectors) is implemented.

**Your job for the semantic space**: understand the existing code, then design and implement different ways to handle the *set* of seed papers (not just one query paper).

Strategies to implement and compare:

- **Mean aggregation**: average the per-seed similarity scores
- **Max aggregation**: take the maximum score across seeds
- **Soft-AND**: penalize candidates that score high on only one seed but low on others
- **Cluster-then-aggregate**: cluster the seeds first, take max within cluster, then average across clusters

The interesting research question here: under what conditions does each strategy work best? This is a finding worth publishing.

### 3.3 Graph Space (To Be Built)

The citation graph is already constructed and queryable. You will implement graph-based retrieval from scratch (well, using libraries — don't reimplement PPR by hand).

Strategies to implement:

- **Personalized PageRank (PPR)**: seed set defines the personalization vector. Use `networkx` or `igraph` initially; switch to scalable approximations later if needed.
- **Multi-hop neighborhood expansion**: simpler baseline. K-hop neighbors of seeds, weighted by inverse distance.
- **Random Walk with Restart (RWR)**: a variant of PPR, slightly different formulation.

Key implementation considerations:

- Citation graphs are directed (paper A cites paper B is different from B cites A). Decide whether you treat the graph as directed or undirected, and justify the choice.
- Older papers tend to dominate PPR rankings (they have more incoming edges). You will need to think about whether and how to normalize for this.
- Self-loops, dangling nodes, and orphan papers need handling.

### 3.4 Fusion: Combining the Two Spaces

The two scoring systems produce values on totally different scales (cosine similarity vs. PPR probability mass). Direct linear combination doesn't work. Use rank-based fusion:

**Reciprocal Rank Fusion (RRF):**

```
Score_final(c) = 1/(rank_sem(c) + k) + 1/(rank_graph(c) + k)
```

where k is a smoothing constant (typically 60). This is the standard hybrid retrieval approach in IR and works without tuning.

**Stretch goal**: implement an adaptive fusion where the weights depend on query characteristics (e.g., how cohesive the seed set is, how well-connected the seeds are in the graph). This would be a real research contribution.

### 3.5 LLM-as-Judge Evaluation

For evaluation, we use LLMs (GPT-4o or Claude Sonnet) to assess whether retrieved papers are genuinely relevant to the seed set. The protocol:

- For each (seed set, candidate paper) pair, prompt the LLM to score relevance along multiple dimensions (topical, methodological, information gain).
- Use pairwise comparison when comparing methods: "Given this seed set, which list of recommendations is better, list A or list B?"
- Always include a human-validated subset (around 100 pairs) to verify LLM judgments are reliable.

We will discuss the prompt design in detail during Week 6.

---

## 4. Phased Timeline

The timeline below is aggressive but achievable given that you have a dedicated mentor (Ratul) and existing infrastructure. Each phase has a concrete deliverable.

### Phase 1: Onboarding & Baseline (Weeks 1–2, ~28 hours)

**Goal:** Understand the existing system end-to-end and reproduce baseline results.

**Tasks:**
- Set up the development environment. Get access to the data, the codebase, and the compute resources from Ratul.
- Read through the existing semantic retrieval code. Understand what each module does. Take notes.
- Run the existing pipeline on a small subset of papers. Verify you get the same results as Ratul reports.
- Read 3 papers (provided by Ratul): one on late interaction (ColBERT), one on SPECTER2 or similar scientific embeddings, one on PPR-based paper recommendation.
- Write a 1-page document explaining, in your own words, what the system currently does and what we want to build on top of it.

**End-of-Phase Deliverable:** Working environment + 1-page understanding document + ability to run existing baseline.

### Phase 2: Semantic-Side Set Aggregation (Weeks 3–4, ~28 hours)

**Goal:** Implement and compare set aggregation strategies on the semantic side.

**Tasks:**
- Implement four aggregation strategies: mean, max, soft-AND, cluster-then-aggregate.
- Build a small evaluation set: 20 hand-picked seed sets covering different sizes (k=3, 5, 10) and different topical breadths (narrow vs. wide).
- Run all four strategies on the evaluation set. Inspect the top-20 outputs for each.
- Identify cases where strategies disagree. Examine why.

**End-of-Phase Deliverable:** A notebook with side-by-side comparisons of aggregation strategies, plus initial qualitative observations.

### Phase 3: Graph-Side Retrieval (Weeks 5–6, ~28 hours)

**Goal:** Add the graph space and fuse it with the semantic space.

**Tasks:**
- Implement PPR over the citation graph with seed set as the personalization vector.
- Implement multi-hop neighborhood expansion as a simpler baseline.
- Implement RRF fusion of semantic and graph rankings.
- Run the full dual-space pipeline on the evaluation set from Phase 2.
- Compare: semantic-only, graph-only, dual-space. Note where each approach uniquely contributes.

**End-of-Phase Deliverable:** End-to-end working pipeline. Initial qualitative comparison of three configurations.

### Phase 4: LLM-as-Judge Evaluation Framework (Weeks 7–8, ~28 hours)

**Goal:** Build the evaluation system and run the main experiments.

**Tasks:**
- Design the LLM judge prompts (multiple dimensions: topical, methodological, information gain). Iterate with Ratul.
- Build a larger evaluation set: 50–100 seed sets, sampled to cover different subfields.
- Run all configurations (semantic-only, graph-only, dual-space, plus 1–2 external baselines) through the LLM judge.
- Validate a subset (100 pairs) with manual annotation. Compute LLM-human agreement.
- Produce summary tables of results.

**End-of-Phase Deliverable:** First version of the main experimental results table, plus reliability analysis of the LLM judge.

### Phase 5: Deep Dive + Demo + Report (Weeks 9–10, ~28 hours)

**Goal:** Pick one direction to explore deeper, build the interactive demo, write the final report.

**Tasks:**
- Based on Phase 4 findings, choose ONE direction to investigate more deeply. Candidates:
  - Adaptive fusion (weighting by query characteristics)
  - Failure analysis: where does the dual-space system still fail?
  - Set aggregation deep-dive: is there a principled way to pick the right strategy per query?
- Build the interactive web demo: input = seed paper IDs, output = ranked recommendations with explanations of why each was retrieved (semantic vs. graph contribution).
- Write a 6–8 page report covering the system, the experiments, the findings, and limitations.

**End-of-Phase Deliverable:** Web demo + final report + clean codebase + presentation.

---

## 5. Evaluation Methodology Details

We are not training models, so we don't need train/val/test splits in the traditional sense. But we need rigorous evaluation.

### 5.1 Evaluation Set Construction

We construct evaluation sets by sampling from the database:

- **Random seed sets**: pick a paper, take K-1 of its references as the seed set, target = papers the original paper cites but aren't in the seed. This gives us a weak proxy for ground truth.
- **Curated seed sets**: hand-pick seed sets representing known research topics (e.g., "MoE in LLMs", "contrastive learning for vision"). Evaluate using LLM judge.
- **Survey-based seed sets**: use the reference lists of recent survey papers. The full reference list is the "ideal" recommendation set.

Use a mix of all three.

### 5.2 Metrics

- **Precision@K**: of top K recommendations, how many are judged relevant by LLM (and humans on the subset)?
- **MRR / NDCG**: standard ranked retrieval metrics, using LLM-assigned graded relevance.
- **Coverage**: of the survey-paper reference set, what fraction does our system retrieve in top 100?
- **Novelty**: fraction of retrieved papers that are NOT directly cited by any seed paper (these are the "discovery" candidates).

### 5.3 Baselines to Compare Against

- BM25 over titles + abstracts of seed set
- SPECTER2 average embedding similarity
- Pure PPR on citation graph
- Connected Papers' published algorithm (we can approximate it)

---

## 6. Required Background Reading

Don't try to read everything. Read these, in this order:

**Week 1 (required, before doing any coding):**

1. Khattab & Zaharia. "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT." SIGIR 2020. *(Focus on Section 3, late interaction concept. Skip the BERT pretraining details.)*

2. Cohan et al. "SPECTER: Document-level Representation Learning using Citation-informed Transformers." ACL 2020. *(Focus on the citation-based training signal. Skim the rest.)*

3. Kücüktunç et al. "Diversifying Citation Recommendations." *(arxiv 1209.5809. Focus on Section 2.4 on PPR-based methods.)*

**Week 4 (when LLM-as-judge becomes relevant):**

4. Thomas et al. "Large Language Models Can Accurately Predict Searcher Preferences." 2024. *(How LLMs are used as relevance judges.)*

5. Rahmani et al. "LLMJudge: LLMs for Relevance Judgments." SIGIR 2024 LLM4Eval. *(Methodology for using LLMs in IR evaluation.)*

**Optional / Reference (read when needed):**

- LitFM paper (arxiv 2409.12177): for understanding why training-free is a viable alternative to learned graph+text models.
- Sjögårde 2024 JASIST paper: a systematic comparison of seed-based citation retrieval methods. Good for understanding our baselines.

---

## 7. Working Conventions

### 7.1 Repository Structure

Code lives in the shared group repo. Create a branch for your work: `intern-[your-name]-dual-space`.

```
src/
├── data/         # data loading utilities (already exists)
├── semantic/     # semantic-space retrieval (extend existing code here)
├── graph/        # graph-space retrieval (NEW — your main contribution)
├── fusion/       # rank fusion (NEW)
├── eval/         # evaluation framework (NEW)
└── demo/         # web demo (Phase 5)
notebooks/
├── 01_baseline_reproduction.ipynb
├── 02_set_aggregation_comparison.ipynb
└── ...
```

Commit frequently. Small, focused commits with clear messages.

### 7.2 Experiment Logging

Every experiment goes into `experiments/` as a dated, named subdirectory:

```
experiments/
└── 2025-XX-XX_set_aggregation_strategies/
    ├── config.yaml          # what was run
    ├── results.json         # raw output
    ├── analysis.ipynb       # your analysis
    └── README.md            # one-paragraph summary of findings
```

This will save you weeks later when you're writing the report and need to remember what you did.

### 7.3 Meetings & Communication

- **Weekly 1-on-1 with Ratul**: 30–45 minutes. Come with a written progress update (half a page). Cover: what you did, what worked, what didn't, what you're stuck on, plan for next week.
- **Bi-weekly check-in with the PI**: 30 minutes. Higher-level discussion of direction and findings.
- **Slack / messaging**: if you're stuck for more than 2 hours on a coding/setup issue, ping Ratul. Don't burn half a day silently.

### 7.4 Documentation

- Code: docstrings on every function. Type hints where reasonable.
- Notebooks: a markdown cell at the top explaining the goal of the notebook. Conclusions in markdown cells at the end.
- Experiments: README in each experiment folder. Future you (and the rest of the group) will thank present you.

---

## 8. Stretch Goals (If You're Ahead of Schedule)

Only consider these if Phases 1–5 are on track. Don't get distracted from the main pipeline.

- **Adaptive fusion**: design and test a fusion strategy where the weight between semantic and graph depends on query characteristics. This is a real research contribution if it works.
- **Set aggregation theory**: derive (or empirically characterize) the conditions under which each set aggregation strategy is optimal.
- **Temporal awareness**: extend the system to handle the time dimension explicitly (penalize anachronistic recommendations, e.g., recommending a 2023 paper as a "foundational" reference for a 2017 seed paper).
- **Diversity-aware re-ranking**: top results from the fused system might be redundant. Add an MMR-style diversification step.

---

## 9. Expected Outcomes

By the end of week 10, we expect:

- **Functional system**: an end-to-end pipeline that takes seed papers, runs dual-space retrieval, returns ranked recommendations.
- **Interactive demo**: a web interface where a user can input seed papers and explore results.
- **Experimental report**: ~6-8 pages covering the system design, experiments, results, and limitations.
- **Findings**: at least 2-3 concrete, defensible findings (e.g., "soft-AND aggregation outperforms mean by X% on narrow-topic queries but underperforms on broad queries", or "graph-space and semantic-space have only Y% overlap in top-50, justifying fusion").

If the findings are strong, the next step (post-internship) is paper writing for SIGIR / ECIR / WSDM. You would be a co-author. The PI and Ratul will handle the bulk of the writing, but you'd be involved in revisions and rebuttal.

If results are weaker than hoped, the system and findings still feed into the broader group project, and you still have a substantial portfolio piece.

---

## 10. First-Week Concrete Tasks

To get you started without ambiguity, here is exactly what to do in Week 1:

**Day 1 (3-4 hours)**
- Meet with Ratul. Get access to: the data, the codebase, compute resources, the Slack workspace.
- Read this entire document and the ColBERT paper.

**Day 2 (3-4 hours)**
- Read the SPECTER paper.
- Set up your development environment: clone the repo, install dependencies, verify you can load the data.

**Day 3 (3-4 hours)**
- Walk through the existing semantic retrieval code with Ratul. Take notes on what each module does.
- Run the existing pipeline on a small example. Make sure you understand the data flow end-to-end.

**Day 4 (3-4 hours)**
- Read the Kücüktunç paper (focus on PPR section).
- Write the 1-page understanding document. Discuss with Ratul.
- Plan your Phase 2 implementation.

By end of Week 1, you should be able to answer:
- What does the existing system do, exactly?
- What is the input/output of each module?
- What do I need to add to handle a *set* of seed papers (vs. a single one)?
- What's my plan for Week 2?

---

## Questions to Ask Early

If you finish reading this and don't have at least 5 questions, you didn't read carefully enough. Some good early questions to bring to your first meeting:

- What is the current size of the database and the citation graph?
- Are the structured field summaries reliable, or are there known quality issues?
- Are there particular subfields (NLP / CV / RL) the data covers better than others?
- What compute resources do I have access to? Is GPU needed for any part of this?
- What's the budget for LLM API calls? Will I be limited in how much I can evaluate?
- Has anyone in the group tried set-level queries before? What did they find?

---

*Welcome to the project. Ask questions early, commit often, and don't be afraid to push back on the spec if something doesn't make sense once you're closer to the code.*
