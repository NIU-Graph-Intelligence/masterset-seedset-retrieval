#!/usr/bin/env bash
# Full pipeline, phase by phase. Run from the repository root:
#     bash scripts/run_all.sh
#
# Every step is also runnable on its own; this file just documents the order
# and stops at the first failure.
set -euo pipefail

echo "=== Phase 2a: build citation graph ==========================="
python -m src.phase2_graph.build_citation_graph

echo "=== Phase 1a: validate seeds (BLOCKING) ======================"
python -m src.phase1_semantic.validate_seeds --strict

echo "=== Phase 1b: semantic retrieval ============================="
python -m src.phase1_semantic.run_semantic_retrieval

echo "=== Phase 2b: PPR retrieval =================================="
python -m src.phase2_graph.run_ppr_retrieval --damping 0.85 0.90 0.95

echo "=== Phase 3: RRF fusion ======================================"
python -m src.phase3_fusion.run_fusion --rrf-k 60 20 100

echo "=== Phase 4: LLM judge ======================================="
python -m src.phase4_judge.run_judge

echo "=== Phase 4b: judge consistency =============================="
python -m src.phase4_judge.run_consistency --n-papers 20 --n-runs 3

echo "=== Phase 5: analysis + LaTeX tables ========================="
python -m src.phase5_analysis.compute_overlap
python -m src.phase5_analysis.compute_metrics
python -m src.phase5_analysis.export_tables

echo
echo "Pipeline complete. Tables in outputs/phase5_analysis/tables/"
