#!/usr/bin/env python3
"""Standing verification gate: assert no seed paper appears in any result list.

A seed scores 1.0 against itself, so a single leak puts it at rank 1 and
inflates every metric in the paper. Run after any pipeline change.

    python scripts/check_seed_leakage.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import config  # noqa: E402

EXTRACTORS = {
    "phase1_semantic": lambda d: d.get("strategies", {}),
    "phase2_graph": lambda d: {"ppr": d.get("results", [])},
    "phase3_fusion": lambda d: {k: v["results"] for k, v in d.get("fusions", {}).items()},
    "phase4_judge": lambda d: {k: v["papers"] for k, v in d.get("methods", {}).items()},
}


def main() -> int:
    checked = leaks = 0
    for phase, extract in EXTRACTORS.items():
        directory = config.OUTPUT_DIR / phase
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            data = json.loads(path.read_text(encoding='utf-8'))
            if "seed_paper_ids" not in data:
                continue
            seeds = set(data["seed_paper_ids"])
            for name, records in extract(data).items():
                checked += 1
                found = seeds & {r["paper_id"] for r in records}
                if found:
                    leaks += 1
                    print(f"LEAK {phase}/{path.stem}/{name}: {sorted(found)}")

    print(f"\n{checked} ranked list(s) checked, {leaks} leak(s).")
    if leaks:
        print("FAIL -- seed papers are present in results.")
        return 1
    print("PASS -- no seed paper appears in any result list.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
