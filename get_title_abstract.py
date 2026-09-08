import json
import os
from pathlib import Path

import pandas as pd

DATAFRAME_PATH = Path("/home/ratul/mustcite/data/train_eval_set/v7.0/all_papers_with_refs_and_labels.parquet")

SEEDS_JSON_PATH = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "seeds.json"))

TOPIC = "Mixed Set 2: Graph-Compatible"  # user inputs every time

PAPER_IDS = [
    "aa57623e-39e6-53d0-a173-5b9971aabec1",
    "6c981e85-a9eb-52f6-a5d3-309d3f0b83e7",
    "5f42feb8-cb1b-57aa-bf28-e4c237e77b2c",
    "bc2e7938-a748-51a7-8c2d-611307604176",
    "aab76fc3-3dc1-5895-8e75-fe5de66a40f0"
]


def to_str(value):
    """Safely cast any parquet cell (numpy types, NaN, None) to a plain string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def load_seeds(path):
    """Load the existing seeds file, or start a fresh structure if there is none."""
    if not path.exists() or path.stat().st_size == 0:
        return {"SEED_SETS": []}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or not isinstance(data.get("SEED_SETS"), list):
        raise ValueError(f"{path} does not look like a seeds file with a 'SEED_SETS' list.")

    return data


def build_papers(df, paper_ids):
    papers = []
    missing = []

    for paper_id in paper_ids:
        row = df[df["paper_id"] == paper_id]

        if row.empty:
            missing.append(paper_id)
            continue

        row = row.iloc[0]
        papers.append(
            {
                "paper_id": to_str(row["paper_id"]),
                "title": to_str(row["title"]),
                "abstract": to_str(row["abstract"]),
                "year": to_str(row["year"]),
                "venue": to_str(row["venue"]),
            }
        )

    return papers, missing


def main():
    df = pd.read_parquet(DATAFRAME_PATH)

    papers, missing = build_papers(df, PAPER_IDS)

    for paper_id in missing:
        print(f"Warning: paper not found in the candidate pool -> {paper_id}")

    if not papers:
        print("No papers found. seeds.json left unchanged.")
        return

    data = load_seeds(SEEDS_JSON_PATH)

    # Replace the topic if it already exists, otherwise append it.
    for seed_set in data["SEED_SETS"]:
        if seed_set.get("topic") == TOPIC:
            seed_set["papers"] = papers
            action = "replaced"
            break
    else:
        data["SEED_SETS"].append({"topic": TOPIC, "papers": papers})
        action = "appended"

    # Write atomically via a temp file so a crash can never truncate seeds.json.
    tmp_path = SEEDS_JSON_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp_path, SEEDS_JSON_PATH)

    print(f"{action.capitalize()} topic '{TOPIC}' with {len(papers)} paper(s) in {SEEDS_JSON_PATH}")


main()