# MasterSet Citation Retrieval — Demo

End-to-end pipeline: seed papers in → ranked + scored results out.

## Usage

```bash
python src/demo/run_pipeline.py --seeds data/seeds_nar.json --top-k 20
```

## Arguments

| Flag | Default | Description |
|---|---|---|
| `--seeds` | required | Path to seed papers JSON file |
| `--strategy` | mean | Aggregation strategy: mean, max, soft-and, cluster |
| `--no-graph` | off | Skip graph retrieval, use semantic only |
| `--no-judge` | off | Skip LLM scoring, faster results |
| `--top-k` | 20 | Number of results to return |
| `--rerank` | off | Rerank final list by relevance score |

## Seed File Format

```json
{
    "topic": "Your Topic Name",
    "papers": [
        {
            "title": "Paper Title Here",
            "abstract": "Paper abstract here."
        }
    ]
}
```

## Example Output

Rank Title Ret. Rel.

1 Open-Book Neural Algorithmic Reasoning 0.0173 5

2 The CLRS Algorithmic Reasoning Benchmark 0.0286 4

...
- - - 


Results are saved to `output/demo_results_<seed_file_name>.json`.