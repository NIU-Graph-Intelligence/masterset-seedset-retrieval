#!/usr/bin/env python3
"""Pre-flight the LLM judge before committing to a long run.

Catches the failures that otherwise surface only after a job has been detached:
server not reachable, model not pulled, replies that will not parse, and a
throughput estimate so the wall-clock cost is known in advance.

    python scripts/check_judge.py
    python scripts/check_judge.py --n 10        # more timing samples
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config                       # noqa: E402
from src.phase4_judge.judge_client import parse_reply  # noqa: E402

SAMPLES = [
    ("Neural Algorithmic Reasoning",
     ["Discrete Neural Algorithmic Reasoning",
      "Deep Equilibrium Algorithmic Reasoning"],
     "The CLRS Algorithmic Reasoning Benchmark",
     "We introduce a benchmark for neural algorithmic reasoning built on thirty "
     "classical algorithms from CLRS, with graph-structured inputs and "
     "step-by-step execution traces."),
    ("Neural Algorithmic Reasoning",
     ["Discrete Neural Algorithmic Reasoning",
      "Deep Equilibrium Algorithmic Reasoning"],
     "Decoupled Weight Decay Regularization",
     "We propose AdamW, which decouples weight decay from the gradient update "
     "in the Adam optimizer and improves generalization."),
]


def model_names(client):
    """Installed model names, across ollama client versions."""
    listing = client.list()
    models = listing.get("models", []) if isinstance(listing, dict) else \
        getattr(listing, "models", [])
    names = []
    for m in models:
        name = (m.get("model") or m.get("name")) if isinstance(m, dict) else \
            (getattr(m, "model", None) or getattr(m, "name", None))
        if name:
            names.append(name)
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=4, help="Timing samples.")
    parser.add_argument("--estimate", type=int,
                        help="Unique judgment count from run_judge --dry-run.")
    args = parser.parse_args()

    print(f"Model : {config.OLLAMA_MODEL}")
    print(f"Host  : {config.OLLAMA_HOST}\n")

    try:
        import ollama
    except ImportError:
        print("FAIL  ollama python package missing.  pip install ollama")
        return 1

    client = ollama.Client(host=config.OLLAMA_HOST)

    # 1. server reachable
    try:
        installed = model_names(client)
    except Exception as exc:
        print(f"FAIL  cannot reach the Ollama server at {config.OLLAMA_HOST}")
        print(f"      {type(exc).__name__}: {exc}\n")
        print("      Start it with:   ollama serve")
        print("      Or, if installed as a service:   sudo systemctl start ollama")
        return 1
    print(f"OK    server reachable, {len(installed)} model(s) installed")

    # 2. model present
    wanted = config.OLLAMA_MODEL
    if wanted not in installed and wanted.split(":")[0] not in \
            [n.split(":")[0] for n in installed]:
        print(f"FAIL  model {wanted!r} is not installed.\n")
        print(f"      Pull it:   ollama pull {wanted}")
        if installed:
            print("\n      Installed models:")
            for name in installed:
                print(f"        {name}")
            print("\n      To use one of those instead, set OLLAMA_MODEL in .env.")
        return 1
    if wanted not in installed:
        print(f"WARN  exact tag {wanted!r} not found, but a same-family model is "
              f"installed:\n      {installed}")
        print(f"      Either `ollama pull {wanted}` or point OLLAMA_MODEL at an "
              f"installed tag.")
        return 1
    print(f"OK    model {wanted!r} installed")

    # 3. real judgments, timed
    from src.phase4_judge.judge_client import OllamaJudge
    judge = OllamaJudge()

    print(f"\nRunning {args.n} test judgment(s)...\n")
    times, scores = [], []
    for i in range(args.n):
        topic, seeds, title, abstract = SAMPLES[i % len(SAMPLES)]
        started = time.time()
        try:
            judgment = judge.judge(topic, seeds, title, abstract)
        except Exception as exc:
            print(f"FAIL  judgment raised {type(exc).__name__}: {exc}")
            return 1
        elapsed = time.time() - started
        times.append(elapsed)
        scores.append(judgment.score)
        status = "OK  " if judgment.score > 0 else "PARSE-FAIL"
        print(f"  {status} {elapsed:5.2f}s  score={judgment.score}  {title[:44]}")
        if judgment.score == 0:
            print(f"        raw reply: {judgment.raw[:120]!r}")

    if all(s == 0 for s in scores):
        print("\nFAIL  no reply could be parsed. The model is ignoring the "
              "output format.\n      Try a different model or adjust "
              "JUDGE_PROMPT in src/phase4_judge/judge_client.py")
        return 1

    # A relevant paper should outscore an unrelated one. If it does not, the
    # model is not really reading the prompt and the metrics will be noise.
    if len(scores) >= 2 and scores[0] <= scores[1]:
        print(f"\nWARN  the on-topic sample scored {scores[0]} and the unrelated "
              f"sample {scores[1]}.\n      Expected the first to be clearly "
              f"higher. Judgments may be unreliable.")

    mean = sum(times) / len(times)
    print(f"\nOK    mean {mean:.2f}s per judgment")

    n = args.estimate
    if n is None:
        print("\nPass --estimate <N> (the 'unique judgments' number from "
              "run_judge --dry-run) for a wall-clock estimate.")
    else:
        total = mean * n
        h, m = divmod(int(total // 60), 60)
        print(f"      {n:,} unique judgments  ->  ~{h}h {m}m")

    print("\nPre-flight passed. Safe to start the full run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
