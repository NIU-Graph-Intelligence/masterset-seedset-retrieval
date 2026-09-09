#!/usr/bin/env python3
"""Phase 5 -- Step 3: emit paper-ready LaTeX tables.

Generates the tables directly from the result JSON so that no number is ever
retyped into the manuscript by hand. Re-running the pipeline and re-running
this script keeps the paper in sync with the data.

Output: outputs/phase5_analysis/tables/*.tex

    python -m src.phase5_analysis.export_tables
"""
import argparse
import sys

from ..common import config
from ..common.io_utils import ensure_dir, read_json
from ..common.seeds import load_query_sets

STRATEGY_LABEL = {"mean": "Mean", "max": "Max",
                  "soft_and": "Soft-AND", "cluster": "Cluster"}
METHOD_LABEL = {"graph": "Graph (PPR)"}


def label_for(name: str) -> str:
    if name in METHOD_LABEL:
        return METHOD_LABEL[name]
    if name.startswith("semantic_"):
        return f"Semantic ({STRATEGY_LABEL.get(name[9:], name[9:])})"
    if name.startswith("fused_"):
        return f"Fused ({STRATEGY_LABEL.get(name[6:], name[6:])}+PPR)"
    return name.replace("_", " ").title()


def escape(text: str) -> str:
    for char in ["&", "%", "$", "#", "_", "{", "}"]:
        text = text.replace(char, "\\" + char)
    return text


def short(topic: str, width: int = 22) -> str:
    return escape(topic if len(topic) <= width else topic[:width - 1] + ".")


def table_strategy_overlap(overlap) -> str:
    pairs = ["mean|max", "mean|soft_and", "max|soft_and",
             "cluster|mean", "cluster|max"]
    headers = ["Mean/Max", "Mean/Soft", "Max/Soft", "Clust/Mean", "Clust/Max"]

    lines = [
        r"\begin{table}[t]",
        r"\caption{Jaccard similarity between aggregation strategies (top "
        f"{overlap['top_k']}). Higher means the strategies agree more; "
        r"``All'' counts papers every strategy returned.}",
        r"\label{tab:aggregation}", r"\centering", r"\footnotesize",
        r"\begin{tabular}{@{}l" + "c" * (len(headers) + 2) + r"@{}}", r"\toprule",
        r"\textbf{Query} & \textbf{$|S|$} & "
        + " & ".join(rf"\textbf{{{h}}}" for h in headers)
        + r" & \textbf{All} \\", r"\midrule",
    ]
    for q in overlap["queries"]:
        cells = []
        for pair in pairs:
            a, b = pair.split("|")
            value = q["strategy_jaccard"].get(f"{a}|{b}",
                                              q["strategy_jaccard"].get(f"{b}|{a}"))
            cells.append("--" if value is None else f"{value:.3f}")
        lines.append(f"{short(q['topic'])} & {q['n_seeds']} & "
                     + " & ".join(cells)
                     + f" & {q['n_in_all_strategies']} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_signal_overlap(overlap, strategy="mean") -> str:
    lines = [
        r"\begin{table}[t]",
        r"\caption{Overlap between the semantic ("
        + STRATEGY_LABEL.get(strategy, strategy)
        + r") and graph signals (Jaccard, top "
        + f"{overlap['top_k']}"
        + r"). ``Graph only'' counts papers PPR returned that the semantic "
          r"ranker did not.}",
        r"\label{tab:overlap}", r"\centering", r"\footnotesize",
        r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
        r"\textbf{Query} & \textbf{Sem/Graph} & \textbf{Sem/Fused} & "
        r"\textbf{Graph/Fused} & \textbf{Graph only} \\", r"\midrule",
    ]
    for q in overlap["queries"]:
        s = q.get("signal_jaccard", {}).get(strategy)
        if not s:
            lines.append(f"{short(q['topic'])} & -- & -- & -- & -- " + r"\\")
            continue
        lines.append(
            f"{short(q['topic'])} & {s['semantic_vs_graph']:.3f} & "
            f"{s['semantic_vs_fused']:.3f} & {s['graph_vs_fused']:.3f} & "
            f"{s['graph_only_papers']} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_judge(metrics, methods=None) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\caption{LLM judge evaluation. Avg@$k$ is the mean relevance score "
        r"(1--5) over the top $k$; P@10 is the fraction of the top 10 scored "
        r"$\geq 4$. Best value per query in bold.}",
        r"\label{tab:judge}", r"\centering", r"\footnotesize",
        r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
        r"\textbf{Query} & \textbf{Method} & \textbf{Avg@10} & "
        r"\textbf{Avg@50} & \textbf{P@10} \\", r"\midrule",
    ]
    for qi, q in enumerate(metrics["queries"]):
        names = [m for m in (methods or q["methods"]) if m in q["methods"]]
        if not names:
            continue
        best = {key: max(q["methods"][n][key] for n in names)
                for key in ["avg_relevance_at_10", "avg_relevance_at_50",
                            "precision_at_10"]}
        lines.append(r"\multirow{" + str(len(names)) + r"}{*}{"
                     + short(q["topic"], 16) + "}")
        for i, name in enumerate(names):
            m = q["methods"][name]

            def cell(key, fmt="{:.2f}"):
                value = m[key]
                text = fmt.format(value)
                return rf"\textbf{{{text}}}" if value == best[key] else text

            prefix = " " if i == 0 else ""
            lines.append(f"{prefix} & {label_for(name)} & "
                         f"{cell('avg_relevance_at_10')} & "
                         f"{cell('avg_relevance_at_50')} & "
                         f"{cell('precision_at_10')} " + r"\\")
        if qi < len(metrics["queries"]) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_distribution(metrics, methods=None) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\caption{Relevance score distribution over the retrieved papers. "
        r"1 = not relevant, 5 = directly on-topic.}",
        r"\label{tab:distribution}", r"\centering", r"\footnotesize",
        r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
        r"\textbf{Query} & \textbf{Method} & \textbf{1} & \textbf{2} & "
        r"\textbf{3} & \textbf{4} & \textbf{5} \\", r"\midrule",
    ]
    for qi, q in enumerate(metrics["queries"]):
        names = [m for m in (methods or q["methods"]) if m in q["methods"]]
        if not names:
            continue
        lines.append(r"\multirow{" + str(len(names)) + r"}{*}{"
                     + short(q["topic"], 16) + "}")
        for i, name in enumerate(names):
            dist = q["methods"][name]["distribution"]
            cells = " & ".join(str(dist.get(str(s), 0)) for s in range(1, 6))
            prefix = " " if i == 0 else ""
            lines.append(f"{prefix} & {label_for(name)} & {cells} " + r"\\")
        if qi < len(metrics["queries"]) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+",
                        default=["semantic_mean", "graph", "fused_mean"],
                        help="Methods (in order) for the judge tables.")
    parser.add_argument("--strategy", default="mean",
                        help="Semantic strategy for the signal-overlap table.")
    args = parser.parse_args()

    tables_dir = ensure_dir(config.PHASE5_DIR / "tables")
    written = []

    overlap_path = config.PHASE5_DIR / "overlap.json"
    if overlap_path.exists():
        overlap = read_json(overlap_path)
        for name, tex in [
            ("table_aggregation.tex", table_strategy_overlap(overlap)),
            ("table_signal_overlap.tex", table_signal_overlap(overlap, args.strategy)),
        ]:
            (tables_dir / name).write_text(tex + "\n", encoding="utf-8")
            written.append(name)
    else:
        print("[skip] overlap.json not found -- run compute_overlap.py")

    metrics_path = config.PHASE5_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = read_json(metrics_path)
        for name, tex in [
            ("table_judge.tex", table_judge(metrics, args.methods)),
            ("table_distribution.tex", table_distribution(metrics, args.methods)),
        ]:
            (tables_dir / name).write_text(tex + "\n", encoding="utf-8")
            written.append(name)
    else:
        print("[skip] metrics.json not found -- run compute_metrics.py")

    # Corpus / graph numbers that appear in the prose.
    facts = {}
    if config.GRAPH_STATS_PATH.exists():
        facts.update(read_json(config.GRAPH_STATS_PATH))
    if config.SEED_VALIDATION_PATH.exists():
        v = read_json(config.SEED_VALIDATION_PATH)
        facts["pool_size"] = v.get("pool_size")
        facts["n_query_sets"] = len(v.get("query_sets", []))
        facts["n_seed_papers"] = sum(q["n_seeds"] for q in v.get("query_sets", []))
    if facts:
        lines = ["% Auto-generated corpus facts. \\input{} this file, then use",
                 "% the macros below instead of typing numbers into the prose."]
        for key, value in sorted(facts.items()):
            if isinstance(value, (int, float)):
                macro = "".join(w.capitalize() for w in key.split("_"))
                formatted = f"{value:,}" if isinstance(value, int) else f"{value}"
                lines.append(rf"\newcommand{{\fact{macro}}}{{{formatted}}}")
        (tables_dir / "corpus_facts.tex").write_text("\n".join(lines) + "\n",
                                                     encoding="utf-8")
        written.append("corpus_facts.tex")

    for name in written:
        print(f"  wrote {tables_dir / name}")
    print(f"\n{len(written)} file(s) in {tables_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
