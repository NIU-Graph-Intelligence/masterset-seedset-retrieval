"""LLM judge: prompt, call, parse, cache.

Isolated from the run scripts so the prompt and the parser can be tested
without invoking the model, and so the consistency check and the main run
share exactly one implementation.
"""
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..common import config
from ..common.io_utils import append_jsonl, read_jsonl

PROMPT_VERSION = "v2"

JUDGE_PROMPT = """You are an expert in AI/ML research literature. You will be given a research topic defined by a set of seed papers, and one candidate paper. Judge how relevant the candidate is to the topic.

Research topic: {topic}

Seed papers defining the topic:
{seed_titles}

Candidate paper:
Title: {title}
Abstract: {abstract}

Rate the candidate's relevance on this scale:
5 = Directly on-topic; could itself be a seed paper
4 = Closely related; addresses the same research question or method
3 = Somewhat related; shares methods or problem domain
2 = Loosely related; tangential connection
1 = Not relevant

Answer with the digit first, then a period, then one sentence of justification.
Begin your reply with the digit and nothing else.

Example reply:
4. This paper studies the same expressivity question using a different proof technique.
"""

# The digit must open the reply. Searching the whole reply for any digit 1-5
# picks up incidental numbers ("relevant to 3 of the seeds", "GPT-4") and
# silently mis-scores the paper.
_LEADING_SCORE = re.compile(r"^\s*(?:score\s*[:=]?\s*)?([1-5])\b", re.IGNORECASE)
_FALLBACK_SCORE = re.compile(r"(?:^|\n)\s*(?:score|rating)\s*[:=]\s*([1-5])\b", re.IGNORECASE)


@dataclass
class Judgment:
    score: int              # 1-5, or 0 when the reply could not be parsed
    justification: str
    raw: str
    cached: bool = False


def parse_reply(reply: str) -> Tuple[int, str]:
    reply = (reply or "").strip()
    if not reply:
        return 0, ""

    match = _LEADING_SCORE.match(reply) or _FALLBACK_SCORE.search(reply)
    if not match:
        return 0, reply

    score = int(match.group(1))
    justification = reply[match.end():].lstrip(" .:-—").strip()
    return score, justification


def cache_key(topic: str, seed_ids: Sequence[str], paper_id: str) -> str:
    """Identifies one judgment.

    Includes the seed set and prompt version, so changing either invalidates
    the cache rather than silently reusing judgments made under a different
    definition of the topic.
    """
    digest = hashlib.sha256()
    digest.update(PROMPT_VERSION.encode())
    digest.update(config.OLLAMA_MODEL.encode())
    digest.update(topic.encode())
    digest.update("|".join(sorted(seed_ids)).encode())
    digest.update(paper_id.encode())
    return digest.hexdigest()[:24]


class JudgeCache:
    """Append-only JSONL cache of judgments.

    The same paper appears in the semantic, graph and fused lists for a query,
    so without caching each one is judged three to five times. Caching also
    makes a crashed run resumable at no cost.
    """

    def __init__(self, path: Path = None, enabled: bool = True):
        self.path = path or config.JUDGE_CACHE_PATH
        self.enabled = enabled
        self._store: Dict[str, dict] = {}
        if enabled:
            for record in read_jsonl(self.path):
                self._store[record["key"]] = record

    def __len__(self) -> int:
        return len(self._store)

    def get(self, key: str) -> Optional[Judgment]:
        if not self.enabled:
            return None
        record = self._store.get(key)
        if record is None:
            return None
        return Judgment(score=record["score"], justification=record["justification"],
                        raw=record.get("raw", ""), cached=True)

    def put(self, key: str, judgment: Judgment, meta: dict) -> None:
        if not self.enabled:
            return
        record = {"key": key, "score": judgment.score,
                  "justification": judgment.justification,
                  "raw": judgment.raw, **meta}
        self._store[key] = record
        append_jsonl(self.path, record)


class OllamaJudge:
    """Thin wrapper over the local Ollama chat endpoint.

    temperature=0 and a fixed seed make judgments reproducible, which is what
    lets the consistency check measure genuine model variance rather than
    sampling noise.
    """

    def __init__(self, model: str = None, host: str = None,
                 temperature: float = 0.0, seed: int = 42):
        import ollama
        self.model = model or config.OLLAMA_MODEL
        self.client = ollama.Client(host=host or config.OLLAMA_HOST)
        self.options = {"temperature": temperature, "seed": seed}

    def judge(self, topic: str, seed_titles: Sequence[str],
              title: str, abstract: str, abstract_chars: int = 1200) -> Judgment:
        prompt = JUDGE_PROMPT.format(
            topic=topic,
            seed_titles="\n".join(f"- {t}" for t in seed_titles),
            title=title or "[no title]",
            abstract=(abstract or "[no abstract available]")[:abstract_chars],
        )
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options=self.options,
        )
        raw = response["message"]["content"].strip()
        score, justification = parse_reply(raw)
        return Judgment(score=score, justification=justification, raw=raw)


def judge_many(
    judge: "OllamaJudge",
    cache: JudgeCache,
    topic: str,
    seed_ids: Sequence[str],
    seed_titles: Sequence[str],
    paper_ids: Sequence[str],
    metadata: dict,
    label: str = "",
    show_progress: bool = True,
) -> List[dict]:
    """Judge a ranked list, reusing cached judgments where possible."""
    results, n_cached, n_called = [], 0, 0

    for i, pid in enumerate(paper_ids, start=1):
        meta = metadata.get(pid, {})
        title = meta.get("title", "")
        abstract = meta.get("abstract", "")

        key = cache_key(topic, seed_ids, pid)
        judgment = cache.get(key)

        if judgment is None:
            judgment = judge.judge(topic, seed_titles, title, abstract)
            cache.put(key, judgment, {"topic": topic, "paper_id": pid,
                                      "title": title, "model": judge.model})
            n_called += 1
        else:
            n_cached += 1

        results.append({
            "rank": i,
            "paper_id": pid,
            "title": title,
            "relevance_score": judgment.score,
            "justification": judgment.justification,
            "from_cache": judgment.cached,
        })

        if show_progress and (i % 10 == 0 or i == len(paper_ids)):
            print(f"      {label} {i}/{len(paper_ids)} "
                  f"(model={n_called}, cache={n_cached})", flush=True)

    unparsed = sum(1 for r in results if r["relevance_score"] == 0)
    if unparsed:
        print(f"      WARNING: {unparsed} reply/replies could not be parsed "
              f"and are excluded from averages")
    return results
