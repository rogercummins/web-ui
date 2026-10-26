"""Lexical retrieval over the Hindsight store.

Stdlib-only, dependency-free scoring so the agent works with no embedding
model and no network access: a simple term-overlap score over Latin tokens
plus CJK bigrams (Hindsight content is sometimes Chinese — see CLAUDE.md),
with a light boost for tag and title matches.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.memory.exceptions import HindsightError, NotFoundError
from src.memory.store import HindsightStore

from .exceptions import RetrievalError

_LATIN_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_CJK_RE = re.compile(r"[一-鿿]+")


def tokenize(text: str) -> List[str]:
    """Split text into lowercase Latin word tokens plus CJK bigrams.

    A pure word-boundary tokenizer misses CJK text entirely (no spaces), so
    any run of CJK characters is additionally split into overlapping
    bigrams, which gives reasonable recall for short Chinese phrases without
    pulling in a real segmenter.
    """
    if not text:
        return []
    tokens = [t.lower() for t in _LATIN_TOKEN_RE.findall(text)]
    for run in _CJK_RE.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


@dataclass
class QASource:
    """One retrieved memory, scored against a query."""

    id: str
    project_id: str
    title: str
    content: str
    type: str
    tags: List[str] = field(default_factory=list)
    score: float = 0.0

    def citation_tag(self) -> str:
        return f"[{self.id}]"

    def as_prompt_block(self) -> str:
        return (f"{self.citation_tag()} {self.title}\n"
                f"type={self.type} tags={','.join(self.tags) or '-'}\n"
                f"{self.content}")


class Retriever:
    """Ranks Hindsight memories against a free-text query."""

    def __init__(self, store: HindsightStore):
        self.store = store

    def retrieve(self, query: str, project_id: Optional[str] = None,
                 top_k: int = 5) -> List[QASource]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        q_set = set(q_tokens)
        try:
            # Cast a wide net with the store's own substring search first —
            # this keeps retrieval correct even for exact-phrase matches that
            # the bag-of-tokens scorer below might under-weight — then fold
            # in every other memory in scope so scoring can still surface
            # partial token overlaps the substring search missed.
            substring_hits = {m.id: m for m in self.store.search(query, project_id=project_id)}
            project_ids = ([project_id] if project_id
                           else [p.id for p in self.store.list_projects()])
            pool: Dict[str, object] = dict(substring_hits)
            for pid in project_ids:
                try:
                    for mem in self.store.list_memories(pid):
                        pool.setdefault(mem.id, mem)
                except NotFoundError:
                    continue
        except HindsightError as exc:
            raise RetrievalError("could not read the memory store while retrieving",
                                  detail=str(exc)) from exc

        scored: List[QASource] = []
        for mem in pool.values():
            haystack_tokens = tokenize(f"{mem.title} {mem.content}")
            h_set = set(haystack_tokens)
            if not h_set:
                continue
            overlap = q_set & h_set
            if not overlap:
                continue
            # Overlap ratio against the query (recall-oriented) with a small
            # title-match and substring-hit bonus.
            score = len(overlap) / max(len(q_set), 1)
            if any(t in tokenize(mem.title) for t in overlap):
                score += 0.25
            if mem.id in substring_hits:
                score += 0.5
            scored.append(QASource(id=mem.id, project_id=mem.project_id,
                                    title=mem.title, content=mem.content,
                                    type=mem.type, tags=list(mem.tags),
                                    score=round(score, 4)))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]
