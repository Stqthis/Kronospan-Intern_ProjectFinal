

from __future__ import annotations

import math
import re
from collections import Counter

# Unicode-aware: [a-z0-9]+ silently dropped every non-Latin token, making
# Greek text invisible to BM25 and to all keyword heuristics.
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").casefold())


def chunk_embed_text(chunk: dict) -> str:

    tag = f"{chunk.get('source', '')} {chunk.get('location', '')}".strip()
    return f"[{tag}] {chunk['text']}" if tag else chunk["text"]


class BM25Index:
    """Minimal BM25 (Okapi) over the chunk corpus, keyed by chunk_id."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunk_ids: list[int] = []
        self.doc_tf: list[Counter] = []
        self.doc_len: list[int] = []
        self.avgdl: float = 0.0
        self.idf: dict[str, float] = {}

    @property
    def size(self) -> int:
        return len(self.doc_tf)

    def build(self, chunks: list[dict]) -> None:
        self.chunk_ids = []
        self.doc_tf = []
        self.doc_len = []
        df: Counter = Counter()
        for ch in chunks:
            toks = tokenize(chunk_embed_text(ch))
            tf = Counter(toks)
            self.doc_tf.append(tf)
            self.doc_len.append(len(toks))
            self.chunk_ids.append(ch["chunk_id"])
            for term in tf:
                df[term] += 1
        n = len(chunks)
        self.avgdl = (sum(self.doc_len) / n) if n else 0.0
        # BM25 idf with +1 smoothing so weights stay positive.
        self.idf = {t: math.log(1 + (n - dfi + 0.5) / (dfi + 0.5)) for t, dfi in df.items()}

    def rank(self, query: str, n: int) -> list[int]:
        if not self.doc_tf:
            return []
        q_terms = [t for t in tokenize(query) if t in self.idf]
        if not q_terms:
            return []
        scores: list[tuple[float, int]] = []
        avgdl = self.avgdl or 1.0
        for i, tf in enumerate(self.doc_tf):
            dl = self.doc_len[i] or 1
            s = 0.0
            for t in q_terms:
                f = tf.get(t, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / avgdl)
                s += self.idf[t] * (f * (self.k1 + 1)) / denom
            if s > 0:
                scores.append((s, self.chunk_ids[i]))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [cid for _, cid in scores[:n]]


def rrf_fuse(ranked_lists: list[tuple[list[int], float]], k: int = 60) -> list[tuple[float, int]]:
    """Weighted Reciprocal Rank Fusion.

    ``ranked_lists`` is a list of (ids_best_to_worst, weight). Returns
    (fused_score, id) sorted descending. RRF needs no score normalisation,
    which is why it fuses heterogeneous scorers (cosine vs BM25) robustly.
    """
    fused: dict[int, float] = {}
    for ids, weight in ranked_lists:
        for rank, cid in enumerate(ids):
            fused[cid] = fused.get(cid, 0.0) + weight * (1.0 / (k + rank + 1))
    out = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [(score, cid) for cid, score in out]