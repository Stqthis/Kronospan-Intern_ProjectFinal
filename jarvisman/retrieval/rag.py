

from __future__ import annotations

import os
import pickle
from typing import Callable, Optional

import numpy as np

from jarvisman import config as cfg
from jarvisman.retrieval.embedding_cache import EmbeddingCache
from jarvisman.ingest.ingestion import build_chunks, ingest_file
from jarvisman.llm.ollama_client import OllamaClient
from jarvisman.retrieval.retrieval import BM25Index, chunk_embed_text, rrf_fuse
from jarvisman.retrieval.vector_store import VectorStore
from jarvisman.ingest.column_types import profile_and_apply, save_profile, load_profile
from jarvisman.semantics.semantic_model import build_semantic_model, save_semantic_model, load_semantic_model

GROUNDED_SYSTEM = (
    "You are a careful assistant answering questions about the user's documents. "
    "Use ONLY the information in the provided context excerpts. "
    "If the answer is not contained in the context, say clearly that the "
    "documents do not contain that information -- do not guess. "
    "Where useful, cite the source like [file p.3]. "
    "Be concise and factual."
)


# --------------------------------------------------------------------------- #
# Table persistence (separate from the vector index)                          #
# --------------------------------------------------------------------------- #
def _tables_path(directory: str) -> str:
    return os.path.join(directory, "tables.pkl")


def save_tables(directory: str, dataframes: dict) -> None:
    """Best-effort pickle of the loaded tables next to the index."""
    try:
        os.makedirs(directory, exist_ok=True)
        with open(_tables_path(directory), "wb") as fh:
            pickle.dump(dataframes, fh)
    except Exception:
        pass  # never fail a build over persistence


def load_tables(directory: str) -> dict:
    path = _tables_path(directory)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


class RAGPipeline:
    table_cards: dict = {}
    index_version: str = "0"

    def __init__(
        self,
        ollama: OllamaClient,
        vector_store: VectorStore,
        chat_model: str,
        embed_model: str,
    ) -> None:
        self.ollama = ollama
        self.vector_store = vector_store
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.bm25 = BM25Index(cfg.BM25_K1, cfg.BM25_B)
        self._embed_cache: Optional[EmbeddingCache] = None
        self._query_vec_cache: dict[str, list[float]] = {}
        self.table_profile: dict = {}   # {table: {summary, columns:{col:{type,format,meaning}}}}
        self.semantic_model = None      # statistical SemanticModel built at index time

    @property
    def embed_cache(self) -> EmbeddingCache:
        if self._embed_cache is None:
            self._embed_cache = EmbeddingCache(cfg.EMBED_CACHE_PATH)
        return self._embed_cache

    # ------------------------------------------------------------------ #
    # Embedding (cached + batched)                                       #
    # ------------------------------------------------------------------ #
    def _embed_many(
        self, texts: list[str], report: Optional[Callable[[str], None]] = None
    ) -> list[list[float]]:
        cache = self.embed_cache
        vectors: list[Optional[list[float]]] = [None] * len(texts)
        misses: list[int] = []
        for i, t in enumerate(texts):
            cached = cache.get(self.embed_model, t)
            if cached is None:
                misses.append(i)
            else:
                vectors[i] = cached

        total_miss = len(misses)
        cached_n = len(texts) - total_miss
        done = 0
        for start in range(0, total_miss, cfg.EMBED_BATCH_SIZE):
            idx_batch = misses[start : start + cfg.EMBED_BATCH_SIZE]
            batch_texts = [texts[i] for i in idx_batch]
            embs = self.ollama.embed_batch(batch_texts, self.embed_model)
            for i, e in zip(idx_batch, embs):
                vectors[i] = e
                cache.put(self.embed_model, texts[i], e)
            done += len(idx_batch)
            if report:
                report(f"Embedding {done}/{total_miss} new chunk(s) ({cached_n} cached) ...")
        cache.save()
        # Defensive: any leftover None (shouldn't happen) embedded individually.
        for i, v in enumerate(vectors):
            if v is None:
                vectors[i] = self.ollama.embed(texts[i], self.embed_model)
        return vectors  # type: ignore[return-value]

    _QUERY_CACHE_MAX = 512   # bound the per-session cache: a long session of
                             # unique questions must not grow memory forever

    def _embed_query(self, query: str) -> list[float]:
        v = self._query_vec_cache.get(query)
        if v is None:
            v = self.ollama.embed(query, self.embed_model)
            if len(self._query_vec_cache) >= self._QUERY_CACHE_MAX:
                self._query_vec_cache.clear()
            self._query_vec_cache[query] = v
        return v

    # ------------------------------------------------------------------ #
    # Indexing (PDF text -> FAISS + BM25; Excel -> tables)               #
    # ------------------------------------------------------------------ #
    def index_documents(
        self, paths: list[str], progress_callback: Optional[Callable[[str], None]] = None
    ):
        def report(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)

        pdf_records: list[dict] = []
        dataframes: dict = {}
        pdf_files = 0
        excel_files = 0
        for path in paths:
            report(f"Reading {os.path.basename(path)} ...")
            recs, dfs = ingest_file(path)
            if dfs:
                excel_files += 1
                dataframes.update(dfs)
            if recs:
                pdf_files += 1
                pdf_records.extend(recs)

        chunks = build_chunks(pdf_records, cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)
        stats = {
            "files": len(paths),
            "pdf_files": pdf_files,
            "excel_files": excel_files,
            "pdf_chunks": len(chunks),
            "tables": len(dataframes),
        }
        self._query_vec_cache.clear()

        if chunks:
            embed_texts = [chunk_embed_text(ch) for ch in chunks]
            vectors = self._embed_many(embed_texts, report)
            report("Building vector index ...")
            embeddings = np.array(vectors, dtype="float32")
            self.vector_store.build(embeddings, chunks, self.embed_model)
            self.bm25.build(chunks)
            try:
                self.vector_store.save(cfg.INDEX_DIR)
            except Exception:
                pass  # persistence is best-effort
        else:
            # Pure-table workload: no prose to retrieve, so no index needed.
            self.vector_store.reset()
            self.bm25 = BM25Index(cfg.BM25_K1, cfg.BM25_B)

        if dataframes:
            # The model reads each table's columns + sample rows and decides
            # every column's type, date format, and meaning; the code applies
            # the types and keeps the meanings. Done once here, so the saved
            # tables are typed and the meanings are available at query time.
            # Enrichment (table summary + per-column meaning) is an
            # UNDERSTANDING task -> use the understanding model when one is
            # configured (llama3:70b writes better descriptions than the
            # code model). Falls back to the chat model on single-model setups.
            _enrich_model = cfg.model_for("understand", self.chat_model)
            dataframes, self.table_profile = profile_and_apply(
                self.ollama, _enrich_model, dataframes, report
            )
            report("Saving tables ...")
            save_tables(cfg.INDEX_DIR, dataframes)
            save_profile(cfg.INDEX_DIR, self.table_profile)
            # Statistical semantic model: full-column stats, roles, summary
            # rows, relationships -- code-computed; meanings come from the
            # LLM profile above, so this adds NO extra LLM calls.
            report("Building semantic model ...")
            self.semantic_model = build_semantic_model(
                dataframes, meanings=self.table_profile
            )
            save_semantic_model(cfg.INDEX_DIR, self.semantic_model)
            # Table cards: one LLM call per table, regenerated only when a
            # table's schema hash changes. Optional and best-effort.
            if cfg.TABLE_CARDS:
                try:
                    from jarvisman.semantics import table_cards as tc
                    report("Understanding tables ...")
                    existing = tc.load_cards(cfg.INDEX_DIR)
                    
                    tc.save_cards(cfg.INDEX_DIR, self.table_cards)
                except Exception:
                    self.table_cards = {}
            self.index_version = str(int(self.index_version) + 1) \
                if str(getattr(self, "index_version", "0")).isdigit() else "1"
        else:
            self.table_profile = {}
            self.semantic_model = None

        return stats, dataframes

    def load_persisted(self, directory: str = cfg.INDEX_DIR) -> dict:
        """Restore the FAISS index (+ rebuild BM25) AND the tables.

        Tolerant of either piece being absent: a PDF-only index has no
        ``tables.pkl``; an Excel-only build has no FAISS index. BM25 is rebuilt
        from the restored chunks (cheap), so it is not persisted separately.
        """
        tables = load_tables(directory)
        self.table_profile = load_profile(directory)
        self.semantic_model = load_semantic_model(directory)
        try:
            from jarvisman.semantics import table_cards as tc
            self.table_cards = tc.load_cards(directory)
        except Exception:
            self.table_cards = {}
        if self.semantic_model is None and tables:
            # older index without a saved model: rebuild statistically
            self.semantic_model = build_semantic_model(
                tables, meanings=self.table_profile
            )
        self._query_vec_cache.clear()
        try:
            self.vector_store.load(directory)
            self.bm25.build(self.vector_store.chunks)
        except FileNotFoundError:
            self.vector_store.reset()
            self.bm25 = BM25Index(cfg.BM25_K1, cfg.BM25_B)
            if not tables:
                raise  # nothing at all was saved
        return tables

    # ------------------------------------------------------------------ #
    # Hybrid retrieval (dense + BM25, fused with RRF)                    #
    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, k: int) -> list[tuple[float, dict]]:
        if self.vector_store.count == 0:
            return []

        q_vec = np.array([self._embed_query(query)], dtype="float32")
        dense_hits = self.vector_store.search(q_vec, cfg.FETCH_K)  # [(score, chunk)]
        dense_ids = [c["chunk_id"] for _, c in dense_hits]
        sparse_ids = self.bm25.rank(query, cfg.FETCH_K)

        # Resolve chunk_id -> chunk for everything we might return.
        id_to_chunk: dict[int, dict] = {c["chunk_id"]: c for _, c in dense_hits}
        for cid in sparse_ids:
            if cid not in id_to_chunk and 0 <= cid < len(self.vector_store.chunks):
                id_to_chunk[cid] = self.vector_store.chunks[cid]

        fused = rrf_fuse(
            [
                (dense_ids, cfg.HYBRID_DENSE_WEIGHT),
                (sparse_ids, cfg.HYBRID_SPARSE_WEIGHT),
            ],
            k=cfg.RRF_K,
        )
        results: list[tuple[float, dict]] = []
        for score, cid in fused[:k]:
            ch = id_to_chunk.get(cid)
            if ch is not None:
                results.append((score, ch))
        return results

    def _build_context(self, hits: list[tuple[float, dict]]):
        blocks: list[str] = []
        sources: list[dict] = []
        used = 0
        for score, chunk in hits:
            tag = f"{chunk['source']} {chunk['location']}"
            block = f"[{tag}]\n{chunk['text']}"
            if used + len(block) > cfg.MAX_CONTEXT_CHARS and blocks:
                break
            blocks.append(block)
            used += len(block)
            sources.append({"source": chunk["source"], "location": chunk["location"], "score": round(score, 4)})
        return "\n\n---\n\n".join(blocks), sources

    # ------------------------------------------------------------------ #
    # Answering (PDF prose)                                              #
    # ------------------------------------------------------------------ #
    def answer(
        self,
        query: str,
        k: Optional[int] = None,
        token_callback: Optional[Callable[[str], None]] = None,
    ):
        hits = self.retrieve(query, k or cfg.TOP_K)
        if not hits:
            msg = "The indexed documents do not contain information relevant to that question."
            if token_callback:
                token_callback(msg)
            return msg, []

        context, sources = self._build_context(hits)
        messages = [
            {"role": "system", "content": GROUNDED_SYSTEM},
            {"role": "user", "content": f"Context excerpts:\n{context}\n\nQuestion: {query}"},
        ]
        text = self.ollama.chat(
            self.chat_model,
            messages,
            options={"temperature": 0.2},
            stream=bool(token_callback),
            on_token=token_callback,
        )
        return text, sources