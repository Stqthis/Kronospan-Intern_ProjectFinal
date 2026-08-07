

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
from concurrent.futures import ThreadPoolExecutor, as_completed

GROUNDED_SYSTEM = (
    "You are analyzing company financial data with multiple currencies (EUR and Local Currency). "
    "When answering:\n"
    "1. Clearly identify ALL columns you're summing (list them by name)\n"
    "2. If multiple currencies exist, show totals in EACH currency separately\n"
    "3. Show the company name and date range used\n"
    "4. Show the calculation: which rows were included, which columns summed\n"
    "5. If result is 0.00, explain why (e.g., 'No funds allocated', or 'Company not found in table')\n"
    "6. Use ONLY provided context data"
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
    
def analyze_relationships(self, dataframes: dict) -> dict:
    """Analyze and report table relationships."""
    from jarvisman.semantics.relationship_detector import RelationshipDetector
    
    relationships = RelationshipDetector.find_relationships(dataframes)
    
    print("\n📊 DETECTED TABLE RELATIONSHIPS:")
    print("="*70)
    
    for rel_key, rels in relationships.items():
        print(f"\n{rel_key}:")
        for rel in rels:
            print(f"  {rel['left_column']} → {rel['right_column']}")
            print(f"    Type: {rel['type']}")
            print(f"    Strength: {rel['strength']:.2%}")
    
    print("="*70 + "\n")
    
    return relationships


class RAGPipeline:
    table_cards: dict = {}
    index_version: str = "0"

    @staticmethod
    def _content_version(dataframes: dict, chunk_count: int = 0) -> str:
        """A stable signature of WHAT is indexed: table keys, columns and
        row counts, plus the text-chunk count. Keys the persisted plan
        cache, so a plan cached for one dataset can never replay against a
        different one (the old per-session counter collided: every fresh
        session's first index was version '1'), and cached plans for the
        SAME dataset now survive a restart as intended."""
        import hashlib
        h = hashlib.sha1()
        for key in sorted(dataframes or {}):
            df = dataframes[key]
            try:
                h.update(str(key).encode("utf-8", "ignore"))
                h.update(str(df.shape).encode())
                h.update("|".join(str(c) for c in df.columns)
                         .encode("utf-8", "ignore"))
            except Exception:
                continue
        h.update(str(int(chunk_count)).encode())
        return h.hexdigest()[:12]

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
        self.query_cache = {}  # Add this

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
        batches = [misses[s:s + cfg.EMBED_BATCH_SIZE]
                   for s in range(0, total_miss, cfg.EMBED_BATCH_SIZE)]
        workers = max(1, min(getattr(cfg, "EMBED_CONCURRENCY", 4), len(batches)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(self.ollama.embed_batch,
                                [texts[i] for i in b], self.embed_model): b
                    for b in batches}
            for fut in as_completed(futs):
                idx_batch = futs[fut]
                for i, e in zip(idx_batch, fut.result()):
                    vectors[i] = e
                    cache.put(self.embed_model, texts[i], e)   # main thread only: safe
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

        # ------------------------------------------------------------------
        # INCREMENTAL UPDATE semantics ("Build / Update Index" means UPDATE):
        # previously indexed files are KEPT; a re-indexed file replaces its own
        # old chunks/tables. This is what makes the index scale to hundreds of
        # files added over many sessions -- indexing a new batch no longer
        # wipes everything indexed before. "Clear" is the explicit wipe.
        # ------------------------------------------------------------------
        new_sources = {os.path.basename(p) for p in paths}

        if chunks:
            embed_texts = [chunk_embed_text(ch) for ch in chunks]
            vectors = self._embed_many(embed_texts, report)
            report("Building vector index ...")
            embeddings = np.array(vectors, dtype="float32")
            if self.vector_store.count and \
                    self.vector_store.embed_model == self.embed_model:
                self.vector_store.remove_sources(new_sources)
                self.vector_store.append(embeddings, chunks, self.embed_model)
            else:
                self.vector_store.build(embeddings, chunks, self.embed_model)
            self.bm25.build(self.vector_store.chunks)
            try:
                self.vector_store.save(cfg.INDEX_DIR)
            except Exception:
                pass  # persistence is best-effort
        elif self.vector_store.count:
            # Excel-only batch: previously indexed PDFs stay searchable.
            self.vector_store.remove_sources(new_sources)
            self.bm25.build(self.vector_store.chunks)
        else:
            self.vector_store.reset()
            self.bm25 = BM25Index(cfg.BM25_K1, cfg.BM25_B)

        # Merge with previously persisted tables (dropping old versions of
        # any file being re-indexed) so the semantic model, cards and value
        # index are built over the FULL collection.
        try:
            prior_tables = load_tables(cfg.INDEX_DIR) or {}
        except Exception:
            prior_tables = {}
        prior_tables = {k: v for k, v in prior_tables.items()
                        if str(k).split(":", 1)[0] not in new_sources}
        try:
            prior_profile = load_profile(cfg.INDEX_DIR) or {}
        except Exception:
            prior_profile = {}

        if dataframes:
            # Check if we should skip profiling for speed
            if hasattr(cfg, 'SKIP_TABLE_PROFILING') and cfg.SKIP_TABLE_PROFILING:
                report("Skipping table profiling (fast mode)...")
                self.table_profile = {}
            else:
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
                    self.ollama, _enrich_model, dataframes, report,
                    existing=prior_profile,
                )
            # fold the new batch into the existing collection
            merged = dict(prior_tables)
            merged.update(dataframes)
            dataframes = merged
            merged_profile = {k: v for k, v in prior_profile.items()
                              if k in dataframes}
            merged_profile.update(self.table_profile or {})
            self.table_profile = merged_profile

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
            
            # ===== NEW: Detect relationships between tables =====
            # Detect relationships (can be enabled later when codegen improves)
            report("Analyzing table relationships ...")
            if hasattr(cfg, 'DETECT_RELATIONSHIPS') and cfg.DETECT_RELATIONSHIPS:
                try:
                    from jarvisman.semantics.relationship_detector import RelationshipDetector
                    self.relationships = RelationshipDetector.find_relationships(dataframes)
                    
                    if self.relationships:
                        print("\n📊 DETECTED TABLE RELATIONSHIPS:")
                        print("="*70)
                        for rel_key, rels in self.relationships.items():
                            print(f"\n{rel_key}:")
                            for rel in rels:
                                print(f"  {rel['left_column']} → {rel['right_column']}")
                                print(f"    Type: {rel['type']}")
                                print(f"    Strength: {rel['strength']:.2%}")
                        print("="*70 + "\n")
                        report(f"Found {len(self.relationships)} relationship(s)")
                    else:
                        self.relationships = {}
                except Exception as e:
                    print(f"\n⚠️ Error detecting relationships: {e}\n")
                    self.relationships = {}
            else:
                self.relationships = {}
            # ===== END: Relationship detection =====
            
            # Table cards: one LLM call per table, regenerated only when a
            # table's schema hash changes. Optional and best-effort.
            if cfg.TABLE_CARDS:
                try:
                    from jarvisman.semantics import table_cards as tc
                    report("Understanding tables ...")
                    existing = tc.load_cards(cfg.INDEX_DIR)
                    self.table_cards = tc.build_cards(
                        self.ollama, self.chat_model, self.semantic_model,
                        dataframes, existing, report)
                    tc.save_cards(cfg.INDEX_DIR, self.table_cards)
                except Exception:
                    self.table_cards = {}
            
            self.index_version = self._content_version(
                dataframes,
                len(getattr(self.vector_store, "chunks", []) or []))
            stats["tables"] = len(dataframes)
        elif prior_tables:
            # PDF-only batch on top of an existing table collection: keep it.
            dataframes = prior_tables
            self.table_profile = prior_profile
            report("Building semantic model ...")
            self.semantic_model = build_semantic_model(
                dataframes, meanings=self.table_profile)
            self.index_version = self._content_version(
                dataframes,
                len(getattr(self.vector_store, "chunks", []) or []))
        else:
            self.table_profile = {}
            self.semantic_model = None
            self.relationships = {}
            self.index_version = self._content_version(
                {}, len(getattr(self.vector_store, "chunks", []) or []))

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
        self.index_version = self._content_version(
            tables or {}, len(getattr(self.vector_store, "chunks", []) or []))
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
        import time
        
        # Track retrieval time
        retrieval_start = time.time()
        hits = self.retrieve(query, k or cfg.RAG_TOP_K)
        retrieval_time = time.time() - retrieval_start
        
        if not hits:
            msg = "The indexed documents do not contain information relevant to that question."
            if token_callback:
                token_callback(msg)
            return {
                "text": msg,
                "sources": [],
                "retrieval_time": retrieval_time,
                "chunks_used": 0,
                "confidence": 0.0,
            }
        
        context, sources = self._build_context(hits)
        
        # Calculate confidence (average score)
        # hits are (score, chunk) tuples -- see retrieve()'s return type. This
        # line treated them as dicts, so every document-path answer raised
        # AttributeError: 'tuple' object has no attribute 'get'.
        confidence = (sum(score for score, _ in hits) / len(hits)) if hits else 0.0
        
        # Use simple system prompt
        messages = [
            {"role": "system", "content": GROUNDED_SYSTEM},
            {"role": "user", "content": f"Context excerpts:\n{context}\n\nQuestion: {query}"},
        ]
        
        # Track LLM time
        llm_start = time.time()
        text = self.ollama.chat(
            self.chat_model,
            messages,
            options={
                "temperature": 0.0,
                "num_predict": 300,
                "top_k": 40,
                "top_p": 0.9,
            },
            stream=bool(token_callback),
            on_token=token_callback,
        )
        llm_time = time.time() - llm_start
        
        # Return dict with metadata
        return {
            "text": text,
            "sources": sources,
            "retrieval_time": retrieval_time,
            "llm_time": llm_time,
            "chunks_used": len(hits),
            "confidence": confidence,
        }
        

    def _build_relationship_context(self) -> str:
        """Build context about available table relationships."""
        if not hasattr(self, 'relationships') or not self.relationships:
            return ""
        
        context = "\n# Available table relationships for joins:\n"
        
        for rel_key, rels in self.relationships.items():
            for rel in rels:
                context += f"# {rel['left_table']} can join {rel['right_table']} on: "
                context += f"{rel['left_column']} = {rel['right_column']}\n"
        
        return context
    
    