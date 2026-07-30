

from __future__ import annotations

import os
import pickle

import faiss
import numpy as np


class VectorStore:
    def __init__(self) -> None:
        self.index = None
        self.chunks: list[dict] = []
        self.dim: int | None = None
        self.embed_model: str | None = None

    @property
    def count(self) -> int:
        return 0 if self.index is None else int(self.index.ntotal)

    def reset(self) -> None:
        self.index = None
        self.chunks = []
        self.dim = None
        self.embed_model = None

    def build(self, embeddings: np.ndarray, chunks: list[dict], embed_model: str) -> None:
        emb = np.ascontiguousarray(embeddings, dtype="float32")
        if emb.ndim != 2 or emb.shape[0] != len(chunks):
            raise ValueError("Embedding matrix shape does not match number of chunks.")
        self.dim = int(emb.shape[1])
        faiss.normalize_L2(emb)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(emb)
        self.chunks = list(chunks)
        self.embed_model = embed_model

    def append(self, embeddings: np.ndarray, chunks: list[dict],
               embed_model: str) -> None:
        """Add new chunks to an EXISTING index (incremental updates). The new
        chunks get ids continuing after the current ones. Falls back to a full
        build when the store is empty or the embed model changed (vectors from
        different models are not comparable)."""
        if self.index is None or self.count == 0 \
                or (self.embed_model and embed_model != self.embed_model):
            return self.build(embeddings, chunks, embed_model)
        emb = np.ascontiguousarray(embeddings, dtype="float32")
        if emb.ndim != 2 or emb.shape[0] != len(chunks):
            raise ValueError("Embedding matrix shape does not match number of chunks.")
        faiss.normalize_L2(emb)
        base = len(self.chunks)
        added = []
        for j, ch in enumerate(chunks):
            ch = dict(ch)
            ch["chunk_id"] = base + j
            added.append(ch)
        self.index.add(emb)
        self.chunks.extend(added)

    def remove_sources(self, sources: set) -> int:
        """Drop every chunk whose source file is in ``sources`` (used when a
        file is RE-indexed, so its old chunks don't duplicate the new ones).
        Vectors are recovered from the flat index and the index is rebuilt.
        Returns the number of removed chunks."""
        if self.index is None or not self.chunks or not sources:
            return 0
        keep = [i for i, c in enumerate(self.chunks)
                if c.get("source") not in sources]
        removed = len(self.chunks) - len(keep)
        if removed == 0:
            return 0
        if keep:
            # one C-level batch reconstruct, then numpy-pick the kept rows
            # (replaces one Python->FAISS call per kept vector)
            all_vecs = self.index.reconstruct_n(0, self.index.ntotal)
            vecs = all_vecs[keep]
        else:
            vecs = np.zeros((0, self.dim or 1), dtype="float32")
        kept_chunks = []
        for j, i in enumerate(keep):
            ch = dict(self.chunks[i])
            ch["chunk_id"] = j
            kept_chunks.append(ch)
        self.index = faiss.IndexFlatIP(self.dim)
        if len(kept_chunks):
            self.index.add(np.ascontiguousarray(vecs, dtype="float32"))
        self.chunks = kept_chunks
        return removed

    def search(self, query_emb: np.ndarray, k: int) -> list[tuple[float, dict]]:
        if self.index is None or self.count == 0:
            return []
        q = np.ascontiguousarray(query_emb, dtype="float32")
        faiss.normalize_L2(q)
        k = min(k, self.count)
        scores, idx = self.index.search(q, k)
        results: list[tuple[float, dict]] = []
        for score, i in zip(scores[0], idx[0]):
            if i == -1:
                continue
            results.append((float(score), self.chunks[i]))
        return results

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #
    def save(self, directory: str) -> None:
        if self.index is None:
            return
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "meta.pkl"), "wb") as fh:
            pickle.dump(
                {"chunks": self.chunks, "dim": self.dim, "embed_model": self.embed_model},
                fh,
            )

    def load(self, directory: str) -> None:
        index_path = os.path.join(directory, "index.faiss")
        meta_path = os.path.join(directory, "meta.pkl")
        if not (os.path.exists(index_path) and os.path.exists(meta_path)):
            raise FileNotFoundError("No saved index found.")
        try:
            self.index = faiss.read_index(index_path)
            with open(meta_path, "rb") as fh:
                meta = pickle.load(fh)
            if not isinstance(meta, dict):
                raise ValueError("meta.pkl has an unexpected format")
        except FileNotFoundError:
            raise
        except Exception as exc:
            # a corrupt / numpy-incompatible index must surface as the same
            # clean signal the caller already handles, not a raw traceback
            self.reset()
            raise FileNotFoundError(
                f"Saved index could not be loaded ({exc}); "
                "rebuild the index.") from exc
        self.chunks = meta.get("chunks", [])
        self.dim = meta.get("dim")
        self.embed_model = meta.get("embed_model")