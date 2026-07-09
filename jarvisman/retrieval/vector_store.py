

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