

from __future__ import annotations

import hashlib
import os
import pickle
from typing import Optional


class EmbeddingCache:
    def __init__(self, path: str) -> None:
        self.path = path
        self._data: dict[str, list[float]] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "rb") as fh:
                    data = pickle.load(fh)
                if isinstance(data, dict):
                    self._data = data
            except Exception:
                self._data = {}

    @staticmethod
    def _key(model: str, text: str) -> str:
        h = hashlib.sha1()
        h.update(model.encode("utf-8"))
        h.update(b"\x00")
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    def get(self, model: str, text: str) -> Optional[list[float]]:
        return self._data.get(self._key(model, text))

    def put(self, model: str, text: str, vector: list[float]) -> None:
        self._data[self._key(model, text)] = list(vector)
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "wb") as fh:
                pickle.dump(self._data, fh)
            os.replace(tmp, self.path)  # atomic on POSIX and Windows
            self._dirty = False
        except Exception:
            pass  # best-effort; never fail indexing over the cache

    def __len__(self) -> int:
        return len(self._data)