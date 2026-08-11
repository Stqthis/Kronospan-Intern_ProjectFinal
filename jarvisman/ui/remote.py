"""Client bridge to the Dockerised backend (jarvisman.server).

When RAG_BACKEND_URL is set (e.g. http://dgx:8800) the desktop app becomes a
thin client: questions, dashboard charts, reports and inventory all come from
the backend over JSON — the model, the index and the data never leave the
server. The bridge exposes the same little surface the GUI already uses on
the local Agent (handle / clear_history / chat_model / dataframes), so the
rest of the app does not care which mode it is in.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import requests

_API_KEY = os.environ.get("RAG_API_KEY", "").strip()


def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if _API_KEY:
        h["X-Api-Key"] = _API_KEY
    return h


class RemoteBridge:
    """Talks to jarvisman.server; every call is fail-soft (returns something
    renderable rather than raising into the UI)."""

    def __init__(self, base_url: str) -> None:
        self.base = base_url.rstrip("/")
        self.chat_model = ""
        self.dataframes: dict = {}       # tables live on the server
        self._status: dict = {}
        self._status_t = 0.0

    # -- agent-compatible surface ---------------------------------------- #
    def handle(self, query: str, *args, **kwargs) -> dict:
        try:
            r = requests.post(f"{self.base}/ask", json={"q": query},
                              headers=_headers(), timeout=(10, 900))
            r.raise_for_status()
            res = r.json()
            if not isinstance(res, dict):
                return {"text": str(res)}
            return res
        except Exception as exc:
            return {"text": f"Backend error: {type(exc).__name__}: {exc}. "
                            f"Is the API container running at {self.base}?"}

    def clear_history(self) -> None:
        try:
            requests.post(f"{self.base}/clear_history", json={},
                          headers=_headers(), timeout=10)
        except Exception:
            pass

    # -- data for the pages ------------------------------------------------ #
    def health(self) -> Optional[dict]:
        try:
            r = requests.get(f"{self.base}/health", headers=_headers(), timeout=8)
            r.raise_for_status()
            h = r.json()
            self.chat_model = h.get("model") or self.chat_model
            return h
        except Exception:
            return None

    def status(self, max_age: float = 5.0) -> dict:
        """Inventory (tables/docs/chunks/model), cached briefly."""
        now = time.time()
        if self._status and now - self._status_t < max_age:
            return self._status
        try:
            r = requests.get(f"{self.base}/status", headers=_headers(), timeout=15)
            r.raise_for_status()
            self._status = r.json() or {}
            self._status_t = now
            self.chat_model = self._status.get("model") or self.chat_model
        except Exception:
            self._status = self._status or {}
        return self._status

    def charts(self) -> list:
        try:
            r = requests.get(f"{self.base}/charts", headers=_headers(), timeout=30)
            r.raise_for_status()
            return (r.json() or {}).get("charts") or []
        except Exception:
            return []

    def reports(self) -> list:
        try:
            r = requests.get(f"{self.base}/reports", headers=_headers(), timeout=30)
            r.raise_for_status()
            return (r.json() or {}).get("reports") or []
        except Exception:
            return []

    def scoreboard(self) -> dict:
        try:
            r = requests.get(f"{self.base}/scoreboard", headers=_headers(),
                             timeout=10)
            r.raise_for_status()
            return r.json() or {}
        except Exception:
            return {}

    def rules_get(self) -> dict:
        try:
            r = requests.get(f"{self.base}/rules", headers=_headers(),
                             timeout=10)
            r.raise_for_status()
            return r.json() or {}
        except Exception:
            return {}

    def rules_save(self, text: str) -> bool:
        try:
            r = requests.post(f"{self.base}/rules", json={"text": text},
                              headers=_headers(), timeout=15)
            return r.ok
        except Exception:
            return False

    def run_report(self, rid: int) -> Optional[dict]:
        try:
            r = requests.post(f"{self.base}/report", json={"id": rid},
                              headers=_headers(), timeout=120)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None
