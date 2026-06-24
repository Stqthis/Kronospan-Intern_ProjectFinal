
from __future__ import annotations

import json
from typing import Callable, Optional

import requests

from jarvisman import config as cfg


class OllamaError(RuntimeError):
    """Raised for any problem talking to the local Ollama server."""


class OllamaClient:
    def __init__(self, host: str = cfg.OLLAMA_HOST):
        self.host = host.rstrip("/")

    # ------------------------------------------------------------------ #
    # Health / discovery                                                 #
    # ------------------------------------------------------------------ #
    def is_alive(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/version", timeout=cfg.LIST_TIMEOUT)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=cfg.LIST_TIMEOUT)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(self._conn_hint(exc)) from exc
        data = r.json()
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]

    # ------------------------------------------------------------------ #
    # Chat / generation                                                  #
    # ------------------------------------------------------------------ #
    def chat(
        self,
        model: str,
        messages: list[dict],
        options: Optional[dict] = None,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
        keep_alive: Optional[str] = None,
    ) -> str:
        """Run a chat completion. Returns the full assistant text.

        If ``stream`` is True and ``on_token`` is given, partial tokens are
        delivered to the callback as they arrive (so the GUI can render
        progressively); the complete text is still returned.
        """
        # Always send an explicit context window: Ollama's default num_ctx is
        # tiny and it silently truncates longer prompts from the FRONT, which
        # destroys schema-heavy prompts. Callers can still override num_ctx.
        opts = {"num_ctx": cfg.NUM_CTX}
        opts.update(options or {})
        payload = {
            "model": model,
            "messages": messages,
            "stream": bool(stream and on_token),
            "options": opts,
            "keep_alive": keep_alive or cfg.KEEP_ALIVE,
        }
        url = f"{self.host}/api/chat"

        if not payload["stream"]:
            try:
                r = requests.post(url, json=payload, timeout=cfg.CHAT_TIMEOUT)
                r.raise_for_status()
            except requests.RequestException as exc:
                raise OllamaError(self._chat_hint(exc, model)) from exc
            data = r.json()
            return (data.get("message") or {}).get("content", "")

        # Streaming: Ollama returns newline-delimited JSON objects.
        parts: list[str] = []
        try:
            with requests.post(url, json=payload, stream=True, timeout=cfg.CHAT_TIMEOUT) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = (obj.get("message") or {}).get("content", "")
                    if token:
                        parts.append(token)
                        if on_token:
                            on_token(token)
                    if obj.get("done"):
                        break
        except requests.RequestException as exc:
            raise OllamaError(self._chat_hint(exc, model)) from exc
        return "".join(parts)

    # ------------------------------------------------------------------ #
    # Embeddings                                                         #
    # ------------------------------------------------------------------ #
    def embed(self, text: str, model: str, keep_alive: Optional[str] = None) -> list[float]:
        payload = {"model": model, "prompt": text, "keep_alive": keep_alive or cfg.KEEP_ALIVE}
        url = f"{self.host}/api/embeddings"
        try:
            r = requests.post(url, json=payload, timeout=cfg.EMBED_TIMEOUT)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(self._chat_hint(exc, model)) from exc
        vec = r.json().get("embedding")
        if not vec:
            raise OllamaError(
                f"Embedding model '{model}' returned no vector. "
                f"Is it an embedding model (e.g. nomic-embed-text)?"
            )
        return vec

    def embed_batch(
        self, texts: list[str], model: str, keep_alive: Optional[str] = None
    ) -> list[list[float]]:
        """Embed many texts in one call.

        Tries the batch ``/api/embed`` endpoint (newer Ollama) and falls back
        to per-item ``/api/embeddings`` if that endpoint is unavailable, so it
        works across Ollama versions. Batching cuts per-request overhead, which
        speeds up indexing on CPU.
        """
        if not texts:
            return []
        url = f"{self.host}/api/embed"
        payload = {"model": model, "input": texts, "keep_alive": keep_alive or cfg.KEEP_ALIVE}
        try:
            r = requests.post(url, json=payload, timeout=cfg.EMBED_TIMEOUT)
            r.raise_for_status()
            embs = r.json().get("embeddings")
            if embs and len(embs) == len(texts):
                return embs
        except requests.RequestException:
            pass  # fall back to the stable single-item endpoint
        return [self.embed(t, model, keep_alive=keep_alive) for t in texts]

    # ------------------------------------------------------------------ #
    # Error message helpers                                              #
    # ------------------------------------------------------------------ #
    def _conn_hint(self, exc: Exception) -> str:
        if isinstance(exc, requests.ConnectionError):
            return (
                f"Cannot reach Ollama at {self.host}. "
                f"Start it with `ollama serve` and try again."
            )
        return f"Ollama request failed: {exc}"

    def _chat_hint(self, exc: Exception, model: str) -> str:
        if isinstance(exc, requests.ConnectionError):
            return self._conn_hint(exc)
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            if exc.response.status_code == 404:
                return (
                    f"Model '{model}' is not available locally. "
                    f"Pull it first with `ollama pull {model}`."
                )
            return f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text[:200]}"
        if isinstance(exc, requests.Timeout):
            return (
                f"Ollama timed out while using '{model}'. "
                f"Large models on CPU can be slow; try a smaller model."
            )
        return f"Ollama request failed: {exc}"