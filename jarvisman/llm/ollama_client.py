
from __future__ import annotations

import json
from typing import Callable, Optional

import requests

from jarvisman import config as cfg


def _ka(value):
    """Normalize keep_alive: the string "-1" -> int -1 so Ollama reads it as
    'never unload' instead of failing to parse it as a duration."""
    if str(value).strip() in ("-1", "-1s", "-1m", "-1h"):
        return -1
    return value


class OllamaError(RuntimeError):
    """Raised for any problem talking to the local Ollama server."""


class OllamaClient:
    def __init__(self, host: str = cfg.OLLAMA_HOST):
        self.host = host.rstrip("/")

    # ------------------------------------------------------------------ #
    # Health / discovery                                                 #
    # ------------------------------------------------------------------ #
    def warmup(self, model: str, keep_alive: Optional[str] = None) -> bool:
        """Best-effort: load and pin the model so the first real query is fast."""
        try:
            self.chat(model, [{"role": "user", "content": "ok"}],
                      options={"num_predict": 1, "temperature": 0.0},
                      keep_alive=keep_alive)
            return True
        except Exception:
            return False

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
        format: Optional[str] = None,
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
        # Right-size the KV context for length-capped calls. When the caller
        # bounds generation with num_predict we can shrink num_ctx to fit the
        # actual prompt plus that budget, which lowers prefill/allocation cost
        # on large models. We estimate prompt length at an upper bound of one
        # token per character (real ratio is >= 1 char/token for every script,
        # including Greek), and never go above the caller's request nor below a
        # safe floor -- so prompts are never truncated. Unbounded calls
        # (num_predict unset) keep the full requested context untouched.
        try:
            requested_ctx = int(opts.get("num_ctx") or cfg.NUM_CTX)
            npredict = int(opts.get("num_predict") or 0)
            if npredict > 0:
                chars = sum(len(str(m.get("content", ""))) for m in messages)
                needed = chars + npredict + 256           # prompt + gen + slack
                sized = ((needed + 511) // 512) * 512      # round up to 512
                opts["num_ctx"] = max(2048, min(requested_ctx, sized))
        except Exception:
            pass
        payload = {
            "model": model,
            "messages": messages,
            "stream": bool(stream and on_token),
            "options": opts,
            "keep_alive": _ka(keep_alive or cfg.KEEP_ALIVE),
        }
        # Constrain decoding to strict JSON when the caller asks for it. Ollama
        # stops sampling at the end of the JSON value, so structured calls
        # (planner, router, table profiling) don't waste tokens on trailing
        # prose and parse on the first attempt.
        if format:
            payload["format"] = format
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
        payload = {"model": model, "prompt": text, "keep_alive": _ka(keep_alive or cfg.KEEP_ALIVE)}
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
        payload = {"model": model, "input": texts, "keep_alive": _ka(keep_alive or cfg.KEEP_ALIVE)}
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