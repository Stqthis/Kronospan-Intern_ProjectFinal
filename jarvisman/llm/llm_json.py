"""Shared helpers for parsing LLM text output.

Every place that asks the local model for JSON must tolerate the same two
realities: (1) reasoning models (Qwen3 etc.) wrap output in <think> blocks,
and (2) small models often surround the JSON with prose or code fences.
Centralising the parsing here means a model upgrade cannot break one call
site while the others keep working.
"""

from __future__ import annotations

import json
import re
from typing import Optional

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def strip_think(raw: str) -> str:
    """Remove <think>...</think> blocks (reasoning models emit them)."""
    return _THINK_RE.sub("", raw or "")


def _balanced_object(text: str) -> Optional[str]:
    """Return the first brace-balanced {...} slice, string-aware."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json(raw: str) -> Optional[dict]:
    """Best-effort: strip think blocks / fences, then parse the first JSON
    object. Tries the balanced slice first (robust to prose after the JSON
    that contains '}'), then the widest first-{ to last-} slice."""
    if not raw:
        return None
    raw = strip_think(raw)
    m = _FENCE_RE.search(raw)
    if m:
        raw = m.group(1)
    for candidate in (_balanced_object(raw),):
        if candidate:
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
    a, b = raw.find("{"), raw.rfind("}")
    if a != -1 and b > a:
        try:
            obj = json.loads(raw[a : b + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None
