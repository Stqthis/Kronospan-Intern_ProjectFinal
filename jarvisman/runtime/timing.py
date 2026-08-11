"""Per-stage timing for one answered question.

The wall-clock number in the audit log says a question took 14s; it does not
say whether that was planning, codegen, or synthesis. Without the split you
cannot tell whether RAG_UNDERSTANDING_MODEL=qwen2.5:7b actually saved time or
just sent queries down a wrong path that cost a codegen retry -- which is the
exact trade-off the model-routing config asks you to make.

Attribution: cfg.model_for(role, ...) records the role of the call about to be
made, and OllamaClient.chat charges its elapsed time to that role. Every call
site already names its role, so no call site needs changing. An explicit
``with stage(name)`` block wins where one is open.

Never raises: timing must not break answering.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

_local = threading.local()


def _state() -> dict:
    s = getattr(_local, "s", None)
    if s is None:
        s = {"stack": [], "totals": {}, "calls": {}, "pending": ""}
        _local.s = s
    return s


def reset() -> None:
    """Start a fresh measurement (called once per question)."""
    s = _state()
    s["stack"] = []
    s["totals"] = {}
    s["calls"] = {}
    s["pending"] = ""


def set_pending(role: str) -> None:
    """Remember the role of the model call about to be made."""
    try:
        _state()["pending"] = role or ""
    except Exception:
        pass


def _take_pending() -> str:
    s = _state()
    r = s.get("pending") or ""
    s["pending"] = ""          # consumed once, so a stray lookup that is not
    return r                   # followed by a call cannot mislabel a later one


@contextmanager
def stage(name: str):
    """Attribute model time inside this block to ``name``."""
    s = _state()
    s["stack"].append(name)
    try:
        yield
    finally:
        try:
            s["stack"].pop()
        except Exception:
            pass


def record(seconds: float, model: str = "") -> None:
    """Add one model call's elapsed time to the open stage, or to the role
    recorded by the most recent model_for() call."""
    try:
        s = _state()
        name = (s["stack"][-1] if s["stack"]
                else (_take_pending() or "other"))
        s["totals"][name] = s["totals"].get(name, 0.0) + float(seconds)
        s["calls"][name] = s["calls"].get(name, 0) + 1
        if model:
            key = f"model:{model}"
            s["totals"][key] = s["totals"].get(key, 0.0) + float(seconds)
            s["calls"][key] = s["calls"].get(key, 0) + 1
    except Exception:
        pass


def snapshot() -> dict:
    """{stage: {"s": seconds, "n": calls}} for the current question."""
    try:
        s = _state()
        return {k: {"s": round(v, 2), "n": s["calls"].get(k, 0)}
                for k, v in sorted(s["totals"].items(),
                                   key=lambda kv: -kv[1])}
    except Exception:
        return {}


def model_seconds() -> float:
    """Total time waiting on the model (roles only, not the model:* keys)."""
    try:
        s = _state()
        return round(sum(v for k, v in s["totals"].items()
                         if not k.startswith("model:")), 2)
    except Exception:
        return 0.0


@contextmanager
def timed(name: str, model: str = ""):
    """Convenience: open a stage AND record its own wall time. Use for
    non-model work worth measuring (sandbox runs, index lookups)."""
    t0 = time.monotonic()
    with stage(name):
        try:
            yield
        finally:
            record(time.monotonic() - t0, model)