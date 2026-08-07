"""Per-question audit trail.

Every answered question is appended as one JSON line to
<RAG_DATA_DIR>/audit/audit-YYYYMM.jsonl: timestamp, question, tool, the
tables the generated code touched, the generated code itself, the first
part of the answer, and how long it took. This is the trail treasury can
show an auditor: question -> code -> tables -> answer.

Disable with RAG_AUDIT=0. Failures never disturb answering.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime

from jarvisman import config as cfg

_DFS_RE = re.compile(r"""dfs\[\s*['"]([^'"]+)['"]\s*\]""")


def enabled() -> bool:
    return os.environ.get("RAG_AUDIT", "1") == "1"


def _path() -> str:
    d = os.path.join(cfg.DATA_DIR, "audit")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"audit-{datetime.now():%Y%m}.jsonl")


def tables_used(result: dict) -> list:
    """Which tables the generated code actually read."""
    code = (result or {}).get("code") or ""
    seen, out = set(), []
    for name in _DFS_RE.findall(code):
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out

def _stages() -> dict:
    """Per-stage model time for the question just answered; {} if unavailable."""
    try:
        from jarvisman.runtime import timing
        return timing.snapshot()
    except Exception:
        return {}


def log(question: str, result: dict, seconds: float = 0.0,
        client: str = "app") -> None:
    """Append one audit line; never raises."""
    if not enabled():
        return
    try:
        result = result or {}
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "client": client,
            "q": question,
            "tool": result.get("tool"),
            "tables_used": tables_used(result),
            "sources": [f"{s.get('source')} {s.get('location', '')}".strip()
                        for s in (result.get("sources") or [])][:8],
            "answer": ((result.get("text") or "")[:500]
                       or ("<table>" if result.get("table_html") else "")),
            "rows": result.get("row_count"),
            "code": result.get("code") or "",
            "seconds": round(float(seconds), 2),
            "stages": _stages(),
        }
        with open(_path(), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def recent(n: int = 50) -> list:
    """Last n audit entries (most recent first); [] when none."""
    try:
        p = _path()
        if not os.path.exists(p):
            return []
        with open(p, encoding="utf-8") as fh:
            lines = fh.readlines()[-n:]
        out = []
        for ln in reversed(lines):
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
        return out
    except Exception:
        return []
