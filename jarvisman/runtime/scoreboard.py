"""Accuracy scoreboard: latest verification results, shown on the System
page so every user can see when the checks last ran and whether they pass.

Written by eval/run_use_cases.py (deterministic data checks) and
test_queries.py (graded LLM answers) into <RAG_DATA_DIR>/scoreboard.json.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

from jarvisman import config as cfg


def _path() -> str:
    os.makedirs(cfg.DATA_DIR, exist_ok=True)
    return os.path.join(cfg.DATA_DIR, "scoreboard.json")


def read() -> dict:
    try:
        with open(_path(), encoding="utf-8") as fh:
            return json.load(fh) or {}
    except Exception:
        return {}


def record(kind: str, payload: dict) -> None:
    """Merge {kind: payload+timestamp} into the scoreboard; never raises."""
    try:
        data = read()
        payload = dict(payload)
        payload["ts"] = datetime.now().isoformat(timespec="seconds")
        data[kind] = payload
        with open(_path(), "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1)
    except Exception:
        pass
