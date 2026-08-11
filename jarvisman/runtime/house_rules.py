"""House rules: standing business policies, maintained as a plain text file.

Requirements like "always show the company name AND its company code" or
"'in EUR' without 'equivalent' still means converted" are POLICY, not code.
They belong in a file the deployment owner (or their managers' requirements)
can edit without touching Python: one rule per line, '#' for comments.

The rules are injected verbatim into BOTH places the model makes judgement
calls -- the JSON planner prompt and the codegen-fallback prompt -- so a
policy holds no matter which tier answers. The file is re-read only when its
mtime changes (cheap enough for every question, robust to edits while the
app runs). An absent or empty file is a silent no-op.

Default location: <RAG_DATA_DIR>/rules.txt (override: RAG_HOUSE_RULES).
"""

from __future__ import annotations

import os

from jarvisman import config as cfg

_cache: dict = {"path": None, "mtime": None, "text": ""}

# Shipped defaults, tuned to the deployment's data shapes. Used only when the
# owner has not created their own rules file, so it is a safe starting point
# that any deployment can override or delete.
_DEFAULT_RULES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "default_rules.txt")

MAX_RULES_CHARS = 6000   # keep the prompt overhead bounded


def load_rules(path: str = None) -> str:
    """The raw rule lines (comments/blanks removed), or ''. mtime-cached."""
    p = path or getattr(cfg, "HOUSE_RULES_PATH", "")
    # Fall back to the shipped defaults when no user rules file exists yet.
    if not p or not os.path.exists(p):
        p = _DEFAULT_RULES_PATH
    if not p or not os.path.exists(p):
        return ""
    try:
        mtime = os.path.getmtime(p)
    except OSError:
        return ""
    if _cache["path"] == p and _cache["mtime"] == mtime:
        return _cache["text"]
    try:
        with open(p, encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh]
    except Exception:
        return ""
    rules = [ln for ln in lines if ln and not ln.startswith("#")]
    text = "\n".join(f"- {r}" for r in rules)[:MAX_RULES_CHARS]
    _cache.update(path=p, mtime=mtime, text=text)
    return text


def prompt_block(path: str = None) -> str:
    """Ready-to-inject block, or '' when there are no rules."""
    rules = load_rules(path)
    if not rules:
        return ""
    return ("House rules (standing policies of this deployment -- apply "
            "them to EVERY answer):\n" + rules + "\n")
