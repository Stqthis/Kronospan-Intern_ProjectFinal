

from __future__ import annotations

import hashlib
import json
import os
import re
import warnings
from typing import Callable, Optional
from jarvisman.llm.llm_json import extract_json as _extract_json

SAMPLE_ROWS = 100          # rows shown to the model per table
_MAX_SAMPLE_CHARS = 8000   # cap the rendered sample so the prompt stays bounded
_KEEP_RATIO = 0.8          # a conversion is accepted only if it keeps >=80% of
                           # the column's original non-null values
_DATE_PARTS = re.compile(r"^\s*(\d{1,4})\D+(\d{1,4})\D+(\d{1,4})")


# --------------------------------------------------------------------------- #
# Persistence (so the meanings survive a reload; the file is human-readable)  #
# --------------------------------------------------------------------------- #
def _profile_path(directory: str) -> str:
    return os.path.join(directory, "table_profile.json")


def save_profile(directory: str, profile: dict) -> None:
    try:
        os.makedirs(directory, exist_ok=True)
        with open(_profile_path(directory), "w", encoding="utf-8") as fh:
            json.dump(profile, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_profile(directory: str) -> dict:
    path = _profile_path(directory)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# Prompt + parsing                                                            #
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _sample_text(df) -> str:
    n = min(SAMPLE_ROWS, len(df))
    try:
        txt = df.head(n).to_string(max_rows=n, max_cols=60)
    except Exception:
        txt = str(df.head(n))
    if len(txt) > _MAX_SAMPLE_CHARS:
        txt = txt[:_MAX_SAMPLE_CHARS] + "\n... (sample truncated)"
    return txt


def _prompt(name: str, df) -> str:
    cols = ", ".join(repr(str(c)) for c in df.columns)
    return (
        "You are shown a data table: its column names and its first rows. Study "
        "the actual values and describe the table. For EACH column decide its "
        "data type, and if it is a date, work out the exact format from how the "
        "values are written in THIS table.\n"
        "Types (choose one per column):\n"
        '  - "date"   : the values are dates or timestamps\n'
        '  - "number" : numeric values (amounts, counts, years, ids)\n'
        '  - "text"   : anything else (names, codes, categories)\n'
        'For a "date" column, infer "format" as a Python strptime pattern that '
        "reproduces exactly how the dates appear in the sample (the separators, "
        "the order of day/month/year, 2- vs 4-digit year, any month names). "
        "Determine the day/month order from the values themselves: a component "
        "greater than 12 must be the day. Do not assume a regional convention; "
        "read it from the data. For non-date columns set \"format\" to null. "
        '"meaning" is a short plain-language description of what the column holds, IN BUSINESS TERMS. When several columns look similar (many money columns, for example), say which ONE is the primary/canonical figure a business user means by default, and note that the others are derived, scaled (e.g. in thousands), or internal -- so a downstream step can pick the right column instead of guessing. Name the unit/currency and whether the value is already converted.\n\n'
        f"Table name: {name}\nRows: {len(df)}\nColumns: [{cols}]\n\n"
        f"First rows:\n{_sample_text(df)}\n\n"
        "Reply with ONLY a JSON object -- no prose, no markdown, no code fences -- "
        "exactly:\n"
        '{"table_summary": "<what each ROW represents and the table purpose>", '
        '"columns": {"<column name>": {"type": "date|number|text", '
        '"format": "<strptime pattern or null>", "meaning": "<short>"}}}\n'
        f"Use a key for every column, copied VERBATIM from: [{cols}]"
    )





def _profile_one(ollama, chat_model: str, name: str, df) -> dict:
    """Ask the model to understand ONE table. Returns
    {"summary": str, "columns": {col: {"type","format","meaning"}}} with keys
    normalised back onto the real columns. Empty/safe on any failure."""
    try:
        # one JSON entry per column: budget scales with width, hard-capped
        budget = min(4096, 300 + 60 * max(1, len(df.columns)))
        raw = ollama.chat(
            chat_model, [{"role": "user", "content": _prompt(name, df)}],
            options={"temperature": 0.0, "num_ctx": 8192,
                     "num_predict": budget},
            format="json",
        )
    except Exception:
        raw = ""
    parsed = _extract_json(raw) or {}
    cols_in = parsed.get("columns")
    cols_in = cols_in if isinstance(cols_in, dict) else {}

    real = {str(c) for c in df.columns}
    norm = {_norm(c): str(c) for c in df.columns}
    columns: dict[str, dict] = {}
    for k, v in cols_in.items():
        key = k if k in real else norm.get(_norm(k))
        if key and isinstance(v, dict):
            columns[key] = {
                "type": str(v.get("type", "")).strip().lower(),
                "format": v.get("format"),
                "meaning": str(v.get("meaning", "")).strip(),
            }
    return {"summary": str(parsed.get("table_summary", "")).strip(), "columns": columns}


# --------------------------------------------------------------------------- #
# Applying the verdict (with a deterministic safety net)                      #
# --------------------------------------------------------------------------- #
def _kept_enough(orig, converted) -> bool:
    o = int(orig.notna().sum())
    if o == 0:
        return True
    return int(converted.notna().sum()) >= _KEEP_RATIO * o


def _to_numeric(col, pd):
    return pd.to_numeric(col, errors="coerce")


def _detect_dayfirst(col) -> Optional[bool]:
    """Fallback day/month detection from the values: a part >12 must be the day."""
    a_max = b_max = seen = 0
    for v in col.dropna().astype(str):
        m = _DATE_PARTS.match(v)
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        if a > 31:
            return None
        a_max, b_max, seen = max(a_max, a), max(b_max, b), seen + 1
    if seen == 0:
        return None
    if a_max > 12:
        return True
    if b_max > 12:
        return False
    return None


def _to_datetime_auto(col, pd):
    """Parse to datetime with day/month order detected from the data."""
    dayfirst = _detect_dayfirst(col)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if dayfirst is None:
                return pd.to_datetime(col, errors="coerce")
            return pd.to_datetime(col, dayfirst=dayfirst, errors="coerce")
        except Exception:
            return None


def _to_datetime_fmt(col, pd, fmt: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.to_datetime(col, format=fmt, errors="coerce")
        except Exception:
            return None


def _coerce_column(col, verdict: Optional[dict], pd):
    """Apply the model's per-column verdict; fall back deterministically if it
    is missing or would destroy the column."""
    if getattr(col.dtype, "kind", "O") == "M":            # already datetime
        return col

    t = (verdict or {}).get("type")
    fmt = (verdict or {}).get("format")
    fmt = fmt if (isinstance(fmt, str) and fmt.strip() and fmt.strip().lower() != "null") else None

    if t == "number":
        conv = _to_numeric(col, pd)
        return conv if _kept_enough(col, conv) else col

    if t == "date":
        if fmt:                                            # try the model's exact format
            conv = _to_datetime_fmt(col, pd, fmt)
            if conv is not None and _kept_enough(col, conv):
                return conv
        conv = _to_datetime_auto(col, pd)                  # safety net: detect from data
        return conv if (conv is not None and _kept_enough(col, conv)) else col

    if t == "text":
        num = _to_numeric(col, pd)                         # never drop a plainly-numeric column
        if int(num.notna().sum()) > 0 and int(num.notna().sum()) >= 0.95 * max(int(col.notna().sum()), 1):
            return num
        return col

    # No / unknown verdict -> deterministic: number, then date, then text.
    num = _to_numeric(col, pd)
    if int(num.notna().sum()) > 0 and _kept_enough(col, num):
        return num
    dt = _to_datetime_auto(col, pd)
    if dt is not None and getattr(dt.dtype, "kind", "O") == "M" and int(dt.notna().sum()) > 0 and _kept_enough(col, dt):
        return dt
    return col


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #
def _schema_sig(df) -> str:
    """Stable signature of a table's raw shape: ordered (column, dtype) pairs.
    Uses hashlib so the value is identical across processes/runs (the builtin
    hash() is per-process salted and would never match a persisted one)."""
    try:
        parts = [f"{c}:{df[c].dtype}" for c in df.columns]
        payload = "|".join(parts) + f"#rows>0={len(df) > 0}"
        return hashlib.md5(payload.encode("utf-8")).hexdigest()
    except Exception:
        return ""


def profile_and_apply(ollama, chat_model: str, dataframes: dict,
                      progress: Optional[Callable[[str], None]] = None,
                      existing: Optional[dict] = None) -> tuple[dict, dict]:
    """For every table: the model reads the sample rows and decides each column's
    type + format + meaning; the code applies the types and keeps the meanings.

    Returns (typed_dataframes, profile) where
      profile[table_key] = {"summary": str,
                            "columns": {col: {"type","format","meaning"}}}.
    """
    import pandas as pd

    typed_out: dict = {}
    profile: dict = {}
    batch_seen: dict[str, dict] = {}   # schema_sig -> profile computed THIS run
    total = len(dataframes)
    for i, (name, df) in enumerate(dataframes.items(), 1):
        if progress:
            progress(f"Understanding table {i}/{total}: {name} ...")
        if df is None or len(df) == 0:
            typed_out[name] = df
            profile[name] = {"summary": "", "columns": {}}
            continue

        sig = _schema_sig(df)
        prior = (existing or {}).get(name) or {}
        if sig and prior.get("schema_hash") == sig and prior.get("columns"):
            # unchanged schema across runs -> reuse the saved understanding
            if progress:
                progress(f"Reusing profile for unchanged table: {name}")
            prof = prior
        elif sig and sig in batch_seen:
            # sibling table with an identical schema already profiled THIS run
            # (monthly DDR files, per-entity loan schedules) -> no 2nd LLM call
            if progress:
                progress(f"Reusing sibling-schema profile for: {name}")
            prof = dict(batch_seen[sig])
        else:
            prof = _profile_one(ollama, chat_model, name, df)
            prof["schema_hash"] = sig
            if sig:
                batch_seen[sig] = prof
        verdicts = prof.get("columns", {})

        typed = df.copy()
        for c in pd.unique(typed.columns):
            col = typed[c]
            if hasattr(col, "columns"):       # duplicated label -> a DataFrame; skip
                continue
            typed[c] = _coerce_column(col, verdicts.get(str(c)), pd)
        typed_out[name] = typed
        profile[name] = prof
    return typed_out, profile