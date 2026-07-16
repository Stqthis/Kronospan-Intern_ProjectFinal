"""Table semantic cards: the model's understanding of what each table IS.

Generated ONCE per table at index time (one LLM call), validated against the
actual data, persisted as ``table_cards.json`` in the index directory, and
rendered into the planner's schema block at zero query-time cost. A card:

    {
      "purpose": "Directorships of persons at companies.",
      "grain": "one row per person-company directorship",
      "entity_key": ["COMPANY CODE"],
      "validity": [{"start": "START_DATE", "end": "END_DATE",
                    "null_end_means": "active"}],
      "date_columns": {"START_DATE": "appointment date"},
      "status_columns": {"STATUS": "A=active, T=terminated"},
      "measures": {"AMOUNT": "EUR"},
      "caveats": ["END_DATE is the directorship end, not company closure"]
    }

Hand-editable: correct any field and keep ``schema_hash`` untouched -- a
card is regenerated ONLY when its table's schema hash changes. Validation
(``sanitize_card``) drops claims that contradict reality (nonexistent
columns, non-date validity pairs, grossly non-unique keys) with a logged
note. Cards INFORM planning; they never bypass value verification or the
validator. With RAG_TABLE_CARDS=0 or no file present, name-pattern
heuristics apply and nothing else changes.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

from jarvisman import config as cfg
from jarvisman.llm.llm_json import extract_json

_FILE = "table_cards.json"

_CARD_PROMPT = (
    "You document database tables. Given the schema and sample data of ONE "
    "table, answer with ONLY a JSON object (no prose, no markdown) with "
    "these keys:\n"
    '{"purpose": "1-2 sentences: what this table shows",\n'
    ' "table_kind": "snapshot | history | transactions | reference",\n'
    ' "as_of": "dd.mm.yyyy if this table is a snapshot of ONE date (check the\n'
    'file name and any single-valued date column); else null",\n'
    ' "grain": "one row per <entity or event>",\n'
    ' "entity_key": ["column(s) identifying the main entity"],\n'
    ' "validity": [{"start": "<col>", "end": "<col>", '
    '"null_end_means": "active"}],\n'
    ' "date_columns": {"<col>": "meaning"},\n'
    ' "status_columns": {"<col>": "meaning of the codes"},\n'
    ' "measures": {"<col>": "unit/currency; state if it is the NATIVE '
    'currency amount or a CONVERTED/equivalent amount"},\n'
    ' "caveats": ["important interpretation notes"]}\n'
    "Only mention columns that really exist. Use [] / {} when a key does "
    "not apply.\n\n"
)


# --------------------------------------------------------------------------- #
# Schema hash: a card is tied to its table's structure                        #
# --------------------------------------------------------------------------- #
_FNAME_DATE_RE = __import__("re").compile(
    r"(\d{1,2})[-._](\d{1,2})[-._](\d{4})|(\d{4})[-._](\d{1,2})[-._](\d{1,2})")


def filename_asof(table_name: str):
    """Snapshot date encoded in the file name (CY01-DDR-31-07-2023-DATA.xlsx
    -> '31.07.2023'). Deterministic; seeds/verifies a card's as_of."""
    m = _FNAME_DATE_RE.search(table_name or "")
    if not m:
        return None
    g = m.groups()
    try:
        if g[0]:
            d, mo, y = int(g[0]), int(g[1]), int(g[2])
        else:
            y, mo, d = int(g[3]), int(g[4]), int(g[5])
        if 1 <= d <= 31 and 1 <= mo <= 12 and 1990 <= y <= 2100:
            return f"{d:02d}.{mo:02d}.{y}"
    except (TypeError, ValueError):
        pass
    return None


def schema_hash(df) -> str:
    sig = "|".join(f"{c}:{getattr(df[c].dtype, 'kind', 'O')}"
                   for c in map(str, df.columns))
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Reality validation                                                          #
# --------------------------------------------------------------------------- #
def sanitize_card(card: dict, tprof, df) -> tuple[dict, list]:
    """Drop every claim the data contradicts. Returns (clean_card, notes)."""
    notes: list[str] = []
    if not isinstance(card, dict):
        return {}, ["card was not a JSON object; ignored"]
    cols = {str(c) for c in df.columns}
    by_name = {c.name: c for c in tprof.columns} if tprof else {}

    def _exists(c) -> bool:
        return isinstance(c, str) and c in cols

    clean: dict = {
        "purpose": str(card.get("purpose") or "")[:300],
        "grain": str(card.get("grain") or "")[:160],
    }
    import re as _re
    kind = str(card.get("table_kind") or "").lower().strip()
    clean["table_kind"] = kind if kind in (
        "snapshot", "history", "transactions", "reference") else ""
    asof = str(card.get("as_of") or "").strip()
    if asof and asof.lower() not in ("null", "none") \
            and _re.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{4}", asof):
        d, mo, y = asof.split(".")
        clean["as_of"] = f"{int(d):02d}.{int(mo):02d}.{y}"
    else:
        clean["as_of"] = None
        if asof and asof.lower() not in ("null", "none"):
            notes.append(f"as_of '{asof}' not a dd.mm.yyyy date; dropped")

    keys = [k for k in (card.get("entity_key") or []) if _exists(k)]
    dropped = [k for k in (card.get("entity_key") or []) if not _exists(k)]
    for k in dropped:
        notes.append(f"entity_key '{k}' does not exist; dropped")
    kept_keys = []
    for k in keys:
        try:
            nn = df[k].dropna()
            ur = nn.nunique() / max(len(nn), 1)
        except Exception:
            ur = 0.0
        if ur >= 0.9 or len(keys) > 1:   # composite keys judged leniently
            kept_keys.append(k)
        else:
            notes.append(f"entity_key '{k}' is not unique "
                         f"({ur:.0%} distinct); dropped")
    clean["entity_key"] = kept_keys

    validity = []
    for vp in (card.get("validity") or []):
        s, e = vp.get("start"), vp.get("end")
        if not (_exists(s) and _exists(e)):
            notes.append(f"validity pair {s!r}/{e!r} references missing "
                         "columns; dropped")
            continue
        roles = {by_name[c].role if c in by_name else "?" for c in (s, e)}
        if not roles <= {"date", "year"}:
            notes.append(f"validity pair {s!r}/{e!r} is not date-roled; "
                         "dropped")
            continue
        validity.append({"start": s, "end": e,
                         "null_end_means":
                             str(vp.get("null_end_means") or "active")[:40]})
    clean["validity"] = validity

    for key in ("date_columns", "status_columns", "measures"):
        src = card.get(key) or {}
        clean[key] = {c: str(m)[:120] for c, m in src.items() if _exists(c)}
        for c in src:
            if not _exists(c):
                notes.append(f"{key} entry '{c}' does not exist; dropped")

    clean["caveats"] = [str(x)[:200] for x in (card.get("caveats") or [])][:4]
    return clean, notes


# --------------------------------------------------------------------------- #
# Generation (index time only)                                                #
# --------------------------------------------------------------------------- #
def generate_card(ollama, chat_model: str, table_name: str,
                  schema_block: str) -> Optional[dict]:
    try:
        out = ollama.chat(
            chat_model,
            [{"role": "system", "content": _CARD_PROMPT},
             {"role": "user", "content": f"Table:\n{schema_block}\n\nJSON:"}],
            options={"temperature": 0.0, "num_predict": 768},
            format="json",
        )
        return extract_json(out)
    except Exception:
        return None


def _localize_card(card: dict, table_name: str) -> dict:
    """Strip a donor sheet's file-specific facts from a shared card so it can
    honestly describe a sibling. The card prompt tells the model to read
    ``as_of`` off the FILE NAME, so a reused card would otherwise assert the
    donor's snapshot date -- drop it (the caller re-derives it from this
    table's own name) and retarget any date written into the prose."""
    import copy
    import re as _re
    out = copy.deepcopy(card) if isinstance(card, dict) else {}
    out.pop("as_of", None)
    mine = filename_asof(table_name)
    date_re = _re.compile(r"\b\d{1,2}[.\-/]\d{1,2}[.\-/]\d{4}\b")

    def _fix(s):
        if not isinstance(s, str) or not date_re.search(s):
            return s
        if mine:
            return date_re.sub(mine, s)
        # no date of our own -> drop the clause rather than assert a wrong one
        return _re.sub(r"\s*\(?\bas at\b[^.,;)]*\)?", "", s, flags=_re.I).strip()

    if out.get("purpose"):
        out["purpose"] = _fix(out["purpose"])
    if out.get("caveats"):
        out["caveats"] = [_fix(c) for c in out["caveats"]]
    return out


def build_cards(ollama, chat_model: str, model, dataframes: dict,
                existing: Optional[dict] = None, progress=None) -> dict:
    """Returns the full persistable structure {table: {schema_hash, card,
    notes}}, regenerating ONLY tables whose schema hash changed (hand edits
    with an unchanged hash are preserved verbatim).

    Sibling sheets -- same columns, same dtypes, hence the same schema_hash --
    share ONE LLM call between them: N monthly snapshots of one report cost a
    single card instead of N near-identical ones. Each sibling is still
    sanitized against its OWN data and gets its OWN as_of from its file name,
    so sharing never smuggles the donor's facts across."""
    from jarvisman.semantics.semantic_model import render_for_prompt
    store = dict(existing or {})
    if not cfg.TABLE_CARDS:
        return store

    todo: dict = {}                    # schema_hash -> [table names needing one]
    for name, df in dataframes.items():
        h = schema_hash(df)
        prev = store.get(name)
        if prev and prev.get("schema_hash") == h:
            continue                   # unchanged (or hand-edited): keep as is
        todo.setdefault(h, []).append(name)

    # A persisted card of the SAME schema is a free donor: reuse it rather
    # than paying for a new call (e.g. adding August to an indexed year).
    cache: dict = {}                   # schema_hash -> (raw_card, donor_name)
    for name, entry in store.items():
        h = entry.get("schema_hash")
        if h in todo and h not in cache and entry.get("card"):
            cache[h] = (entry["card"], name)

    for h, names in todo.items():
        for name in names:
            df = dataframes[name]
            if h in cache:
                raw, donor = cache[h]
            else:
                if progress:
                    progress(f"Understanding table {name} ...")
                block = render_for_prompt(model, [name])
                raw, donor = (generate_card(ollama, chat_model, name, block)
                              or {}), name
                cache[h] = (raw, donor)
            shared = donor != name
            src = _localize_card(raw, name) if shared else raw
            card, notes = sanitize_card(src, model.tables.get(name), df)
            fdate = filename_asof(name)
            if fdate and not card.get("as_of"):
                card["as_of"] = fdate
                if not card.get("table_kind"):
                    card["table_kind"] = "snapshot"
                notes.append(f"as_of {fdate} taken from the file name")
            if shared:
                notes.append(f"card reused from '{donor}' (identical schema; "
                             "0 extra LLM calls)")
            store[name] = {"schema_hash": h, "card": card, "notes": notes}
    return store


# --------------------------------------------------------------------------- #
# Persistence                                                                 #
# --------------------------------------------------------------------------- #
def cards_path(index_dir: str) -> str:
    return os.path.join(index_dir, _FILE)


def save_cards(index_dir: str, store: dict) -> None:
    try:
        os.makedirs(index_dir, exist_ok=True)
        with open(cards_path(index_dir), "w", encoding="utf-8") as f:
            json.dump({"version": 1, "tables": store}, f,
                      ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_cards(index_dir: str) -> dict:
    try:
        with open(cards_path(index_dir), encoding="utf-8") as f:
            data = json.load(f)
        return data.get("tables") or {}
    except Exception:
        return {}


def active_cards(store: dict, dataframes: dict, model) -> dict:
    """{table: clean_card} for current tables, RE-validated on load so a
    hand-edited card can never smuggle a claim past reality."""
    out = {}
    for name, entry in (store or {}).items():
        if name not in dataframes:
            continue
        card, _ = sanitize_card(entry.get("card") or {},
                                model.tables.get(name) if model else None,
                                dataframes[name])
        if card.get("purpose") or card.get("validity") \
                or card.get("entity_key") or card.get("as_of") \
                or card.get("table_kind"):
            out[name] = card
    return out