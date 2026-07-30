from __future__ import annotations

import os
import re
import unicodedata
import difflib
from dataclasses import dataclass, field

from jarvisman import config as cfg

_WS_RE = re.compile(r"\s+")


def norm_text(s) -> str:
    """The single normalisation used everywhere (index, validator, codegen)."""
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    # Greek final sigma: casefold keeps 'ς' as-is, so 'πωλήσεις' would not
    # match 'ΠΩΛΗΣΕΙΣ' (which casefolds to '...σ'). Fold ς -> σ explicitly.
    return _WS_RE.sub(" ", s.casefold().replace("\u03c2", "\u03c3")).strip()


_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


# Per-token similarity required for a TYPED name token to be accepted as a
# (misspelled) match of a STORED name token, in resolve_phrase_fuzzy. Only a
# same-first-letter pair is ever compared, and the whole resolution still
# requires EXACTLY ONE stored value to qualify -- so this bar governs recall,
# not safety. It must stay <= 0.667: the documented two-typo case
# 'Spuros spirou' -> 'Spyrou, Spiros' hinges on 'spuros' vs 'spyrou', whose
# SequenceMatcher ratio is exactly 0.667. A stricter bar (the old in-code
# comment said 0.78) silently breaks that case. Override per dataset with
# RAG_FUZZY_RATIO if a corpus needs tighter or looser matching.
_FUZZY_TOKEN_RATIO = float(os.environ.get("RAG_FUZZY_RATIO", "0.6"))


def tokenize_unicode(text: str) -> list[str]:
    """Unicode-aware tokenizer (Greek included), on normalised text."""
    return _TOKEN_RE.findall(norm_text(text))


@dataclass
class ValueHit:
    """One verified occurrence of a question term in the data."""
    norm: str           # normalised value as stored in the index
    display: str        # an original spelling of the value (for prompts)
    table: str
    column: str
    count: int          # rows in that column carrying the value
    kind: str           # 'value_in_question' | 'token_fuzzy'
    matched_text: str   # the question fragment that matched
    score: float


@dataclass
class ValueIndex:
    # norm value -> list of (table, column, count, display)
    entries: dict = field(default_factory=dict)
    # (table, column) -> {norm value -> [all original spellings]} for
    # full-domain checks; the validator resolves filters to these LITERAL
    # spellings so compiled code can use a plain exact isin().
    domains: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Build                                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def build(dataframes: dict, max_cardinality: int = None) -> "ValueIndex":
        max_card = max_cardinality or cfg.VALUE_INDEX_MAX_CARDINALITY
        idx = ValueIndex()
        for tname, df in (dataframes or {}).items():
            if df is None or len(df) == 0:
                continue
            for col in df.columns:
                series = df[col]
                if hasattr(series, "columns"):        # duplicated label
                    continue
                kind = getattr(series.dtype, "kind", "O")
                if kind in "iufcMmb":                 # numbers/dates: not indexed
                    continue
                try:
                    vc = series.dropna().astype(str).value_counts()
                except (TypeError, ValueError) as exc:
                    cfg.dbg(f"value_index.build value_counts [{tname}:{col}]", exc)
                    continue
                if len(vc) == 0 or len(vc) > max_card:
                    continue
                dom: dict[str, list] = {}
                for val, cnt in vc.items():
                    nv = norm_text(val)
                    if not nv:
                        continue
                    dom.setdefault(nv, []).append(str(val))
                    idx.entries.setdefault(nv, []).append(
                        (tname, str(col), int(cnt), str(val))
                    )
                if dom:
                    idx.domains[(tname, str(col))] = dom
        return idx

    # ------------------------------------------------------------------ #
    # Question matching                                                  #
    # ------------------------------------------------------------------ #
    @property
    def token_index(self) -> dict:
        """tok -> [(value_norm, locs)] for multi-token values, built lazily so
        every construction path (and any persisted instance) gets it."""
        ti = getattr(self, "_token_index", None)
        if ti is None:
            ti = {}
            for nv, locs in self.entries.items():
                toks = [t for t in tokenize_unicode(nv) if len(t) >= 4]
                if len(toks) < 2:
                    continue          # single-token values are already covered
                for t in toks:
                    bucket = ti.setdefault(t, [])
                    if len(bucket) < 4:
                        bucket.append((nv, locs))
            self._token_index = ti
        return ti

    def find_in_question(self, question: str, max_hits: int = 12) -> list[ValueHit]:
        qn = norm_text(question)
        if not qn:
            return []
        tokens = [t for t in _TOKEN_RE.findall(qn) if len(t) >= 3]
        hits: dict[tuple, ValueHit] = {}   # (norm, table, column) -> best hit

        def add(nv, locs, kind, matched, score):
            for table, column, count, display in locs:
                key = (nv, table, column)
                old = hits.get(key)
                if old is None or score > old.score:
                    hits[key] = ValueHit(nv, display, table, column, count,
                                         kind, matched, score)

        for nv, locs in self.entries.items():
            if len(nv) < 3:
                continue
            # whole value appears verbatim inside the question (multi-word ok)
            if nv in qn:
                add(nv, locs, "value_in_question", nv, 10.0 + min(len(nv), 30) / 10.0)
                continue
            # token-level: 'croatian' ~ 'croatia' (mutual-substring containment)
            for t in tokens:
                a, b = (t, nv) if len(t) <= len(nv) else (nv, t)
                if a in b and len(a) / len(b) >= 0.6:
                    add(nv, locs, "token_fuzzy", t, 5.0 + len(a) / len(b))
                    break

        # token sub-index: question token 'karanikolaos' surfaces the stored
        # multi-token value 'Karanikolaos,Panagiotis' even though neither
        # whole-value nor mutual-substring matching can bridge the length gap.
        for t in tokens:
            if len(t) < 4:
                continue
            for nv, locs in self.token_index.get(t, []):
                # below token_fuzzy (5.0+): a token hit surfaces the stored
                # spelling as evidence but must not outrank real matches,
                # pollute table ranking, or become a required condition
                add(nv, locs, "value_token", t, 4.0 + min(len(t), 20) / 20.0)
            for stored_t, bucket in self.token_index.items():
                if stored_t[:1] == t[:1] and difflib.SequenceMatcher(None, t, stored_t).ratio() >= 0.6:
                    for nv, locs in bucket:
                        add(nv, locs, "token_fuzzy_typo", t, 6.0 + difflib.SequenceMatcher(None, t, stored_t).ratio())

        out = sorted(hits.values(), key=lambda h: (-h.score, -h.count))
        # keep at most 3 hits per (table, column) so one column cannot flood
        per_col: dict[tuple, int] = {}
        kept: list[ValueHit] = []
        for h in out:
            k = (h.table, h.column)
            if per_col.get(k, 0) >= 3:
                continue
            per_col[k] = per_col.get(k, 0) + 1
            kept.append(h)
            if len(kept) >= max_hits:
                break
        return kept

    # ------------------------------------------------------------------ #
    # Domain access (used by the plan validator)                         #
    # ------------------------------------------------------------------ #
    def domain(self, table: str, column: str, df=None) -> dict:
        """{norm -> display} for a column. Falls back to computing it from the
        DataFrame (cached) when the column was too high-cardinality to index."""
        key = (table, str(column))
        dom = self.domains.get(key)
        if dom is not None:
            return dom
        if df is None:
            return {}
        try:
            vc = df[column].dropna().astype(str).value_counts()
        except (TypeError, ValueError, KeyError) as exc:
            cfg.dbg(f"value_index.domain [{table}:{column}]", exc)
            return {}
        dom = {}
        for val in vc.index[: cfg.VALUE_INDEX_MAX_CARDINALITY * 4]:
            nv = norm_text(val)
            if nv:
                dom.setdefault(nv, []).append(str(val))
        self.domains[key] = dom
        return dom


def resolve_phrase(vindex, phrase: str):
    """Deterministically map a free-typed phrase to the ONE stored value it
    token-matches ('Karanikolaos Panagiotis' -> 'Karanikolaos,Panagiotis').
    Returns the stored display string, or None when the phrase already exists
    verbatim, matches nothing, or matches MORE than one stored value (an
    ambiguous surname must never be auto-picked)."""
    pn = norm_text(phrase)
    if not pn or pn in vindex.entries:
        return None
    toks = [t for t in tokenize_unicode(pn) if len(t) >= 3]
    if not toks or not any(len(t) >= 4 for t in toks):
        return None
    seed = max(toks, key=len)
    cands = {}
    for nv, locs in (vindex.token_index.get(seed) or []):
        if set(toks) <= set(tokenize_unicode(nv)) and nv != pn:
            for (_t, _c, _n, display) in locs[:1]:
                cands[nv] = display
    if len(cands) != 1:
        return None
    return next(iter(cands.values()))


def resolve_phrase_fuzzy(vindex, phrase: str):
    import difflib
    pn = norm_text(phrase)
    if not pn or pn in vindex.entries:
        return None
    toks = [t for t in tokenize_unicode(pn) if len(t) >= 3]
    if len(toks) < 2:
        return None                 # need multi-token (a name) for this rule
    tindex = vindex.token_index or {}
    exact = [t for t in toks if t in tindex]
    # allow up to TWO misspelled tokens, but never resolve a single-typo'd
    # token alone (need the multi-token name structure for confidence)
    if len(toks) - len(exact) > 2:
        return None
    # candidate stored values come from scanning every stored multi-token
    # value and checking that EACH typed token is either an exact token of it
    # or a close (same-first-letter, ratio >= _FUZZY_TOKEN_RATIO) match to one
    # of its tokens, used at most once. One typo (the original case
    # 'Spiros Spurou') and the harder two-typo case ('Spuros spirou' ->
    # 'Spyrou, Spiros') are the same rule: two independent near-matches to ONE
    # stored name is a strong, low-false-positive signal. Still: resolve only
    # when EXACTLY ONE stored value qualifies -- never auto-pick among several.
    def _covers(nv_toks):
        remaining = list(nv_toks)
        for tt in toks:
            if tt in remaining:
                remaining.remove(tt); continue
            cand = None
            for r in remaining:
                if r and r[:1] == tt[:1] and \
                        difflib.SequenceMatcher(None, tt, r).ratio() \
                        >= _FUZZY_TOKEN_RATIO:
                    cand = r; break
            if cand is None:
                return False
            remaining.remove(cand)
        return True

    cands = {}
    # seed the scan from any exact token if present (fast path), else scan all
    seeds = exact if exact else toks
    scanned = set()
    pools = []
    for s in seeds:
        pools.extend(tindex.get(s, []))
    if not exact:
        # no exact token at all: scan the whole index (names are few)
        for nv, locs in vindex.entries.items():
            pools.append((nv, locs))
    for nv, locs in pools:
        if nv in scanned:
            continue
        scanned.add(nv)
        nv_toks = tokenize_unicode(nv)
        if len(nv_toks) < 2:
            continue
        if _covers(nv_toks):
            for (_t, _c, _n, display) in locs[:1]:
                cands[nv] = display
    if len(cands) != 1:
        return None                 # ambiguous or none -> never auto-pick
    return next(iter(cands.values()))


def resolve_code_literals(code: str, vindex, tables: dict):
    """Rewrite free-typed string literals inside GENERATED code to their
    verified stored spellings (unique matches only). Returns
    (new_code, [(old, new), ...]). Table keys and column names are never
    touched."""
    import ast as _ast
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return code, []
    skip = set(tables.keys())
    for df in tables.values():
        skip.update(str(c) for c in df.columns)
    subs = []
    for node in _ast.walk(tree):
        if not (isinstance(node, _ast.Constant) and isinstance(node.value, str)):
            continue
        s = node.value
        if len(s) < 4 or s in skip or ".xls" in s or ":" in s \
                or not any(ch.isalpha() for ch in s):
            continue
        d = resolve_phrase(vindex, s) or resolve_phrase_fuzzy(vindex, s)
        if d and d != s:
            subs.append((s, d))
    new_code = code
    for s, d in subs:
        for q in ("'", '"'):
            new_code = new_code.replace(q + s + q, repr(d))
    return new_code, subs
