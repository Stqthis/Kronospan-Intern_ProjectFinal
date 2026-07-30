"""Question grounding: connect what the user SAID to what the data HAS.

Runs before any LLM call and costs no LLM call itself. Produces an Evidence
object: (1) value hits -- question terms verified to exist as cell values,
with their exact table/column/spelling; (2) column candidates -- question
terms matched to column names/meanings; (3) years mentioned; (4) a relevance
ranking of tables. The evidence is injected into the planner prompt (and the
codegen fallback prompt), so the model chooses among VERIFIED options instead
of guessing -- this is what prevents 'Croatian companies' from being bound to
a company-name column when CROATIA lives in COUNTRY.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Optional

from jarvisman import config as cfg
from jarvisman.semantics.semantic_model import SemanticModel
from jarvisman.semantics.value_index import ValueIndex, ValueHit, norm_text, tokenize_unicode

_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

# Minimal bilingual stopword list: only function words, never content words.
_STOP = {
    # English
    "the", "of", "a", "an", "is", "are", "in", "for", "to", "and", "as", "at",
    "all", "what", "which", "who", "how", "much", "many", "please", "me",
    "give", "show", "with", "on", "by", "this", "that", "was", "were", "be",
    "from", "per", "their", "its", "do", "does", "did", "list", "find",
    # Greek (normalised: accents stripped, casefolded)
    "και", "το", "τα", "του", "της", "των", "τον", "την", "η", "ο", "οι",
    "ειναι", "για", "με", "στο", "στη", "στην", "στον", "στα", "στις", "σε",
    "να", "τι", "ποιο", "ποια", "ποιος", "ποσο", "ποσα", "απο", "ως", "ανα",
    "μου", "δειξε", "ολα", "ολες", "ολοι", "καθε",
}


@dataclass
class ColumnCandidate:
    term: str
    table: str
    column: str
    role: str
    score: float
    via: str          # 'name' | 'meaning'


_MONTHS = {m: i + 1 for i, m in enumerate(
    ("january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"))}
_ASAT_CUE_RE = re.compile(
    r"\b(as at|as of|by the end of|end of|until|till|on)\b", re.IGNORECASE)
_DMY_RE = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-]((?:19|20)\d{2})\b")
_MONTH_YEAR_RE = re.compile(
    r"\b(" + "|".join(_MONTHS) + r")\b(?:\s+of)?\s+((?:19|20)\d{2})",
    re.IGNORECASE)


def _last_day(year: int, month: int) -> str:
    import calendar
    return f"{calendar.monthrange(year, month)[1]:02d}.{month:02d}.{year}"


def find_asat_date(question: str):
    """'by the end of december of 2023' -> ('31.12.2023', True). Returns
    (dd.mm.yyyy, is_asat) or None. is_asat means validity-interval semantics
    were requested (a cue word), not just a date mention."""
    cue = bool(_ASAT_CUE_RE.search(question))
    m = _DMY_RE.search(question)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{d:02d}.{mo:02d}.{y}", cue
    m = _MONTH_YEAR_RE.search(question)
    if m:
        return _last_day(int(m.group(2)), _MONTHS[m.group(1).lower()]), cue
    return None


@dataclass
class Evidence:
    question: str
    value_hits: list = field(default_factory=list)        # [ValueHit]
    column_candidates: list = field(default_factory=list)  # [ColumnCandidate]
    years: list = field(default_factory=list)
    asat: object = None        # ('31.12.2023', cue_present) or None              # [int]
    table_ranking: list = field(default_factory=list)      # [(score, name)]

    def ranked_tables(self, max_n: int) -> list[str]:
        return [n for s, n in self.table_ranking if s > 0][:max_n] or \
               [n for _, n in self.table_ranking][:max_n]

    # ------------------------------------------------------------------ #
    def to_prompt_block(self, max_hits: int = 8, max_cands: int = 10) -> str:
        lines: list[str] = []
        if self.value_hits:
            lines.append("Verified value matches (these values EXIST in the data; "
                         "filter on the column shown, with the value AS WRITTEN here):")
            for h in self.value_hits[:max_hits]:
                lines.append(
                    f"  - question term '{h.matched_text}' = value '{h.display}' in "
                    f"column '{h.column}' of table '{h.table}' ({h.count} rows)"
                )
            spellings = sorted({h.display for h in self.value_hits[:max_hits]
                                if h.kind in ("value_in_question",
                                              "value_token")})
            if spellings:
                lines.append("  If you filter on any of these values, "
                             "copy the spelling EXACTLY: "
                             + ", ".join(repr(s) for s in spellings[:8]))
            if self.asat and self.asat[1]:
                lines.append(f"  AS-AT DATE {self.asat[0]}: IF the table has "
                             "start/end validity columns (see schema), filter "
                             "START <= date AND (END >= date OR END empty = "
                             "active, \"or_null\": true). If it has NO such "
                             "columns, the date is the snapshot date of the "
                             "file itself: do NOT invent date columns or add "
                             "any date filter.")
            multi: dict = {}
            for h in self.value_hits[:max_hits]:
                multi.setdefault(h.matched_text, set()).add(h.column)
            for text, cols in multi.items():
                if len(cols) >= 2:
                    lines.append(
                        f"  NOTE: '{text}' exists in MULTIPLE columns "
                        f"({', '.join(sorted(cols))}) -- bind the filter to the "
                        "ONE column whose meaning matches the question (e.g. a "
                        "company's own country vs its bank's country).")
        if self.column_candidates:
            lines.append("Likely column matches for question terms:")
            seen = set()
            n = 0
            for c in sorted(self.column_candidates, key=lambda x: -x.score):
                k = (c.term, c.table, c.column)
                if k in seen:
                    continue
                seen.add(k)
                lines.append(f"  - '{c.term}' ~ column '{c.column}' "
                             f"[{c.role}] in '{c.table}' (matched via {c.via})")
                n += 1
                if n >= max_cands:
                    break
        if self.years:
            lines.append("Years mentioned in the question: "
                         + ", ".join(str(y) for y in self.years))
        if not lines:
            return ""
        return "Evidence gathered from the actual data:\n" + "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
def _content_terms(question: str) -> list[str]:
    toks = [t for t in tokenize_unicode(question)
            if len(t) >= 3 and t not in _STOP and not t.isdigit()]
    # unigrams + bigrams (bigrams catch 'amount eur', 'net revenue')
    bigrams = [f"{a} {b}" for a, b in zip(toks, toks[1:])]
    return toks + bigrams


def _match_columns(term: str, model: SemanticModel) -> list[ColumnCandidate]:
    out: list[ColumnCandidate] = []
    tn = norm_text(term)
    if not tn:
        return out
    for tname, tprof in model.tables.items():
        for col in tprof.columns:
            if col.role == "empty":
                continue
            cn = norm_text(col.name)
            score = 0.0
            via = ""
            if tn == cn:
                score, via = 6.0, "name"
            elif (tn in cn or cn in tn) and min(len(tn), len(cn)) >= 3 \
                    and min(len(tn), len(cn)) / max(len(tn), len(cn)) >= 0.45:
                score, via = 3.5, "name"
            else:
                ratio = difflib.SequenceMatcher(None, tn, cn).ratio()
                if ratio >= 0.84:
                    score, via = 3.0 * ratio, "name"
            if score == 0.0 and col.meaning:
                mn = norm_text(col.meaning)
                if tn in mn.split() or (len(tn) >= 5 and tn in mn):
                    score, via = 2.0, "meaning"
            if score > 0:
                # measures/dimensions are what questions usually reference
                if col.role in ("measure", "dimension", "date", "year"):
                    score += 0.5
                out.append(ColumnCandidate(term, tname, col.name, col.role, score, via))
    return out


def _suppress_descriptor_noise(hits: list, question: str, model=None,
                                vindex=None) -> list:
    """A common question word ('director', 'company') frequently occurs inside
    the descriptive cells of a free-text column (REMARK_DIR holds 'Director',
    'Managing Director', 'Non-Executive Director'). Those are weak filter
    candidates -- the user wrote 'director' as a role word, not a value to
    filter on. When the SAME question matched a distinctive NAME (a
    person/company identifier such as 'Koutouvas, Athanasios'), the role-word
    hits must not compete with it: the model was choosing the role word and
    ignoring the name.

    Signal = how many DISTINCT stored values in the hit's own column contain
    the matched word, fully data-driven. A descriptor word appears across many
    distinct values of its column ('director' in Director / Managing Director /
    Non-Executive Director); a name word appears in essentially one value. When
    a name-style hit exists, drop the multi-value descriptor hits on OTHER
    columns. No column names or domain words are hardcoded."""
    if not hits or vindex is None:
        return hits
    from jarvisman.semantics.value_index import norm_text as _nt, tokenize_unicode as _tk

    domains = getattr(vindex, "domains", {}) or {}

    def _distinct_values_with_word(h):
        """count distinct stored values in this hit's column whose tokens
        include the matched word."""
        mt = _nt(getattr(h, "matched_text", ""))
        dom = domains.get((h.table, h.column)) or {}
        n = 0
        for nv in dom:
            if mt in set(_tk(nv)):
                n += 1
        return n or 1

    # name hit: matched a multi-token value, and the matched word is specific
    # (it belongs to just ONE distinct value of its column)
    name_words = set()
    for h in hits:
        mt = _nt(getattr(h, "matched_text", ""))
        if len(mt) >= 4 and len(_tk(h.norm)) >= 2                 and _distinct_values_with_word(h) <= 1:
            name_words.add(mt)
    if not name_words:
        return hits

    name_cols = {h.column for h in hits
                 if _nt(getattr(h, "matched_text", "")) in name_words}
    kept = []
    for h in hits:
        mt = _nt(getattr(h, "matched_text", ""))
        if mt in name_words:
            kept.append(h)
            continue
        # descriptor: the matched word spans MULTIPLE distinct values of a
        # column that is NOT where the name lives -> role word, drop it
        if h.column not in name_cols and len(_tk(mt)) == 1                 and _distinct_values_with_word(h) >= 2:
            continue
        kept.append(h)
    return kept or hits


def ground(question: str, model: SemanticModel, vindex: ValueIndex) -> Evidence:
    asat = find_asat_date(question)
    ev = Evidence(question=question, asat=asat)
    if model is None or not model.tables:
        return ev

    ev.years = sorted({int(y) for y in _YEAR_RE.findall(question or "")})
    ev.value_hits = vindex.find_in_question(question) if vindex else []
    ev.value_hits = _suppress_descriptor_noise(ev.value_hits, question, model, vindex)

    seen_terms = set()
    for term in _content_terms(question):
        if term in seen_terms:
            continue
        seen_terms.add(term)
        ev.column_candidates.extend(_match_columns(term, model))

    # ---- table ranking ------------------------------------------------- #
    # PRINCIPLE: choose the table(s) by whether the table CAN ANSWER the
    # question -- does it carry the columns the question is about, does its
    # subject/name fit -- BEFORE looking at where a value happens to sit. A
    # value match is a FILTER signal; letting it dominate means a coincidental
    # cell match in the wrong sheet decides the table. So capability
    # (column/meaning fit + subject/name) is the primary score; value presence
    # is a secondary tie-breaker (capped), never the lead term.
    scores: dict[str, float] = {n: 0.0 for n in model.table_names()}
    qn = norm_text(question)

    # (1) PRIMARY: column/meaning fit -- how well the question's terms map to
    # this table's columns (this is "can this table answer it?").
    cap: dict[str, float] = {n: 0.0 for n in model.table_names()}
    for c in ev.column_candidates:
        cap[c.table] = cap.get(c.table, 0.0) + min(c.score, 6.0)
    for n in model.table_names():
        scores[n] += min(cap[n], 12.0)            # capability leads

    # (2) PRIMARY: subject/name fit -- the sheet/file name appears in the
    # question ("the group company sheet", "directors").
    for n in model.table_names():
        base = norm_text(n.split(":")[-1])
        if base and len(base) >= 3 and base in qn:
            scores[n] += 4.0

    # (3) SECONDARY tie-breaker: a verified value lives here. Useful (a
    # question naming CROATIA is likelier about the table holding CROATIA) but
    # CAPPED low so it refines, not dominates, the capability ranking.
    vbump: dict[str, float] = {}
    for h in ev.value_hits:
        w = 1.5 if h.kind == "value_in_question" else 0.6
        vbump[h.table] = vbump.get(h.table, 0.0) + w
    for n, b in vbump.items():
        scores[n] = scores.get(n, 0.0) + min(b, 3.0)   # cap the value signal

    ev.table_ranking = sorted(((s, n) for n, s in scores.items()),
                              key=lambda x: -x[0])
    return ev


def column_word_tokens(model) -> set:
    from jarvisman.semantics.value_index import norm_text as _nt, tokenize_unicode as _tk
    toks = set()
    for t in (model.tables or {}).values():
        for cp in t.columns:
            toks.update(_tk(_nt(cp.name)))
    return toks


def coverage_anchors(evidence: Evidence, model=None) -> list:
    """The grounded conditions a plan is NOT allowed to drop: each distinct
    verified value (minus aggregation/stop words), plus the as-at date when
    interval semantics were requested. Returned as (label, satisfiers) where
    satisfiers are the normalized strings any one of which must appear among
    the plan's filter values."""
    from jarvisman.semantics.value_index import norm_text as _nt
    colwords = column_word_tokens(model) if model is not None else set()
    out, seen = [], set()
    for h in evidence.value_hits:
        mt = _nt(h.matched_text)
        if mt in _AGG_SKIP or mt in _STOP or mt in seen:
            continue
        # 'eur' in "amount eur" names the COLUMN 'AMOUNT EUR', not a currency
        # filter: a value hit that is a token of a column name must never
        # become a required condition (manager-bench find)
        if mt in colwords:
            continue
        # ONLY a whole value found verbatim in the question is a condition
        # the user stated. A token hit ('funds' ~ 'Alpha Funds Ltd') is
        # EVIDENCE for the planner, never a requirement -- treating it as one
        # demoted correct plans to the slow codegen tier (real-usage bug).
        if h.kind != "value_in_question" or h.score < 10.0 or len(mt) < 4:
            continue
        seen.add(mt)
        out.append((h.display, {_nt(h.display), mt}))
        if len(out) >= 2:
            break
    if evidence.asat and evidence.asat[1]:
        d = evidence.asat[0]
        out.append((f"as-at {d}", {d, d.replace(".", "-"),
                                   "-".join(reversed(d.split(".")))}))
    return out


# --------------------------------------------------------------------------- #
# Ambiguity detection (Phase B)                                               #
# --------------------------------------------------------------------------- #
_AGG_SKIP = {"total", "sum", "average", "mean", "count", "max", "min",
             "maximum", "minimum", "median", "highest", "lowest", "amount"}


def _find_typo_options(question: str, evidence: Evidence,
                       model: SemanticModel, vindex) -> Optional[dict]:
    """The user knows WHAT they are searching for, not the schema. When a
    question term matches no verified value but is CLOSE to one
    ('panaghwtes' ~ 'Panagiotis', ratio 0.70), ask 'did you mean ...?' with
    the real values as clickable options. Gated on the question having NO
    verified value hit at all -- if any real anchor exists, ordinary terms
    like 'companies' must not trigger fuzzy noise. A wrong ask costs one
    ignorable chip; a silent wrong guess costs a wrong number."""
    import difflib
    from jarvisman.semantics.value_index import norm_text, tokenize_unicode
    if vindex is None or not getattr(vindex, "entries", None):
        return None
    # Per-WORD gating (not all-or-nothing): a question usually has SOME
    # incidental value hit ('deposits' ~ stored 'Deposit'), which used to
    # suppress the did-you-mean for a genuinely misspelled term in the SAME
    # question ('Croacia'). Instead, skip only the words that themselves
    # grounded to a value; still scan the rest. A term that is itself a
    # verified value is never a typo.
    anchored = set()
    for h in evidence.value_hits:
        mt = norm_text(h.matched_text)
        if mt in _AGG_SKIP or mt in _STOP:
            continue
        anchored.add(mt)
        # the matched token may be a morphological variant of the question
        # word ('deposits' produced the hit 'deposit'); mark the question
        # words that CONTAIN or are contained by it as anchored too
        for qt in tokenize_unicode(question):
            qn2 = norm_text(qt)
            if len(qn2) >= 4 and (qn2 in mt or mt in qn2):
                anchored.add(qn2)
    col_tokens = set()
    for t in (model.tables or {}).values():
        for cp in t.columns:
            col_tokens.update(tokenize_unicode(norm_text(cp.name)))
    keys = list(vindex.entries.keys())
    tindex = getattr(vindex, "token_index", {}) or {}
    # words the user CAPITALIZED (typos of names are capitalized; common
    # nouns like 'loans'/'general' are not) -- gates the token-fuzzy branch
    import re as _re
    caps = {norm_text(m) for m in
            _re.findall(r"\b[A-Z\u0391-\u03a9][\w\-]*", question or "")}
    best = None
    for tok in tokenize_unicode(question):
        tn = norm_text(tok)
        if len(tn) < 4 or tn.isdigit() or tn in _STOP or tn in _AGG_SKIP \
                or tn in col_tokens \
                or tn in anchored or tn in vindex.entries:
            continue
        # a word that IS a token of some stored value is real vocabulary of
        # the data ('loan' in 'Long-Term Loan', 'CR' in 'Kronospan CR ...'),
        # never a typo -- also cover the trivial plural/singular variant
        if tn in tindex or tn.rstrip("s") in tindex or (tn + "s") in tindex:
            continue
        # length-aware confidence: short words need very high similarity (a
        # loose match turned 'sort'->'South'); long words may use a lower bar
        # since random collisions are unlikely ('panaghwtes'->'Panagiotis').
        # Always require a shared 2-char prefix ('athanasios' !-> 'Banasino').
        window = [k for k in keys if abs(len(k) - len(tn)) <= 2
                  and k[:2] == tn[:2]]
        _cut = 0.68 if len(tn) >= 9 else 0.78
        close = difflib.get_close_matches(tn, window, n=3, cutoff=_cut)
        # 'Caixbank' cannot reach the multi-token value 'CaixaBank, S.A.'
        # through whole-value matching -- fuzzy over value TOKENS too, but
        # ONLY for capitalized (name-like) words: real-usage transcript shows
        # this branch turned 'general'->'Generale' and 'each'->'Lech' into
        # nonsense asks when unrestricted
        tok_close = []
        if tn in caps:
            _tok_window = [k for k in tindex if k[:2] == tn[:2]]
            _tcut = 0.74 if len(tn) >= 9 else 0.84
            tok_close = difflib.get_close_matches(tn, _tok_window, n=2,
                                                  cutoff=_tcut)
        options, seen = [], set()
        def _add(table, col, display):
            if display in seen:
                return
            # a summary-row LABEL is an artifact, never a real option
            if norm_text(str(display)) in _AGG_SKIP | {"subtotal"}:
                return
            # a giant free-text cell (comments) is never a usable option
            if len(str(display)) > 60:
                return
            seen.add(display)
            options.append({"label": f"{display} ({col})",
                            "display": display,
                            "table": table, "column": col})
        for k in close:
            if k == tn:
                continue
            for (table, col, _count, display) in vindex.entries[k][:2]:
                _add(table, col, display)
        for k in tok_close:
            if k == tn:
                continue
            for (nv, locs) in (vindex.token_index.get(k) or [])[:2]:
                for (table, col, _count, display) in locs[:1]:
                    _add(table, col, display)
        if options:
            cand = (close + tok_close)[0]
            ratio = difflib.SequenceMatcher(None, tn, cand).ratio()
            if best is None or ratio > best[0]:
                best = (ratio, tok, options[:4])
    if best is None:
        return None
    _ratio, term, options = best
    # final confidence gate (length-aware): a wrong suggestion is worse than
    # none, so short words must clear a high bar; long words may be lower
    _term_norm = norm_text(term)
    _min = 0.68 if len(_term_norm) >= 9 else 0.78
    if _ratio < _min:
        return None
    return {"kind": "value_typo", "term": term,
            "question": f"I couldn't find '{term}' in the data -- "
                        "did you mean one of these?",
            "options": options}


def _find_acronym_options(question: str, evidence: Evidence,
                          model: SemanticModel, vindex) -> Optional[dict]:
    """'RBI' -> 'Raiffeisen Bank International': a short ALL-CAPS token, typed
    that way in the original question, whose letters are the initials of the
    leading tokens of a stored multi-token value. No string-similarity
    algorithm can bridge an acronym (zero shared tokens), so without this the
    question silently falls to codegen and returns zero rows. Deterministic;
    offered as a did-you-mean option, never auto-bound. Skipped when the
    token IS a stored value (EUR), a column-name token (USH), or a stopword.
    """
    if vindex is None or not getattr(vindex, "entries", None):
        return None
    caps = re.findall(r"\b[A-Z]{2,6}\b", question or "")
    if not caps:
        return None
    col_tokens = set()
    for t in (model.tables or {}).values():
        for cp in t.columns:
            col_tokens.update(tokenize_unicode(norm_text(cp.name)))
    tindex = getattr(vindex, "token_index", {}) or {}
    # complete token set of all stored values (token_index drops tokens
    # shorter than 3 chars, which let 'CR' through). Punctuation-aware
    # (norm keys keep commas: 'cr,' != 'cr'), built once and cached on the
    # index object.
    all_value_tokens = getattr(vindex, "_all_tokens", None)
    if all_value_tokens is None:
        all_value_tokens = set()
        for _nv in vindex.entries:
            all_value_tokens.update(tokenize_unicode(_nv))
        try:
            vindex._all_tokens = all_value_tokens
        except AttributeError as exc:
            cfg.dbg("grounding cache _all_tokens", exc)
    for tok in caps:
        tn = norm_text(tok)
        if len(tn) < 2 or tn in _STOP or tn in _AGG_SKIP or tn in col_tokens:
            continue
        if tn in vindex.entries or tn in tindex or tn in all_value_tokens:
            continue   # a real stored value OR a token of one ('CR' inside
                       # 'Kronospan CR, spol s r.o.') -- not an acronym
        options, seen = [], set()
        for nv, locs in vindex.entries.items():
            toks = tokenize_unicode(nv)
            # initials of the LEADING tokens, so 'RBI' still matches
            # 'Raiffeisen Bank International AG' (trailing legal suffix)
            if len(toks) < max(2, len(tn)):
                continue
            if "".join(t[0] for t in toks[:len(tn)]) != tn:
                continue
            for (table, col, _count, display) in locs[:1]:
                if display in seen:
                    continue
                seen.add(display)
                options.append({"label": str(display), "display": str(display),
                                "table": table, "column": col})
            if len(options) >= 4:
                break
        if options:
            return {"kind": "value_typo", "term": tok,
                    "question": f"I couldn't find '{tok}' in the data -- "
                                "did you mean one of these?",
                    "options": options[:4]}
    return None


def find_ambiguity(evidence: Evidence, model: SemanticModel,
                   question: str = "", vindex=None) -> Optional[dict]:
    """Clarification policy (final, from real usage): the user knows the DATA
    they are searching for, never the schema. The ONLY clarification this app
    asks is value-level: "I couldn't find 'X' -- did you mean <real value>?".
    Column choice -- including a value existing in several columns (CROATIA in
    COUNTRY vs COUNTRY OF BANK) and abstract terms ("funds") -- is exactly the
    planner's judgement call: it receives every hit plus an explicit
    multi-column NOTE in the evidence, and its choice is disclosed in the
    provenance ("filter COUNTRY == 'CROATIA': N rows"), so a wrong pick is
    visible, never silent."""
    if not cfg.CLARIFY_ENABLED:
        return None
    if question:
        try:
            from jarvisman.semantics.ambiguity import find_interpretation_ambiguity
            amb = find_interpretation_ambiguity(question, evidence, model,
                                                vindex)
            # auto_interpretation is applied earlier in the reasoner (it needs
            # no user input); only a real 'interpretation' is a clarify here
            if amb and amb.get("kind") == "interpretation":
                return amb
        except Exception as exc:
            # feature-isolation boundary: a whole optional subsystem. Stay
            # broad so an unexpected error here NEVER breaks answering -- but
            # make it visible instead of vanishing.
            cfg.dbg("grounding.find_interpretation_ambiguity", exc)
        try:
            amb = _find_acronym_options(question, evidence, model, vindex)
            if amb:
                return amb
        except Exception as exc:
            cfg.dbg("grounding._find_acronym_options", exc)
    return _find_typo_options(question, evidence, model, vindex) \
        if question else None
