"""Tier 0.5 -- deterministic planning for the simplest question shapes.

A large share of real questions are "<agg> of <measure> [for <verified value>]
[per <dimension>]". When grounding pins exactly one measure, at most one
verified value anchor, and the aggregation word is explicit, no judgement is
required -- the plan is mechanical. Building it in code skips the LLM call
entirely (the biggest latency win there is).

Strictly conservative: ANY ambiguity -- two plausible measures, a value in
several columns, an unrecognised aggregation, an as-at date, a comparison,
"share"/"same", multiple verified values -- returns None and the question
goes to the LLM planner. A wrong silent plan is far worse than a 1-call plan.
"""

from __future__ import annotations

import re
from typing import Optional

from jarvisman import config as cfg
from jarvisman.semantics.grounding import Evidence
from jarvisman.semantics.value_index import norm_text, tokenize_unicode

_AGG = {
    "sum": "sum", "total": "sum", "turnover": None,    # 'turnover' is a noun, skip
    "average": "mean", "avg": "mean", "mean": "mean",
    "median": "median",
    "max": "max", "maximum": "max", "highest": "max", "largest": "max",
    "min": "min", "minimum": "min", "lowest": "min", "smallest": "min",
    "count": "count", "how many": "count",
    "nunique": "nunique", "distinct": "nunique",
}
_GROUP_RE = re.compile(r"\b(per|by|for each)\s+([a-z][\w ]{1,40})", re.IGNORECASE)
_COMPLEX = ("share", "same", "compare", "change", "growth", "versus", " vs ",
            "percent", "% of", "ratio", "as at", "as of", "end of", "until",
            "active", "between", "difference")


def _agg_in(qn: str) -> Optional[str]:
    if "how many" in qn:
        return "count"
    for tok in tokenize_unicode(qn):
        fn = _AGG.get(tok)
        if fn:
            return fn
    return None


def try_plan(question: str, evidence: Evidence, model, validator) -> Optional[dict]:
    if not cfg.TIER05_ENABLED:
        return None
    qn = norm_text(question)
    if any(c in qn for c in _COMPLEX):
        return None
    if evidence.asat:
        return None
    fn = _agg_in(qn)
    if not fn:
        return None

    # exactly one verified value anchor (or none). Aggregation words that
    # happen to match a summary-row LABEL ('Total' in a name column) are NOT
    # anchors -- the same phantom-hit class handled elsewhere.
    from jarvisman.semantics.grounding import column_word_tokens
    colwords = column_word_tokens(model)
    raw_anchors = [h for h in evidence.value_hits
                   if h.kind in ("value_in_question", "value_token")
                   and norm_text(h.matched_text) not in _AGG
                   and h.matched_text.lower() not in ("total", "sum",
                                                      "subtotal", "average",
                                                      "count")]
    anchors = [h for h in raw_anchors
               if norm_text(h.matched_text) not in colwords]
    # a column-word value ('EUR') with NO other anchor is ambiguous: it may
    # really be a currency filter -- defer instead of an unfiltered sum
    if not anchors and raw_anchors:
        return None
    distinct_vals = {norm_text(h.matched_text) for h in anchors}
    # 'cyprus' inside 'bank of cyprus' is the SAME intent: drop matched texts
    # subsumed by a longer one before counting (red-team find)
    distinct_vals = {a for a in distinct_vals
                     if not any(a != b and a in b for b in distinct_vals)}
    if len(distinct_vals) > 1:
        return None
    anchors = [h for h in anchors
               if norm_text(h.matched_text) in distinct_vals] or anchors
    # the same value can live in SEVERAL tables ('CYPRUS' as a country of
    # banks AND of companies): pick the instance whose table also holds the
    # question's measure; if that is still ambiguous, defer (manager-bench)
    anchor = None
    if anchors:
        tables = []
        for h in sorted(anchors, key=lambda h: -h.score):
            if h.table not in tables:
                tables.append(h.table)
        if len(tables) == 1:
            anchor = max(anchors, key=lambda h: h.score)
        else:
            mcands = [c for c in evidence.column_candidates
                      if c.role == "measure"]
            with_measure = [tb for tb in tables
                            if any(c.table == tb for c in mcands)]
            if len(with_measure) == 1:
                anchor = max((h for h in anchors
                              if h.table == with_measure[0]),
                             key=lambda h: h.score)
            else:
                return None
    if anchor is not None:
        # the same value in SEVERAL COLUMNS of the chosen table (COUNTRY vs
        # COUNTRY OF BANK) is the planner's judgement call -- defer so the
        # multi-column evidence NOTE can do its job
        cols_here = {h.column for h in anchors
                     if h.table == anchor.table
                     and norm_text(h.matched_text)
                     == norm_text(anchor.matched_text)}
        if len(cols_here) > 1:
            return None
    table = anchor.table if anchor else None

    # one measure column, unambiguously -- judged WITHIN the anchor's table:
    # in a multi-table world, people.xlsx's 'AMOUNT' must not shadow
    # companies.xlsx's 'AMOUNT EUR' (manager-bench find)
    measures = [c for c in evidence.column_candidates if c.role == "measure"]
    if anchor is not None:
        same = [c for c in measures if c.table == anchor.table]
        if same:
            measures = same
    if fn != "count" and not measures and anchor is not None:
        # punctuation-heavy names ('TOTAL CO. B/CE') evade the term matcher;
        # when the question literally CONTAINS exactly one measure column's
        # normalized name, that is the measure (red-team find)
        tprof = model.tables.get(anchor.table)
        contained = [c for c in (tprof.columns if tprof else [])
                     if c.role == "measure" and len(norm_text(c.name)) >= 4
                     and norm_text(c.name) in qn]
        if len(contained) == 1:
            class _C:  # minimal candidate shim
                pass
            shim = _C()
            shim.column, shim.table, shim.score = contained[0].name, anchor.table, 9.0
            measures = [shim]
    if fn != "count":
        if not measures:
            return None
        top = sorted(measures, key=lambda c: -c.score)
        if len(top) >= 2 and top[1].score / max(top[0].score, 1e-9) >= 0.8 \
                and top[0].score < 6.0:
            return None                      # genuinely ambiguous measure
        measure = top[0]
        table = table or measure.table
        if anchor and measure.table != anchor.table:
            return None                      # cross-table -> let the LLM join
    else:
        measure = None
        if table is None:
            if len(model.tables) != 1:
                return None
            table = next(iter(model.tables))

    # optional single group-by, resolved to a real dimension
    group_by = []
    gm = _GROUP_RE.search(qn)
    if gm:
        want = gm.group(2).strip()
        col = validator.resolve_column(table, want) if validator else None
        if col is None:
            for c in evidence.column_candidates:
                if c.table == table and c.role in ("dimension", "year") \
                        and norm_text(c.column).find(norm_text(want)) >= 0:
                    col = c.column
                    break
        if col is None:
            return None                      # asked to group but unclear -> LLM
        group_by = [col]

    # an unexplained content token (>=4 chars, not an agg/group/stop word, not
    # a column token, not the matched anchor) suggests a misspelled value ->
    # defer to grounding's "did you mean" path instead of answering unfiltered
    matched_tokens = set()
    for h in anchors:
        matched_tokens.update(tokenize_unicode(norm_text(h.matched_text)))
    col_tokens = set()
    for c in model.tables.get(table).columns if model.tables.get(table) else []:
        col_tokens.update(tokenize_unicode(norm_text(c.name)))
    skip = set(_AGG) | {"for", "of", "the", "all", "in", "per", "by", "each",
                        "company", "companies", "value", "values", "total",
                        "many", "much", "what", "which", "give", "show",
                        "rows", "records", "entries", "number", "numbers"}
    def _explained(tok: str) -> bool:
        if tok in skip or tok in col_tokens or tok in matched_tokens:
            return True
        # 'croatian' explains 'croatia' (and vice-versa) by containment
        for mt in matched_tokens:
            a, b = (tok, mt) if len(tok) <= len(mt) else (mt, tok)
            if len(a) >= 4 and a in b and len(a) / len(b) >= 0.6:
                return True
        return False

    for tok in tokenize_unicode(qn):
        if len(tok) < 4 or tok.isdigit():
            continue
        if not _explained(tok):
            return None     # unexplained word -> full pipeline (did-you-mean)

    filters = []
    if anchor:
        filters.append({"column": anchor.column, "op": "match",
                        "value": anchor.display})
    alias = (f"{fn}_{measure.column}" if measure else "count").replace(" ", "_").lower()
    return {
        "table": table, "join": None, "filters": filters,
        "group_by": group_by,
        "aggregations": [{"fn": fn,
                          "column": measure.column if measure else None,
                          "alias": alias}],
        "select": [], "sort": None, "limit": None,
        "requires_code": False, "clarification": None,
    }
