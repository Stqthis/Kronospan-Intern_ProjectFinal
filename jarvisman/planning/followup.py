"""Follow-up resolution: "and for 2024?", "και για Κύπρο;", "the same per country".

A follow-up is the previous question with one slot changed. Since the previous
PLAN is stored in bound form (real table, real columns, verified values), the
change can usually be applied IN CODE: swap a year, swap a filter value, set a
group-by. When that succeeds, the rewritten plan goes straight to the compiler
-- zero LLM calls, sub-second answers, and the bindings are exactly as
verified as the original's.

Conservative by design: it only fires on explicit follow-up markers or on
fragments that are nothing but a new value/year, and any rewrite that fails
validation falls through to the normal planner path. A wrong silent rewrite
would be worse than a slow correct answer (accuracy beats speed).
"""

from __future__ import annotations

import copy
import re
from typing import Optional

from jarvisman import config as cfg
from jarvisman.semantics.grounding import ground
from jarvisman.semantics.semantic_model import SemanticModel
from jarvisman.semantics.value_index import ValueIndex, norm_text, tokenize_unicode

_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

# Explicit follow-up openers (normalised: accents stripped, casefolded)
_MARKERS = (
    "and ", "also ", "what about", "how about", "same ", "the same",
    "και ", "επισης", "το ιδιο", "ιδιο ", "τι γινεται", "για ",
)
_GROUP_MARKERS = ("per ", "by ", "ανα ", "καθε ", "for each")
_CHART_MARKERS = ("plot", "chart", "graph", "γραφημα", "διαγραμμα", "πιτα")

# aggregation keywords -> plan fn; a follow-up fragment naming a DIFFERENT
# aggregation than the previous plan is NOT a safe slot-swap (the user wants
# a different computation) -- bail to the planner instead of silently keeping
# the old fn. Accuracy beats speed.
_AGG_WORDS = {
    "sum": "sum", "total": "sum", "συνολο": "sum", "αθροισμα": "sum",
    "average": "mean", "mean": "mean", "μεσος": "mean", "μεσο": "mean",
    "median": "median", "διαμεσος": "median",
    "max": "max", "maximum": "max", "μεγιστο": "max", "highest": "max",
    "min": "min", "minimum": "min", "ελαχιστο": "min", "lowest": "min",
    "count": "count", "ποσα": "count", "ποσες": "count", "how many": "count",
}


def _conflicting_agg(qn: str, plan: dict) -> bool:
    fns = {a.get("fn") for a in (plan.get("aggregations") or [])}
    if not fns:
        return False
    for word, fn in _AGG_WORDS.items():
        if word in qn and fn not in fns:
            return True
    return False


def _is_followup_shaped(qn: str, n_tokens: int) -> bool:
    """Short fragment + explicit marker, or a bare year/value fragment."""
    if n_tokens > 9:
        return False
    if any(qn.startswith(m) or f" {m}" in f" {qn}" for m in _MARKERS):
        return True
    if any(m in qn for m in _GROUP_MARKERS):
        return True
    # a fragment that is essentially just a year ("2024?", "in 2024")
    stripped = _YEAR_RE.sub("", qn)
    return bool(_YEAR_RE.search(qn)) and len(tokenize_unicode(stripped)) <= 2


def _year_filter_index(plan: dict, model: SemanticModel, table: str) -> Optional[int]:
    tprof = model.tables.get(table.split("#")[0]) if model else None
    for i, f in enumerate(plan.get("filters") or []):
        col = f.get("column", "")
        if col == "YEAR":
            return i
        cp = tprof.column(col) if tprof else None
        if cp is not None and cp.role in ("year", "date"):
            return i
    return None


def _first_year_column(model: SemanticModel, table: str) -> Optional[str]:
    base = table.split("#")[0]
    tprof = model.tables.get(base) if model else None
    if tprof is None:
        return None
    if table.endswith("#long"):
        return "YEAR"
    for c in tprof.columns:
        if c.role == "year":
            return c.name
    for c in tprof.columns:
        if c.role == "date":
            return c.name
    return None


def rewrite(question: str, last_plan: Optional[dict],
            model: SemanticModel, vindex: ValueIndex) -> Optional[dict]:
    """Returns a rewritten bound-plan dict, or None when this is not a
    confident follow-up. ``last_plan`` is the previous plan as a dict with
    REAL table/column names and verified values."""
    if not cfg.FOLLOWUP_ENABLED or not last_plan or model is None or vindex is None:
        return None
    qn = norm_text(question)
    toks = tokenize_unicode(question)
    if not _is_followup_shaped(qn, len(toks)):
        return None

    plan = copy.deepcopy(last_plan)
    table = plan.get("table") or ""
    if not table:
        return None
    if _conflicting_agg(qn, plan):
        return None
    changed = False

    # ---- 1. a new year: replace the year condition (or add one) ----------
    years = _YEAR_RE.findall(question)
    if years:
        y = int(years[0])
        idx = _year_filter_index(plan, model, table)
        if idx is not None:
            plan["filters"][idx]["value"] = y
            plan["filters"][idx]["op"] = "eq"
            changed = True
        else:
            ycol = _first_year_column(model, table)
            if ycol:
                plan.setdefault("filters", []).append(
                    {"column": ycol, "op": "eq", "value": y})
                changed = True

    # ---- 2. a new verified value: swap it into the matching filter -------
    ev = ground(question, model, vindex)
    base = table.split("#")[0]
    same_table_hits = [h for h in ev.value_hits if h.table == base]
    if same_table_hits:
        h = max(same_table_hits, key=lambda x: x.score)
        filters = plan.setdefault("filters", [])
        for f in filters:
            if f.get("column") == h.column:
                f["value"], f["op"] = h.display, "match"
                changed = True
                break
        else:
            filters.append({"column": h.column, "op": "match", "value": h.display})
            changed = True

    # ---- 3. "per X" / "by X": set the group-by ---------------------------
    if any(m in qn for m in _GROUP_MARKERS) and plan.get("aggregations"):
        for c in ev.column_candidates:
            if c.table == base and c.role in ("dimension", "year", "date"):
                col = c.column if not table.endswith("#long") else c.column
                plan["group_by"] = [col]
                plan["sort"] = None
                plan["limit"] = None
                changed = True
                break
        else:
            if "year" in qn or "ετος" in qn or "ετη" in qn or "χρονια" in qn:
                ycol = _first_year_column(model, table)
                if ycol:
                    plan["group_by"] = [ycol]
                    # drop a fixed-year filter when grouping by year
                    idx = _year_filter_index(plan, model, table)
                    if idx is not None:
                        plan["filters"].pop(idx)
                    plan["sort"] = None
                    changed = True

    if not changed:
        return None
    plan["requires_code"] = False
    plan["clarification"] = None
    return plan


def wants_chart(question: str) -> bool:
    qn = norm_text(question)
    return any(m in qn for m in _CHART_MARKERS)
