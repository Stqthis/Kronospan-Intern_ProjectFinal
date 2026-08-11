"""Structured query plan: schema -> validation -> deterministic pandas code.

The LLM emits a small JSON plan (table, filters, group_by, aggregations,
sort, limit). Everything risky is then done by CODE, before execution:

  * columns are resolved against the real schema (normalised + fuzzy);
  * text filter values are resolved against the column's ACTUAL value domain
    -- the compiled filter is an exact isin() over the literal spellings found
    in the data, so case/space/accent differences can never miss;
  * a text value that does not exist in the bound column is looked up in the
    value index, and the filter is REBOUND to the column that actually holds
    it (the 'Croatian companies -> COUNTRY, not COMPANY NAME' fix);
  * joins are allowed only on relationships verified at profiling time;
  * Total/Subtotal rows detected in the data are excluded from aggregations;
  * numeric aggregation over a text-typed column is auto-coerced;
  * every step prints a 'PROV ...' row count, so an empty filter is
    diagnosable and the user gets an honest provenance trail.

Unresolvable problems become structured ValidationIssues with concrete
suggestions ("'Croacia' not found; closest values: CROATIA"), which the
planner feeds back to the model for ONE precise repair -- far more effective
than replaying a runtime traceback.

The emitted code uses only constructs accepted by sandbox.validate_code
(no imports, no underscore attributes, no forbidden names) and runs through
the same spawn-isolated executor as the codegen fallback.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from jarvisman import config as cfg
from jarvisman.semantics.semantic_model import SemanticModel, TableProfile
from jarvisman.semantics.value_index import ValueIndex, norm_text, tokenize_unicode

_OP_ALIASES = {"gte": "ge", "lte": "le", "greater": "gt", "less": "lt",
              "ge=": "ge", "le=": "le", "==": "eq", "!=": "ne",
              ">": "gt", ">=": "ge", "<": "lt", "<=": "le"}
_FILTER_OPS = {"match", "eq", "ne", "in", "contains", "gt", "ge", "lt", "le",
               "between", "isnull", "notnull"}
_AGG_FNS = {"sum", "mean", "median", "min", "max", "count", "nunique"}
_YEAR_VAL_RE = re.compile(r"^(19|20)\d{2}$")


def _needs_coerce(cp) -> bool:
    """A 'number' column whose storage dtype is not numeric (numbers kept as
    text) must be coerced before any arithmetic -- sum() on strings would
    CONCATENATE. parse_rate alone is not enough: it can be 1.0."""
    return bool(cp) and cp.semantic_type == "number" \
        and getattr(cp, "dtype_kind", "O") not in "iufb"


def _norm_col(s: str) -> str:
    return re.sub(r"[^\w]", "", norm_text(s))


# --------------------------------------------------------------------------- #
# Plan schema                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class Filter:
    column: str
    op: str
    value: Any = None
    or_null: bool = False      # cmp passes ALSO when the cell is empty/null
                               # (as-at semantics: END empty = still active)


@dataclass
class Aggregation:
    fn: str
    column: Optional[str] = None
    alias: Optional[str] = None


@dataclass
class JoinSpec:
    right_table: str
    left_column: str
    right_column: str


@dataclass
class SortSpec:
    by: str
    desc: bool = True


@dataclass
class QueryPlan:
    table: str = ""
    join: Optional[JoinSpec] = None
    filters: list = field(default_factory=list)        # [Filter]
    group_by: list = field(default_factory=list)       # [str]
    aggregations: list = field(default_factory=list)   # [Aggregation]
    select: list = field(default_factory=list)         # [str] (row listing)
    sort: Optional[SortSpec] = None
    limit: Optional[int] = None
    requires_code: bool = False
    clarification: Optional[str] = None
    # ---- plan schema v2 (each optional; validated like everything else) ----
    having: list = field(default_factory=list)     # [Filter] on agg aliases
    intent: str = ""                               # optional model summary
    union: list = field(default_factory=list)      # extra aligned tables
    compare: Optional[dict] = None                 # {left_table, right_table,
                                                   #  key, measures, labels,
                                                   #  left/right_filters}
    derived: list = field(default_factory=list)    # [{alias, kind, num, den}]

    @staticmethod
    def from_dict(d: dict) -> "QueryPlan":
        def s(x):
            return str(x).strip() if x is not None else None

        filters = []
        for f in d.get("filters") or []:
            if isinstance(f, dict) and f.get("column") and f.get("op"):
                filters.append(Filter(or_null=bool(f.get("or_null")),
                                      column=s(f["column"]),
                                      op=str(f["op"]).strip().lower(),
                                      value=f.get("value")))
        aggs = []
        for a in d.get("aggregations") or []:
            if isinstance(a, dict) and a.get("fn"):
                aggs.append(Aggregation(fn=str(a["fn"]).strip().lower(),
                                        column=s(a.get("column")),
                                        alias=s(a.get("alias"))))
        join = None
        j = d.get("join")
        if isinstance(j, dict) and j.get("right_table"):
            join = JoinSpec(right_table=s(j["right_table"]),
                            left_column=s(j.get("left_column") or ""),
                            right_column=s(j.get("right_column") or ""))
        sort = None
        so = d.get("sort")
        if isinstance(so, dict) and so.get("by"):
            sort = SortSpec(by=s(so["by"]), desc=bool(so.get("desc", True)))
        limit = d.get("limit")
        try:
            limit = int(limit) if limit is not None else None
        except (TypeError, ValueError):
            limit = None
        intent = s(d.get("intent_summary") or d.get("intent") or "")[:160]
        having = []
        for f in d.get("having") or []:
            if isinstance(f, dict) and f.get("column") and f.get("op"):
                # normalise exactly like plan.filters: lowercase + alias map.
                # Without this, an op of '>' or 'lte' silently compiled as
                # '>' (the old default), flipping the comparison.
                hop = str(f["op"]).strip().lower()
                hop = _OP_ALIASES.get(hop, hop)
                if hop in ("gt", "ge", "lt", "le", "eq", "ne"):
                    having.append(Filter(column=s(f["column"]), op=hop,
                                         value=f.get("value")))
        derived = []
        for dv in d.get("derived") or []:
            if isinstance(dv, dict) and dv.get("alias") and dv.get("kind"):
                derived.append({"alias": s(dv["alias"]),
                                "kind": s(dv["kind"]),
                                "num": s(dv.get("num")),
                                "den": s(dv.get("den"))})
        cmpd = d.get("compare") if isinstance(d.get("compare"), dict) else None
        union = [s(u) for u in (d.get("union") or []) if u]
        return QueryPlan(having=having, derived=derived, compare=cmpd,
                         intent=intent,
                         union=union,
            table=s(d.get("table")) or "",
            join=join, filters=filters,
            group_by=[s(g) for g in (d.get("group_by") or []) if s(g)],
            aggregations=aggs,
            select=[s(c) for c in (d.get("select") or []) if s(c)],
            sort=sort, limit=limit,
            requires_code=bool(d.get("requires_code", False)),
            clarification=s(d.get("clarification")),
        )


@dataclass
class ValidationIssue:
    kind: str       # unknown_table | unknown_column | unknown_value | bad_op |
                    # bad_agg | bad_join | empty_plan
    message: str


@dataclass
class ColumnProfileLite:
    """Minimal profile stand-in for virtual long-view columns."""
    name: str
    semantic_type: str
    role: str
    dtype_kind: str
    numeric_parse_rate: float = 1.0
    date_format: Optional[str] = None


@dataclass
class CodeBundle:
    code: str
    explain: list                  # human-readable provenance of decisions
    tables_used: list              # dfs keys the code touches


# --------------------------------------------------------------------------- #
# Validation / resolution                                                     #
# --------------------------------------------------------------------------- #
class PlanValidator:
    """Resolves a QueryPlan against the semantic model + real tables. Mutates
    a COPY of the plan into fully-bound form and reports issues it could not
    fix deterministically."""

    LONG_SUFFIX = "#long"
    LONG_COLS = ("YEAR", "MEASURE", "VALUE")

    def __init__(self, model: SemanticModel, dataframes: dict, vindex: ValueIndex):
        self.model = model
        self.dfs = dataframes or {}
        self.vindex = vindex
        self._colmap_cache: dict[str, dict] = {}

    # ---- virtual long-view plumbing ----------------------------------- #
    def is_long(self, table: str) -> bool:
        return isinstance(table, str) and table.endswith(self.LONG_SUFFIX)

    def long_base(self, table: str) -> str:
        return table[: -len(self.LONG_SUFFIX)] if self.is_long(table) else table

    def _long_view(self, table: str):
        base = self.long_base(table)
        t = self.model.tables.get(base) if self.model else None
        lv = getattr(t, "long_view", None) if t else None
        return lv if lv else None

    def columns_of(self, table: str) -> list:
        """Real or virtual column list for a table name."""
        if self.is_long(table):
            lv = self._long_view(table)
            if lv:
                return list(lv["id_vars"]) + list(self.LONG_COLS)
            return []
        df = self.dfs.get(table)
        return [str(c) for c in df.columns] if df is not None else []

    # ---- name resolution ------------------------------------------------ #
    def resolve_table(self, name: str) -> Optional[str]:
        if not name:
            return None
        compact = name.replace(" ", "")
        if compact.lower().endswith(self.LONG_SUFFIX):
            base = self.resolve_table(compact[: -len(self.LONG_SUFFIX)])
            if base and self._long_view(base + self.LONG_SUFFIX):
                return base + self.LONG_SUFFIX
            return base
        if name in self.dfs:
            return name
        nn = _norm_col(name)
        best = None
        for t in self.dfs:
            nt = _norm_col(t)
            if nt == nn:
                return t
            if nn and (nn in nt or nt in nn):
                best = best or t
            # model may answer with the sheet part only ('Sheet1')
            tail = _norm_col(t.split(":")[-1])
            if nn and tail and (nn == tail or nn in tail or tail in nn):
                best = best or t
        return best

    def _colmap(self, table: str) -> dict:
        m = self._colmap_cache.get(table)
        if m is None:
            df = self.dfs.get(table)
            m = {_norm_col(str(c)): str(c) for c in (df.columns if df is not None else [])}
            self._colmap_cache[table] = m
        return m

    def resolve_column(self, table: str, name: str) -> Optional[str]:
        if not name:
            return None
        if self.is_long(table):
            cols = self.columns_of(table)
            if name in cols:
                return name
            nn = _norm_col(name)
            for c in cols:
                if _norm_col(c) == nn:
                    return c
            for c in cols:
                nc = _norm_col(c)
                if nn and nc and (nn in nc or nc in nn) \
                        and min(len(nn), len(nc)) / max(len(nn), len(nc)) >= 0.5:
                    return c
            return None
        df = self.dfs.get(table)
        if df is None:
            return None
        real = {str(c) for c in df.columns}
        if name in real:
            return name
        nm = self._colmap(table)
        nn = _norm_col(name)
        if nn in nm:
            return nm[nn]
        for nc, c in nm.items():
            if nn and nc and (nn in nc or nc in nn) \
                    and min(len(nn), len(nc)) / max(len(nn), len(nc)) >= 0.5:
                return c
        cand = difflib.get_close_matches(nn, list(nm.keys()), n=1, cutoff=0.84)
        return nm[cand[0]] if cand else None

    def _profile(self, table: str) -> Optional[TableProfile]:
        return self.model.tables.get(table) if self.model else None

    def _col_profile(self, table: str, column: str):
        if self.is_long(table):
            base = self.long_base(table)
            if column == "YEAR":
                return ColumnProfileLite("YEAR", "number", "year", "i")
            if column == "MEASURE":
                return ColumnProfileLite("MEASURE", "text", "dimension", "O")
            if column == "VALUE":
                # already coerced by the melt block, so dtype_kind numeric
                return ColumnProfileLite("VALUE", "number", "measure", "f")
            t = self.model.tables.get(base) if self.model else None
            return t.column(column) if t else None
        t = self._profile(table)
        return t.column(column) if t else None

    _NAME_TOKENS = {"name", "title", "description"}
    _CODE_TOKENS = {"code", "id", "key", "nr", "no", "number"}

    def _companion_code(self, table: str, col: str) -> Optional[str]:
        """For a name-like column, the matching code column of the same table:
        'COMPANY NAME' -> 'COMPANY CODE' (shared stem + a code token), or a
        bare 'COMPANY' -> 'COMPANY CODE'."""
        from jarvisman.semantics.value_index import norm_text, tokenize_unicode
        ct = set(tokenize_unicode(norm_text(col)))
        stem = ct - self._NAME_TOKENS
        if ct & self._CODE_TOKENS:
            return None                      # it already IS a code column
        for other in self.columns_of(table):
            if other == col:
                continue
            ot = set(tokenize_unicode(norm_text(other)))
            if (ot & self._CODE_TOKENS) and (ot & stem if stem else False):
                return other
        return None

    def _augment_companions(self, plan: QueryPlan, table: str,
                            notes: list) -> None:
        """When the result shows a name-like column, include its code column.
        Selects: appended right after the name. Group-bys: only when the
        (name, code) pairing is verified 1:1 in the actual data -- adding a
        non-1:1 key would change the aggregation, which is never acceptable."""
        if not cfg.COMPANION_CODES:
            return
        for col in list(plan.select):
            code = self._companion_code(table, col)
            if code and code not in plan.select:
                plan.select.insert(plan.select.index(col) + 1, code)
                notes.append(f"included '{code}' alongside '{col}'")
        if plan.group_by and plan.aggregations:
            base = self.long_base(table)
            df = self.dfs.get(base)
            for col in list(plan.group_by):
                code = self._companion_code(table, col)
                if not code or code in plan.group_by or df is None:
                    continue
                if col not in df.columns or code not in df.columns:
                    continue
                try:
                    pairs = df[[col, code]].drop_duplicates()
                    if len(pairs) == df[col].dropna().drop_duplicates().shape[0]:
                        plan.group_by.insert(plan.group_by.index(col) + 1, code)
                        notes.append(f"grouping also by '{code}' "
                                     f"(verified 1:1 with '{col}')")
                except (TypeError, ValueError, KeyError, IndexError) as exc:
                    cfg.dbg(f"query_plan companion 1:1 [{col}->{code}]", exc)

    def domain_for(self, table: str, column: str) -> dict:
        """Value domain ({norm: [spellings]}) for real OR virtual columns."""
        if self.is_long(table):
            lv = self._long_view(table)
            base = self.long_base(table)
            if lv and column == "MEASURE":
                from jarvisman.semantics.value_index import norm_text as _nt
                dom: dict = {}
                for spec in lv["value_cols"].values():
                    dom.setdefault(_nt(spec["measure"]), []).append(spec["measure"])
                return {k: sorted(set(v)) for k, v in dom.items()}
            if lv and column in lv["id_vars"]:
                return self.vindex.domain(base, column, self.dfs.get(base)) \
                    if self.vindex else {}
            return {}
        return self.vindex.domain(table, column, self.dfs.get(table)) \
            if self.vindex else {}

    # ---- value resolution ------------------------------------------------ #
    def _resolve_in_column(self, value, dom):
        """Match a typed value against ONE column's distinct values (dom: {norm
        -> [originals]}), order-independent and tolerant of up to two
        misspelled tokens. Returns [single_original] when exactly one value
        qualifies, [a, b, ...] when several plausible candidates exist (->
        clarify), or None when nothing is close enough. Names only (multi-token
        or a single >=4-char token); never guesses on short/common words."""
        import difflib
        from jarvisman.semantics.value_index import norm_text, tokenize_unicode
        nv = norm_text(value)
        if not nv or not dom:
            return None
        toks = [t for t in tokenize_unicode(nv) if len(t) >= 3]
        if not toks or not any(len(t) >= 4 for t in toks):
            return None

        def _covers(cand_tokens):
            remaining = list(cand_tokens)
            for tt in toks:
                if tt in remaining:
                    remaining.remove(tt); continue
                hit = None
                for r in remaining:
                    if r and r[:1] == tt[:1] and \
                            difflib.SequenceMatcher(None, tt, r).ratio() >= 0.78:
                        hit = r; break
                if hit is None:
                    return False
                remaining.remove(hit)
            return True

        matches = {}
        for k, originals in dom.items():
            ktoks = tokenize_unicode(k)
            if len(ktoks) < max(1, len(toks)):
                continue
            if _covers(ktoks):
                matches[k] = originals[0]
        if not matches:
            return None
        return list(matches.values())

    def resolve_text_value(self, table: str, column: str, value) -> tuple[list, str, list]:
        """Returns (matched_original_spellings, how, suggestions).
        how: 'exact' | 'partial' | 'none'."""
        dom = self.domain_for(table, column)
        nv = norm_text(value)
        if not nv:
            return [], "none", []
        if nv in dom:
            return list(dom[nv]), "exact", []
        # Token-set match: same words, different order/punctuation.
        # 'Panagiotis Karanikolaos' resolves to stored 'Karanikolaos,Panagiotis'
        # -- the model writes names the human way; the data decides the format.
        vtok = frozenset(t for t in tokenize_unicode(nv) if len(t) >= 2)
        if len(vtok) >= 2:
            ts = [k for k in dom
                  if frozenset(t for t in tokenize_unicode(k) if len(t) >= 2) == vtok]
            if len(ts) == 1:
                return list(dom[ts[0]]), "token_set", []
            if len(ts) > 1:
                return [], "none", [dom[k][0] for k in ts[:3]]
        # Unique-subset match: the plan's tokens are contained in exactly ONE
        # stored value ('Karanikolaos' -> 'Karanikolaos,Panagiotis'). Multiple
        # candidates never resolve silently -- they become suggestions.
        if vtok and all(len(t) >= 3 for t in vtok) and any(len(t) >= 4 for t in vtok):
            sub = [k for k in dom if vtok <= frozenset(tokenize_unicode(k))]
            if len(sub) == 1:
                return list(dom[sub[0]]), "token_subset", []
            if len(sub) > 1:
                return [], "none", [dom[k][0] for k in sub[:3]]
        partial: list = []
        for k, originals in dom.items():
            a, b = (nv, k) if len(nv) <= len(k) else (k, nv)
            if len(a) >= 3 and a in b and len(a) / len(b) >= 0.6:
                partial.extend(originals)
        if partial:
            return partial, "partial", []
        sugg_keys = difflib.get_close_matches(nv, list(dom.keys()), n=3, cutoff=0.6)
        suggestions = [dom[k][0] for k in sugg_keys]
        return [], "none", suggestions

    def rebind_value_column(self, table: str, value) -> Optional[tuple[str, str]]:
        """The Croatia fix: the value does not exist in the bound column --
        find where it ACTUALLY lives. Same-table hits only (cross-table
        rebinding is reported as a suggestion instead, never done silently)."""
        if not self.vindex:
            return None
        nv = norm_text(value)
        hits = []
        for key, locs in ((nv, self.vindex.entries.get(nv)),):
            if locs:
                hits.extend(locs)
        if not hits:  # token-level fallback ('croatian' -> 'croatia')
            for k, locs in self.vindex.entries.items():
                a, b = (nv, k) if len(nv) <= len(k) else (k, nv)
                if len(a) >= 3 and a in b and len(a) / len(b) >= 0.6:
                    hits.extend(locs)
        if not hits:  # token-set fallback ('First Last' -> 'Last,First')
            vtok = frozenset(t for t in tokenize_unicode(nv) if len(t) >= 2)
            if len(vtok) >= 2:
                for k, locs in self.vindex.entries.items():
                    if frozenset(t for t in tokenize_unicode(k)
                                 if len(t) >= 2) == vtok:
                        hits.extend(locs)
        same = [(t, c, n, d) for (t, c, n, d) in hits if t == table]
        if same:
            t, c, n, d = max(same, key=lambda x: x[2])
            return c, d
        return None

    # ---- main entry ------------------------------------------------------ #
    def attach_context(self, cards: dict = None, coverage_anchors: list = None):
        self._cards = cards or {}
        self._coverage_anchors = coverage_anchors or []
        return self

    def validate(self, plan: QueryPlan) -> tuple[QueryPlan, list, list]:
        if plan.compare:                      # compare is self-validating in
            return plan, [], []               # _compile_compare; skip here
        """Returns (bound_plan, issues, notes). The bound plan has real table
        and column names and filter values resolved to literal spellings
        (stashed on each Filter as ``_resolved``)."""
        issues: list[ValidationIssue] = []
        notes: list[str] = []

        table = self.resolve_table(plan.table) or (
            next(iter(self.dfs)) if len(self.dfs) == 1 else None)
        if table is None:
            issues.append(ValidationIssue(
                "unknown_table",
                f"table '{plan.table}' not found; available: {list(self.dfs)}"))
            return plan, issues, notes
        if table != plan.table:
            notes.append(f"table '{plan.table}' resolved to '{table}'")
        plan.table = table
        if self.is_long(table) and plan.join:
            notes.append("joins are not supported on the unpivoted view; "
                         "join dropped")
            plan.join = None

        # ---- join
        right = None
        if plan.join:
            right = self.resolve_table(plan.join.right_table)
            lcol = self.resolve_column(table, plan.join.left_column) if right else None
            rcol = self.resolve_column(right, plan.join.right_column) if right else None
            ok = False
            if right and lcol and rcol:
                for r in (self.model.relationships if self.model else []):
                    pair = {(r.child_table, r.child_column, r.parent_table, r.parent_column),
                            (r.parent_table, r.parent_column, r.child_table, r.child_column)}
                    if (table, lcol, right, rcol) in pair:
                        ok = True
                        break
            if not ok:
                issues.append(ValidationIssue(
                    "bad_join",
                    f"join {plan.join.right_table} on "
                    f"{plan.join.left_column}={plan.join.right_column} is not a "
                    "verified relationship; allowed joins: " + (", ".join(
                        f"{r.child_table}.{r.child_column}->"
                        f"{r.parent_table}.{r.parent_column}"
                        for r in (self.model.relationships or [])) or "none")))
                plan.join = None
            else:
                plan.join = JoinSpec(right, lcol, rcol)
                notes.append(f"join verified: {table}.{lcol} = {right}.{rcol}")

        def resolve_any(name: str) -> Optional[str]:
            c = self.resolve_column(table, name)
            if c is None and plan.join:
                rc = self.resolve_column(plan.join.right_table, name)
                if rc is not None:
                    return rc  # collision handling done at compile time
            return c

        # ---- filters
        for f in plan.filters:
            f.op = _OP_ALIASES.get(f.op, f.op)
            if f.op not in _FILTER_OPS:
                issues.append(ValidationIssue("bad_op", f"unknown filter op '{f.op}'"))
                continue
            col = resolve_any(f.column)
            if col is None:
                # maybe the model put a VALUE where the column should be
                rb = self.rebind_value_column(table, f.column) if f.value in (None, "") else None
                if rb:
                    col, display = rb
                    f.value, f.op = display, "match"
                    notes.append(f"'{f.column}' is a value, not a column; "
                                 f"filtering column '{col}' for it")
                else:
                    # An INVENTED validity filter on a snapshot table: the
                    # planner proposed a date filter on a column that does
                    # not exist (e.g. START_DATE/END_DATE) while the table
                    # has NO validity pair and the question is as-at. The
                    # snapshot IS the date -- drop the filter with a note
                    # instead of rejecting the whole plan to codegen.
                    asat_asked = any(str(lbl).startswith("as-at ")
                                     for lbl, _s in
                                     (self._coverage_anchors or []))
                    val_is_date = False
                    if f.value is not None:
                        try:
                            iso = self._parse_dates(
                                [f.value], None)
                            val_is_date = bool(iso)
                        except (TypeError, ValueError) as exc:
                            cfg.dbg("query_plan as-at date probe", exc)
                            val_is_date = False
                    if asat_asked and val_is_date \
                            and self._validity_pair(table) is None:
                        notes.append(
                            f"filter on '{f.column}' dropped: the column "
                            f"does not exist and this table has no validity "
                            f"columns -- it is the snapshot for the asked "
                            f"date, so no date filter applies")
                        plan.filters = [x for x in plan.filters if x is not f]
                        continue
                    cols = self.columns_of(table)
                    sugg = difflib.get_close_matches(
                        _norm_col(f.column), [_norm_col(c) for c in cols], n=2, cutoff=0.5)
                    issues.append(ValidationIssue(
                        "unknown_column",
                        f"filter column '{f.column}' not found in '{table}'; "
                        f"available columns: {cols}"))
                    continue
            if col != f.column:
                notes.append(f"column '{f.column}' resolved to '{col}'")
            f.column = col
            self._resolve_filter(plan, f, issues, notes)

        # ---- group_by / aggregations / select / sort
        # 'SOURCE' is synthesised by a UNION at compile time, so it will not
        # resolve against the base table yet -- preserve it explicitly.
        def _resolve_gb(g):
            real = resolve_any(g)
            if real:
                return real
            if plan.union and str(g).upper() in ("SOURCE", "SOURCE_FILE"):
                has_real = any(str(c).upper() == "SOURCE"
                               for c in self.columns_of(table))
                return "SOURCE_FILE" if has_real else "SOURCE"
            return None
        plan.group_by = [c for c in (_resolve_gb(g) for g in plan.group_by) if c]
        good_aggs = []
        for a in plan.aggregations:
            if a.fn not in _AGG_FNS:
                issues.append(ValidationIssue("bad_agg", f"unknown aggregation '{a.fn}'"))
                continue
            if a.fn != "count" or a.column:
                col = resolve_any(a.column or "")
                if col is None:
                    issues.append(ValidationIssue(
                        "unknown_column",
                        f"aggregation column '{a.column}' not found in '{table}'; "
                        f"available columns: {self.columns_of(table)}"))
                    continue
                cp = self._col_profile(plan.table, col) or (
                    self._col_profile(plan.join.right_table, col) if plan.join else None)
                if a.fn in ("sum", "mean", "median") and cp and cp.semantic_type not in ("number",):
                    issues.append(ValidationIssue(
                        "bad_agg", f"cannot {a.fn} non-numeric column '{col}' "
                                   f"({cp.semantic_type}); pick a numeric column"))
                    continue
                a.column = col
            good_aggs.append(a)
        plan.aggregations = good_aggs
        plan.select = [c for c in (resolve_any(s) for s in plan.select) if c]

        if plan.sort:
            aliases = {self._alias(a) for a in plan.aggregations}
            valid_targets = aliases | set(plan.group_by) | set(plan.select) | \
                set(self.columns_of(table))
            by = plan.sort.by if plan.sort.by in valid_targets else None
            if by is None:
                by = resolve_any(plan.sort.by)
            if by is None:
                cand = difflib.get_close_matches(
                    _norm_col(plan.sort.by), [_norm_col(a) for a in aliases], n=1, cutoff=0.6)
                by = next((a for a in aliases if _norm_col(a) == cand[0]), None) if cand else None
            if by is None:
                notes.append(f"sort key '{plan.sort.by}' not resolvable; sort dropped")
                plan.sort = None
            else:
                plan.sort.by = by

        self._augment_companions(plan, table, notes)

        # ---- having: targets must be agg aliases or group-by columns -------
        if plan.having:
            valid = {self._alias(a) for a in plan.aggregations} | set(plan.group_by)
            kept = []
            for hf in plan.having:
                if hf.column in valid:
                    kept.append(hf)
                else:
                    notes.append(f"having target '{hf.column}' is not an "
                                 "aggregation alias or group key; dropped")
            plan.having = kept

        # ---- union: align extra tables by normalized column names ----------
        if plan.union:
            aligned = []
            base_cols = {_norm_col(c) for c in self.columns_of(table)}
            for u in plan.union:
                ru = self.resolve_table(u)
                if not ru or ru == table:
                    notes.append(f"union table '{u}' unresolved; skipped")
                    continue
                ucols = {_norm_col(c) for c in self.columns_of(ru)}
                overlap = base_cols & ucols
                if len(overlap) < max(2, len(base_cols) // 2):
                    notes.append(f"union table '{ru}' shares too few columns "
                                 f"({len(overlap)}); skipped")
                    continue
                miss = base_cols - ucols
                if miss:
                    notes.append(f"union with '{ru}': {len(miss)} columns "
                                 "absent there will be empty")
                aligned.append(ru)
            plan.union = aligned

        # ---- as-at: a bare comparison on a validity END column must also
        #             keep active rows (empty end) -- repair or_null ---------
        self._enforce_asat(plan, table, notes)

        # ---- condition coverage: no grounded condition silently dropped ----
        if cfg.CONDITION_COVERAGE and self._coverage_anchors:
            present = []
            for f in plan.filters:
                r = getattr(f, "_resolved", {}) or {}
                for v in (r.get("values") or []):
                    present.append(norm_text(str(v)))
                if r.get("value") is not None:
                    present.append(norm_text(str(r.get("value"))))
                if f.value is not None:
                    present.append(norm_text(str(f.value)))
            present_set = set(present)
            for label, satisfiers in self._coverage_anchors:
                if satisfiers & present_set:
                    continue
                if str(label).startswith("as-at "):
                    asked = str(label)[6:].strip()
                    card = (self._cards or {}).get(self.long_base(table)) or {}
                    # The card carries the table's ESSENCE: a snapshot table
                    # satisfies its own as-of date with NO filter, and a
                    # DIFFERENT date is honestly flagged as not covered.
                    if card.get("as_of"):
                        if asked == card["as_of"]:
                            notes.append(f"'{label}': this table IS the "
                                         f"{card['as_of']} snapshot; no date "
                                         "filter needed")
                        else:
                            notes.append(f"WARNING: you asked as at {asked} "
                                         f"but this file is the "
                                         f"{card['as_of']} snapshot -- the "
                                         "answer reflects "
                                         f"{card['as_of']}, not {asked}")
                        continue
                    # No card knowledge: a table without validity columns is
                    # treated as a snapshot of the asked date (fallback).
                    if self._validity_pair(table) is None:
                        notes.append(f"'{label}': treated as the snapshot "
                                     "date of the file (no validity columns)")
                        continue
                issues.append(ValidationIssue(
                    "dropped_condition",
                    f"the question's condition '{label}' is missing from "
                    "the plan's filters; every stated condition must apply"))

        if not plan.aggregations and not plan.select and not plan.filters \
                and not plan.group_by and not plan.union:
            issues.append(ValidationIssue(
                "empty_plan", "the plan has no filters, aggregations or selected "
                              "columns; it would just dump the table"))
        return plan, issues, notes

    def _enforce_asat(self, plan: QueryPlan, table: str, notes: list) -> None:
        """When the data has a validity interval and the question is as-at,
        enforce the CORRECT interval shape: START <= date AND (END >= date OR
        END empty). This is one STRUCTURED repair, not a band-aid: a bare
        'END == date' (which matches nobody active) is rewritten to 'END >=
        date OR null', and the missing 'START <= date' side is added. The
        as-at date is read from the grounded coverage anchors so the repair
        works even when the planner put the date on the wrong column."""
        pair = self._validity_pair(table)
        if not pair:
            return
        start, end = pair
        asat = None
        for label, _sat in (self._coverage_anchors or []):
            if str(label).startswith("as-at "):
                asat = str(label)[6:].strip()
                break

        end_f = next((f for f in plan.filters if f.column == end), None)
        start_f = next((f for f in plan.filters if f.column == start), None)

        # repair the END comparison: any op on END in an as-at context means
        # "still valid at the date" -> END >= date OR active(empty)
        if end_f is not None:
            if end_f.op in ("eq", "gt", "le", "lt"):
                end_f.op = "ge"
                r = getattr(end_f, "_resolved", None)
                if isinstance(r, dict) and r.get("kind") == "date_cmp":
                    r["op"] = "ge"
                notes.append(f"as-at: '{end}' comparison corrected to '>= "
                             "date OR still active' (an exact-date match on an "
                             "end column would have returned no active rows)")
            if not getattr(end_f, "or_null", False):
                end_f.or_null = True
                r = getattr(end_f, "_resolved", None)
                if isinstance(r, dict) and r.get("kind") == "date_cmp":
                    r["or_null"] = True

        # ensure the START <= date side exists (the other half of the interval)
        if asat and start_f is None and end_f is not None:
            nf = Filter(column=start, op="le", value=asat)
            self._resolve_filter(plan, nf, [], notes)
            if getattr(nf, "_resolved", None) is not None:
                plan.filters.append(nf)
                notes.append(f"as-at: added the '{start} <= {asat}' half of "
                             "the validity interval")

    def _validity_pair(self, table: str):
        """(start, end) for the table -- from a table card when present, else
        a name-pattern heuristic over date-roled columns."""
        card = (self._cards or {}).get(self.long_base(table))
        if card:
            for vp in (card.get("validity") or []):
                s = self.resolve_column(table, vp.get("start"))
                e = self.resolve_column(table, vp.get("end"))
                if s and e:
                    return (s, e)
        cols = self.columns_of(table)
        starts = [c for c in cols if "start" in _norm_col(c)
                  or "from" in _norm_col(c) or "appoint" in _norm_col(c)]
        ends = [c for c in cols if "end" in _norm_col(c)
                or _norm_col(c).endswith("to") or "termin" in _norm_col(c)
                or "resign" in _norm_col(c)]
        for s in starts:
            sp = self._col_profile(table, s)
            srole = getattr(sp, "role", None)
            for e in ends:
                ep = self._col_profile(table, e)
                erole = getattr(ep, "role", None)
                # accept date/year, OR empty (an all-null end is still the end)
                if srole in ("date", "year") and erole in ("date", "year", "empty"):
                    return (s, e)
        return None

    # ---- per-filter value handling ---------------------------------------- #
    @staticmethod
    def _alias(a: Aggregation) -> str:
        base = a.alias or (f"{a.fn}_{a.column}" if a.column else a.fn)
        base = re.sub(r"\W+", "_", str(base)).strip("_") or a.fn
        if not re.match(r"[A-Za-z_]", base[0]):
            base = "v_" + base
        return base

    def _resolve_filter(self, plan: QueryPlan, f: Filter,
                        issues: list, notes: list) -> None:
        table = plan.table if self.resolve_column(plan.table, f.column) else \
            (plan.join.right_table if plan.join else plan.table)
        cp = self._col_profile(table, f.column)
        stype = cp.semantic_type if cp else "text"
        role = cp.role if cp else "text"
        # A column that is the START/END of a known validity pair is a date,
        # even if it is transiently all-empty (e.g. an END column where every
        # current row is still active). Treat it as a date for op validity.
        pair = self._validity_pair(table)
        if pair and f.column in pair and stype in ("empty", "text") \
                and f.op in ("eq", "ne", "gt", "ge", "lt", "le", "between"):
            stype, role = "date", "date"
            cp = type("P", (), {"semantic_type": "date", "role": "date",
                                "date_format": getattr(cp, "date_format", None)})()

        if f.op in ("isnull", "notnull"):
            f._resolved = {"kind": f.op}
            return

        # a 4-digit year against a date/year column
        if f.op in ("eq", "match") and f.value is not None \
                and _YEAR_VAL_RE.match(str(f.value).strip()) \
                and (stype == "date" or role == "year"):
            f._resolved = {"kind": "year", "year": int(str(f.value).strip()),
                           "stype": stype}
            return

        if stype == "number" and f.op in ("eq", "ne", "gt", "ge", "lt", "le",
                                          "between", "match", "in"):
            try:
                if f.op == "between":
                    lo, hi = f.value
                    f._resolved = {"kind": "between", "lo": float(lo), "hi": float(hi),
                                   "coerce": _needs_coerce(cp)}
                elif f.op == "in":
                    vals = [float(v) for v in (f.value or [])]
                    f._resolved = {"kind": "num_in", "values": vals,
                                   "coerce": _needs_coerce(cp)}
                else:
                    op = "eq" if f.op == "match" else f.op
                    f._resolved = {"kind": "cmp", "op": op, "value": float(f.value),
                                   "coerce": _needs_coerce(cp)}
                return
            except (TypeError, ValueError):
                issues.append(ValidationIssue(
                    "unknown_value",
                    f"'{f.value}' is not a number, but column '{f.column}' is numeric"))
                return

        if stype == "date" and f.op in ("eq", "ne", "gt", "ge", "lt", "le", "between"):
            iso = self._parse_dates(f.value if f.op == "between" else [f.value], cp)
            if iso is None:
                issues.append(ValidationIssue(
                    "unknown_value", f"could not parse '{f.value}' as a date "
                                     f"for column '{f.column}'"))
                return
            if f.op == "between":
                f._resolved = {"kind": "date_between", "lo": iso[0], "hi": iso[1]}
            else:
                f._resolved = {"kind": "date_cmp", "op": f.op, "value": iso[0],
                               "or_null": bool(getattr(f, "or_null", False))}
            return

        # ---- text path: resolve to literal spellings present in the data
        if f.op == "in":
            all_orig: list = []
            misses: list = []
            for v in (f.value or []):
                orig, how, sugg = self.resolve_text_value(table, f.column, v)
                if orig:
                    all_orig.extend(orig)
                else:
                    misses.append((v, sugg))
            if all_orig:
                f._resolved = {"kind": "isin", "values": sorted(set(all_orig))}
                if misses:
                    notes.append("values not found and skipped: "
                                 + ", ".join(str(m[0]) for m in misses))
            else:
                issues.append(ValidationIssue(
                    "unknown_value",
                    f"none of {f.value} exist in column '{f.column}'"))
            return

        if f.op in ("match", "eq", "ne", "contains"):
            orig, how, sugg = self.resolve_text_value(table, f.column, f.value)
            if orig:
                kind = "not_isin" if f.op == "ne" else "isin"
                f._resolved = {"kind": kind,
                               "values": sorted({str(o).strip() for o in orig})}
                if how != "exact":
                    shown = ", ".join(repr(o) for o in sorted(set(orig))[:6])
                    notes.append(f"'{f.value}' matched value(s) {shown} "
                                 f"in column '{f.column}'")
                return
            # not in this column at all -> try rebinding to the right column
            rb = self.rebind_value_column(plan.table, f.value)
            if rb and rb[0] != f.column:
                new_col, display = rb
                orig2, how2, _ = self.resolve_text_value(plan.table, new_col, f.value)
                if orig2:
                    notes.append(f"value '{f.value}' does not occur in "
                                 f"'{f.column}' but DOES occur in '{new_col}'; "
                                 "filter rebound to that column")
                    f.column = new_col
                    f._resolved = {"kind": "isin",
                                   "values": sorted({str(o).strip() for o in orig2})}
                    return
            if f.op == "contains":   # high-cardinality / free text: raw contains
                f._resolved = {"kind": "contains", "pattern": str(f.value)}
                return
            # Last resort BEFORE declaring not-found: resolve the typed value
            # against THIS column's distinct values, tolerating misspelling
            # (order-independent, up to two near-miss tokens). This is the
            # "check the column's real values" step: a name like
            # 'Spuros spirou' resolves to the one stored 'Spyrou, Spiros' it
            # uniquely near-matches. Scoped to the chosen filter column only.
            if self.vindex is not None:
                from jarvisman.semantics.value_index import resolve_phrase_fuzzy
                dom = self.domain_for(table, f.column)
                fz = self._resolve_in_column(f.value, dom)
                if fz is not None and len(fz) == 1:
                    notes.append(f"interpreted '{f.value}' as the stored "
                                 f"value '{fz[0]}' in column '{f.column}'")
                    f._resolved = {"kind": "isin", "values": [fz[0]]}
                    return
                if fz is not None and len(fz) > 1:
                    # several plausible -> surface as a clarification, do not pick
                    issues.append(ValidationIssue(
                        "unknown_value",
                        f"'{f.value}' could mean several values in "
                        f"'{f.column}': " + ", ".join(repr(s) for s in fz[:4])))
                    return
            msg = f"value '{f.value}' not found in column '{f.column}' of '{table}'"
            if sugg:
                msg += "; closest actual values: " + ", ".join(repr(s) for s in sugg)
            issues.append(ValidationIssue("unknown_value", msg))
            return

        issues.append(ValidationIssue(
            "bad_op", f"op '{f.op}' is not valid for {stype} column '{f.column}'"))

    @staticmethod
    def _parse_dates(values, cp) -> Optional[list]:
        import pandas as pd
        dayfirst = False
        fmt = getattr(cp, "date_format", None) if cp else None
        if fmt and "%d" in fmt and "%m" in fmt:
            dayfirst = fmt.index("%d") < fmt.index("%m")
        out = []
        for v in values:
            sv = str(v).strip()
            # dotted dd.mm.yyyy is the European convention -> dayfirst, which
            # also silences pandas' ambiguity warning on values like 31.12.2023
            df_v = dayfirst or bool(re.match(r"^\d{1,2}\.\d{1,2}\.\d{4}$", sv))
            try:
                ts = pd.to_datetime(sv, dayfirst=df_v, errors="coerce")
            except (TypeError, ValueError, OverflowError) as exc:
                cfg.dbg(f"query_plan._parse_dates [{sv!r}]", exc)
                ts = None
            if ts is None or pd.isna(ts):
                return None
            out.append(ts.strftime("%Y-%m-%d"))
        return out


# --------------------------------------------------------------------------- #
# Compilation                                                                 #
# --------------------------------------------------------------------------- #
def _col_expr(frame: str, column: str) -> str:
    return f"{frame}[{column!r}]"


def _filter_expr(frame: str, f: Filter) -> tuple[str, str]:
    """Returns (mask_expression, description)."""
    r = getattr(f, "_resolved", None) or {}
    col = _col_expr(frame, f.column)
    kind = r.get("kind")
    if kind == "isin":
        # compare on a whitespace-stripped column: a stored cell with a
        # trailing space ('Karanikolaos,Panagiotis ') must still match the
        # verified value. Stripping is safe -- it never changes a real match.
        return (f"{col}.astype(str).str.strip().isin({r['values']!r})",
                f"{f.column} is one of {r['values']!r}")
    if kind == "not_isin":
        return (f"~{col}.astype(str).str.strip().isin({r['values']!r})",
                f"{f.column} is NOT one of {r['values']!r}")
    if kind == "contains":
        return (f"{col}.astype(str).str.contains({r['pattern']!r}, case=False, "
                f"na=False, regex=False)",
                f"{f.column} contains {r['pattern']!r}")
    if kind == "cmp":
        base = (f"pd.to_numeric({col}, errors='coerce')" if r.get("coerce") else col)
        sym = {"eq": "==", "ne": "!=", "gt": ">", "ge": ">=", "lt": "<", "le": "<="}[r["op"]]
        return f"{base} {sym} {r['value']!r}", f"{f.column} {sym} {r['value']}"
    if kind == "num_in":
        base = (f"pd.to_numeric({col}, errors='coerce')" if r.get("coerce") else col)
        return f"{base}.isin({r['values']!r})", f"{f.column} in {r['values']!r}"
    if kind == "between":
        base = (f"pd.to_numeric({col}, errors='coerce')" if r.get("coerce") else col)
        return (f"({base} >= {r['lo']!r}) & ({base} <= {r['hi']!r})",
                f"{f.column} between {r['lo']} and {r['hi']}")
    if kind == "year":
        if r.get("stype") == "date":
            return f"{col}.dt.year == {r['year']}", f"year({f.column}) == {r['year']}"
        return (f"pd.to_numeric({col}, errors='coerce') == {r['year']}",
                f"{f.column} == {r['year']}")
    if kind == "date_cmp":
        sym = {"eq": "==", "ne": "!=", "gt": ">", "ge": ">=", "lt": "<", "le": "<="}[r["op"]]
        # coerce to datetime at compile time so a text-typed date column still
        # compares correctly; each side fully parenthesized (| binds tighter
        # than >=, so the comparison MUST be wrapped).
        lhs = f"pd.to_datetime({col}, errors='coerce')"
        cmp = f"({lhs} {sym} pd.Timestamp({r['value']!r}))"
        if r.get("or_null"):
            return (f"({cmp} | ({col}.isna()))",
                    f"{f.column} {sym} {r['value']} OR {f.column} empty (active)")
        return cmp, f"{f.column} {sym} {r['value']}"
    if kind == "date_between":
        return (f"({col} >= pd.Timestamp({r['lo']!r})) & "
                f"({col} <= pd.Timestamp({r['hi']!r}))",
                f"{f.column} between {r['lo']} and {r['hi']}")
    if kind == "isnull":
        return f"{col}.isna()", f"{f.column} is null"
    if kind == "notnull":
        return f"{col}.notna()", f"{f.column} is not null"
    raise ValueError(f"unresolved filter on '{f.column}'")


def _compile_compare(plan, validator, notes, issues):
    c = plan.compare
    lt = validator.resolve_table(c.get("left_table"))
    rt = validator.resolve_table(c.get("right_table"))
    key = c.get("key")
    measures = [m for m in (c.get("measures") or [])]
    if not (lt and rt and key):
        issues.append(ValidationIssue("bad_join",
                      "compare needs left_table, right_table and key"))
        return None, issues, notes
    lcol = validator.resolve_column(lt, key)
    rcol = validator.resolve_column(rt, key)
    if not lcol or not rcol:
        issues.append(ValidationIssue("unknown_column",
                      f"compare key '{key}' not in both tables"))
        return None, issues, notes
    good = []
    for m in measures:
        ml, mr = validator.resolve_column(lt, m), validator.resolve_column(rt, m)
        if ml and mr:
            good.append((ml, mr))
        else:
            notes.append(f"compare measure '{m}' missing on one side; skipped")
    if not good:
        issues.append(ValidationIssue("bad_agg",
                      "no shared measure to compare"))
        return None, issues, notes
    lab_l = str(c.get("left_label") or "left")
    lab_r = str(c.get("right_label") or "right")
    L = [f"L = dfs[{lt!r}].copy(); R = dfs[{rt!r}].copy()",
         f"print('PROV left rows:', len(L)); print('PROV right rows:', len(R))"]
    explain = [f"comparing '{lt}' ({lab_l}) vs '{rt}' ({lab_r}) on '{lcol}'"]
    lcols = [lcol] + [a for a, _ in good]
    rcols = [rcol] + [b for _, b in good]
    L.append(f"L = L[{lcols!r}].groupby({lcol!r}, dropna=False).sum().reset_index()")
    L.append(f"R = R[{rcols!r}].groupby({rcol!r}, dropna=False).sum().reset_index()")
    L.append(f"m = L.merge(R, left_on={lcol!r}, right_on={rcol!r}, how='outer', "
             f"suffixes=('_{lab_l}', '_{lab_r}'), indicator=True)")
    for (a, b) in good:
        an, bn = f"{a}_{lab_l}", f"{b}_{lab_r}"
        if a == b:
            an, bn = f"{a}_{lab_l}", f"{b}_{lab_r}"
        L.append(f"m[{an!r}] = pd.to_numeric(m.get({an!r}), errors='coerce').fillna(0)")
        L.append(f"m[{bn!r}] = pd.to_numeric(m.get({bn!r}), errors='coerce').fillna(0)")
        L.append(f"m[{(a + '_delta')!r}] = m[{bn!r}] - m[{an!r}]")
        L.append(f"m[{(a + '_pct')!r}] = (m[{(a + '_delta')!r}] / "
                 f"m[{an!r}].replace(0, float('nan')) * 100).round(2)")
        explain.append(f"{a}: delta and % change ({lab_l}->{lab_r})")
    status_map = {"left_only": f"gone ({lab_l})",
                  "right_only": f"new ({lab_r})", "both": "both"}
    L.append(f"m['STATUS'] = m['_merge'].map({status_map!r})")
    L.append("result = m.drop(columns=['_merge'])")
    L.append("print('PROV compared rows:', len(result))")
    explain.append("rows present on one side only flagged in STATUS")
    code = "\n".join(L)
    from jarvisman.runtime.sandbox import validate_code as _vc
    ok, why = _vc(code)
    if not ok:
        issues.append(ValidationIssue("bad_join", f"compare code rejected: {why}"))
        return None, issues, notes
    return CodeBundle(code=code, explain=explain,
                      tables_used=[lt, rt]), issues, notes


def compile_plan(plan: QueryPlan, model: SemanticModel, dataframes: dict,
                 vindex: ValueIndex, cards: dict = None,
                 coverage_anchors: list = None
                 ) -> tuple[Optional[CodeBundle], list, list]:
    """Validate then compile. Returns (bundle_or_None, issues, notes)."""
    validator = PlanValidator(model, dataframes, vindex)
    validator.attach_context(cards, coverage_anchors)
    plan, issues, notes = validator.validate(plan)
    hard = [i for i in issues if i.kind in
            ("unknown_table", "unknown_column", "unknown_value", "bad_agg",
             "bad_op", "empty_plan", "dropped_condition")]
    if hard:
        return None, issues, notes

    if plan.compare:
        return _compile_compare(plan, validator, notes, issues)

    L: list[str] = []
    explain: list[str] = []
    if getattr(plan, "intent", ""):
        explain.append(f"intent: {plan.intent}")
    base = validator.long_base(plan.table)
    is_long = validator.is_long(plan.table)
    tables_used = [base]
    if plan.union:
        frames = [base] + list(plan.union)
        tables_used = list(frames)
        tag = "SOURCE"
        if any(str(c).upper() == "SOURCE"
               for c in validator.columns_of(base)):
            tag = "SOURCE_FILE"     # never overwrite the user's real column
            explain.append("the data already has a SOURCE column; the file "
                           "tag is named SOURCE_FILE")
        L.append("_parts = []")
        for fr in frames:
            L.append(f"_p = dfs[{fr!r}].copy(); _p[{tag!r}] = {fr!r}")
            L.append("_parts.append(_p)")
        L.append("t = pd.concat(_parts, ignore_index=True, sort=False)")
        L.append("print('PROV unioned rows:', len(t))")
        explain.append("combined rows from " + ", ".join(frames)
                       + " (SOURCE column tags origin)")
    else:
        L.append(f"t = dfs[{base!r}].copy()")
    L.append("print('PROV start rows:', len(t))")

    tprof = model.tables.get(base) if model else None
    if plan.aggregations and tprof and tprof.summary_row_indices:
        idxs = list(tprof.summary_row_indices)
        L.append(f"t = t[~t.index.isin({idxs!r})]")
        L.append(f"print('PROV excluded {len(idxs)} total/subtotal row(s):', len(t), 'rows')")
        explain.append(f"excluded {len(idxs)} Total/Subtotal row(s) detected in the sheet")

    if is_long:
        lv = validator._long_view(plan.table) or {}
        id_vars = list(lv.get("id_vars", []))
        vmap = lv.get("value_cols", {})
        year_map = {c: int(s["year"]) for c, s in vmap.items()}
        meas_map = {c: str(s["measure"]) for c, s in vmap.items()}
        L.append(f"t = t.melt(id_vars={id_vars!r}, "
                 f"value_vars={list(vmap.keys())!r}, "
                 f"var_name='PIVOTCOL', value_name='VALUE')")
        L.append(f"t['YEAR'] = t['PIVOTCOL'].map({year_map!r})")
        L.append(f"t['MEASURE'] = t['PIVOTCOL'].map({meas_map!r})")
        L.append("t['VALUE'] = pd.to_numeric(t['VALUE'], errors='coerce')")
        L.append("print('PROV after unpivot:', len(t), 'rows')")
        explain.append(f"unpivoted '{base}' to one row per (entity, year, measure)")

    if plan.join:
        tables_used.append(plan.join.right_table)
        L.append(f"r = dfs[{plan.join.right_table!r}].copy()")
        L.append(f"t = t.merge(r, how='left', left_on={plan.join.left_column!r}, "
                 f"right_on={plan.join.right_column!r}, suffixes=('', '__right'))")
        L.append("print('PROV after join:', len(t), 'rows')")
        explain.append(f"joined {plan.join.right_table} on "
                       f"{plan.join.left_column} = {plan.join.right_column}")

    for f in plan.filters:
        if getattr(f, "_resolved", None) is None:
            continue
        expr, desc = _filter_expr("t", f)
        L.append(f"t = t[{expr}]")
        L.append(f"print('PROV after filter {f.column}:', len(t), 'rows')")
        explain.append("filter: " + desc)

    aliases: list[str] = []
    if plan.aggregations:
        # coerce text-typed numeric columns once, on the working copy
        coerced = set()
        for a in plan.aggregations:
            if a.fn in ("sum", "mean", "median", "min", "max") and a.column:
                cp = (tprof.column(a.column) if tprof else None)
                if _needs_coerce(cp) and a.column not in coerced:
                    L.append(f"t[{a.column!r}] = pd.to_numeric(t[{a.column!r}], "
                             f"errors='coerce')")
                    coerced.add(a.column)
                    explain.append(f"coerced '{a.column}' to numeric "
                                   f"({cp.numeric_parse_rate:.0%} parseable)")
        if plan.group_by:
            specs = []
            for a in plan.aggregations:
                alias = PlanValidator._alias(a)
                aliases.append(alias)
                if a.fn == "count" and not a.column:
                    specs.append(f"{alias}=({plan.group_by[0]!r}, 'size')")
                else:
                    specs.append(f"{alias}=({a.column!r}, {a.fn!r})")
            L.append(f"g = t.groupby({plan.group_by!r}, dropna=False)")
            L.append(f"result = g.agg({', '.join(specs)}).reset_index()")
            explain.append(f"grouped by {plan.group_by}, computed "
                           + ", ".join(aliases))
            for hf in (plan.having or []):
                sym = {"gt": ">", "ge": ">=", "lt": "<", "le": "<=",
                       "eq": "==", "ne": "!="}.get(hf.op)
                if sym is None:
                    # never guess a comparison direction: drop and disclose
                    explain.append(f"having on {hf.column}: unknown operator "
                                   f"'{hf.op}', condition skipped")
                    continue
                try:
                    hv = float(hf.value)
                except (TypeError, ValueError):
                    hv = hf.value
                L.append(f"result = result[result[{hf.column!r}] {sym} {hv!r}]")
                L.append(f"print('PROV after having {hf.column} {sym} {hv!r}:', len(result))")
                explain.append(f"kept groups where {hf.column} {sym} {hv!r}")
            for dv in (plan.derived or []):
                if dv["kind"] == "pct_of_total" and dv.get("num"):
                    num = dv["num"]
                    L.append(f"result[{dv['alias']!r}] = (result[{num!r}] / "
                             f"result[{num!r}].sum() * 100).round(2)")
                    explain.append(f"{dv['alias']} = {num} as % of total")
                elif dv["kind"] == "ratio" and dv.get("num") and dv.get("den"):
                    L.append(f"result[{dv['alias']!r}] = (result[{dv['num']!r}] / "
                             f"result[{dv['den']!r}]).round(4)")
                    explain.append(f"{dv['alias']} = {dv['num']} / {dv['den']}")
        else:
            L.append("vals = {}")
            for a in plan.aggregations:
                alias = PlanValidator._alias(a)
                aliases.append(alias)
                if a.fn == "count" and not a.column:
                    L.append(f"vals[{alias!r}] = int(len(t))")
                elif a.fn == "count":
                    L.append(f"vals[{alias!r}] = int(t[{a.column!r}].notna().sum())")
                elif a.fn == "nunique":
                    L.append(f"vals[{alias!r}] = int(t[{a.column!r}].nunique())")
                else:
                    L.append(f"vals[{alias!r}] = t[{a.column!r}].{a.fn}()")
            L.append("result = pd.DataFrame([vals])")
            explain.append("computed " + ", ".join(aliases))
    else:
        if plan.sort and plan.select and plan.sort.by not in plan.select:
            # sort key not among the selected columns: order rows first
            L.append(f"t = t.sort_values({plan.sort.by!r}, "
                     f"ascending={not plan.sort.desc})")
            explain.append(f"sorted by {plan.sort.by} "
                           + ("descending" if plan.sort.desc else "ascending"))
            plan.sort = None
        if plan.select:
            L.append(f"result = t[{plan.select!r}]")
            explain.append(f"selected columns {plan.select}")
        else:
            L.append("result = t")
            explain.append("returning matching rows")
        if cfg.DISTINCT_SELECT:
            # row listings are DISTINCT by default: asking for "the companies"
            # must not return the same company once per transaction row
            L.append("result = result.drop_duplicates()")
            L.append("print('PROV distinct rows:', len(result))")
            explain.append("duplicates removed (distinct)")

    if plan.sort:
        L.append(f"result = result.sort_values({plan.sort.by!r}, "
                 f"ascending={not plan.sort.desc})")
        explain.append(f"sorted by {plan.sort.by} "
                       + ("descending" if plan.sort.desc else "ascending"))
    if plan.limit:
        L.append(f"result = result.head({int(plan.limit)})")
        explain.append(f"limited to {int(plan.limit)} rows")
    L.append("print('PROV result rows:', len(result))")

    return CodeBundle(code="\n".join(L), explain=explain,
                      tables_used=tables_used), issues, notes
