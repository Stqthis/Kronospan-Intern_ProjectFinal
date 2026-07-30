"""Statistical semantic model of the workbook.

Division of labour: CODE computes everything countable (null rates,
cardinality, uniqueness, value histograms, year-likeness, relationships,
summary rows) over the FULL columns -- Excel sheets are bounded, so a full
vectorised scan costs fractions of a second and removes all sampling bias.
The LLM contributes only what code cannot: business meaning per column and a
table summary (seeded from the column_types profile built at index time, so
no extra LLM calls are needed here).

The model is the single source of truth the planner and validator reason
against; roles are advisory for planning, but column existence, domains and
relationships are facts the compiler enforces.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Optional

from jarvisman import config as cfg
from jarvisman.semantics.value_index import norm_text

_YEAR_IN_NAME_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_ID_NAME_HINTS = ("id", "code", "key", "no", "number", "ref", "κωδ", "αριθμ")
_YEAR_NAME_HINTS = ("year", "yr", "fy", "ετος", "έτος", "χρηση", "χρήση", "period")


# --------------------------------------------------------------------------- #
# Dataclasses                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class ColumnProfile:
    name: str
    dtype: str
    dtype_kind: str               # numpy kind: i/u/f = numeric, M = datetime, O = object
    semantic_type: str            # number | date | text | bool | empty
    role: str                     # identifier | measure | dimension | date | year | text | empty
    null_rate: float
    distinct: int
    unique_ratio: float           # distinct / non-null
    top_values: list = field(default_factory=list)   # [(display, count)] capped
    samples: list = field(default_factory=list)       # up to 3 real values (high-cardinality cols)
    vmin: Optional[str] = None
    vmax: Optional[str] = None
    numeric_parse_rate: float = 0.0
    date_format: Optional[str] = None
    meaning: str = ""             # from the LLM profile (column_types)
    unit: str = ""                # heuristic from the column name (EUR, %, ...)


@dataclass
class Relationship:
    child_table: str
    child_column: str
    parent_table: str
    parent_column: str
    coverage: float               # share of child values found in parent


@dataclass
class TableProfile:
    name: str
    rows: int
    columns: list = field(default_factory=list)       # [ColumnProfile]
    summary: str = ""
    summary_row_indices: list = field(default_factory=list)
    pivot_year_columns: dict = field(default_factory=dict)   # year -> [col, ...]
    # Virtual unpivoted ("long") view of a wide year layout:
    # {"id_vars": [...], "value_cols": {col: {"year": int, "measure": stem}}}
    long_view: dict = field(default_factory=dict)

    def column(self, name: str) -> Optional[ColumnProfile]:
        for c in self.columns:
            if c.name == name:
                return c
        return None


@dataclass
class SemanticModel:
    tables: dict = field(default_factory=dict)        # name -> TableProfile
    relationships: list = field(default_factory=list)  # [Relationship]

    def table_names(self) -> list[str]:
        return list(self.tables.keys())


# --------------------------------------------------------------------------- #
# Column profiling                                                            #
# --------------------------------------------------------------------------- #
def _unit_from_name(name: str) -> str:
    n = norm_text(name)
    for u in ("eur", "usd", "gbp", "chf", "ron", "huf", "pln", "czk", "bgn"):
        if u in n.split() or n.endswith(u):
            return u.upper()
    if "%" in str(name) or "percent" in n or "ποσοστο" in n:
        return "%"
    return ""


def _is_year_like(series, pd) -> bool:
    try:
        vals = pd.to_numeric(series, errors="coerce").dropna()
        if len(vals) == 0:
            return False
        ints = vals[(vals == vals.astype(int))]
        if len(ints) / len(vals) < 0.95:
            return False
        return float(ints.between(1900, 2100).mean()) >= 0.8
    except (TypeError, ValueError) as exc:
        cfg.dbg("semantic_model._is_year_like", exc)
        return False


def _profile_column(name: str, series, rows: int, pd) -> ColumnProfile:
    non_null = int(series.notna().sum())
    null_rate = 1.0 - (non_null / rows) if rows else 1.0
    kind = getattr(series.dtype, "kind", "O")

    if non_null == 0:
        return ColumnProfile(name=name, dtype=str(series.dtype), dtype_kind=kind,
                             semantic_type="empty",
                             role="empty", null_rate=1.0, distinct=0, unique_ratio=0.0)

    try:
        distinct = int(series.nunique(dropna=True))
    except (TypeError, ValueError) as exc:
        cfg.dbg(f"semantic_model.profile nunique [{name}]", exc)
        distinct = 0
    unique_ratio = distinct / non_null if non_null else 0.0

    top_values: list = []
    vmin = vmax = None
    numeric_parse_rate = 0.0
    semantic_type = "text"
    role = "text"

    name_norm = norm_text(name)
    year_named = any(h in name_norm for h in _YEAR_NAME_HINTS)

    if kind in "mM":
        semantic_type, role = "date", "date"
        try:
            vmin, vmax = str(series.min()), str(series.max())
        except (TypeError, ValueError, OverflowError) as exc:
            cfg.dbg(f"semantic_model.profile date min/max [{name}]", exc)
    elif kind == "b":
        semantic_type, role = "bool", "dimension"
    elif kind in "iuf":
        semantic_type = "number"
        numeric_parse_rate = 1.0
        try:
            vmin, vmax = str(series.min()), str(series.max())
        except (TypeError, ValueError, OverflowError) as exc:
            cfg.dbg(f"semantic_model.profile numeric min/max [{name}]", exc)
        if (year_named or _is_year_like(series, pd)) and distinct <= 150:
            role = "year"
        elif unique_ratio >= 0.95 and (
            any(h in name_norm.split() for h in _ID_NAME_HINTS)
            or name_norm.endswith(tuple(_ID_NAME_HINTS))
        ):
            role = "identifier"
        else:
            role = "measure"
    else:
        # text-ish column: measure-as-text detection, then dimension/identifier
        num = pd.to_numeric(series, errors="coerce")
        numeric_parse_rate = float(num.notna().sum()) / non_null
        if numeric_parse_rate >= 0.9:
            semantic_type, role = "number", "measure"
            try:
                vmin, vmax = str(num.min()), str(num.max())
            except (TypeError, ValueError, OverflowError) as exc:
                cfg.dbg(f"semantic_model.profile coerced min/max [{name}]", exc)
        elif unique_ratio >= 0.95 and distinct >= 5:
            semantic_type, role = "text", "identifier"
        elif distinct <= max(cfg.CATEGORICAL_MAX_UNIQUE, 1) or unique_ratio <= 0.1:
            semantic_type, role = "text", "dimension"

    if kind not in "iufmM":
        try:
            vc = series.dropna().astype(str).value_counts().head(cfg.PROFILE_TOP_K)
            top_values = [(str(v), int(c)) for v, c in vc.items()]
        except (TypeError, ValueError) as exc:
            cfg.dbg(f"semantic_model.profile top_values [{name}]", exc)
            top_values = []

    samples = []
    if not top_values and getattr(series.dtype, "kind", "O") == "O":
        try:
            samples = [str(x)[:28] for x in series.dropna().unique()[:3]]
        except (TypeError, ValueError) as exc:
            cfg.dbg(f"semantic_model.profile samples [{name}]", exc)
            samples = []
    return ColumnProfile(
        name=name, dtype=str(series.dtype), dtype_kind=kind,
        semantic_type=semantic_type, role=role,
        null_rate=round(null_rate, 4), distinct=distinct,
        unique_ratio=round(unique_ratio, 4), top_values=top_values,
        vmin=vmin, vmax=vmax, samples=samples,
        numeric_parse_rate=round(numeric_parse_rate, 4),
        unit=_unit_from_name(name),
    )


# --------------------------------------------------------------------------- #
# Summary-row detection (Total / Σύνολο rows inside the data range)           #
# --------------------------------------------------------------------------- #
_SUMMARY_WORDS = (
    "total", "subtotal", "grand total", "sum", "totals",
    "συνολο", "μερικο συνολο", "γενικο συνολο", "αθροισμα", "συνολα",
)


def _detect_summary_rows(df, pd) -> list[int]:
    """Rows whose text cell IS a total/subtotal label (EN/GR, accent- and
    case-insensitive). These poison aggregations (double counting), so the
    compiler excludes them by default and says so in the provenance."""
    out: set[int] = set()
    text_cols = [c for c in df.columns
                 if getattr(df[c].dtype, "kind", "O") not in "iufcMmb"][:8]
    for c in text_cols:
        try:
            normed = df[c].dropna().astype(str).map(norm_text)
        except (TypeError, ValueError, AttributeError) as exc:
            cfg.dbg(f"semantic_model._detect_summary_rows [{c}]", exc)
            continue
        for idx, v in normed.items():
            if not v or len(v) > 40:
                continue
            if any(v == w or v.startswith(w + " ") or v.endswith(" " + w)
                   for w in _SUMMARY_WORDS):
                out.add(int(idx))
    return sorted(out)


# --------------------------------------------------------------------------- #
# Pivot (wide year) layout detection                                          #
# --------------------------------------------------------------------------- #
def _detect_pivot_years(columns: list[str]) -> dict:
    """Detect '2022 Rev | 2022 Cost | 2023 Rev | ...' layouts: column names
    that embed a year, grouped per year. Exposed to the planner as a hint so
    'compare 2024 vs 2025' picks the right measure columns."""
    by_year: dict[str, list[str]] = {}
    for c in columns:
        m = _YEAR_IN_NAME_RE.search(str(c))
        if m:
            by_year.setdefault(m.group(1), []).append(str(c))
    return by_year if len(by_year) >= 2 else {}


def _build_long_view(columns: list, by_year: dict) -> dict:
    """Spec for the virtual unpivoted view of a wide year layout. Each pivot
    column '2023 Rev' contributes (year=2023, measure='Rev'); every non-pivot
    column is an id_var. The compiler materialises this as a melt on demand,
    so 'compare 2024 vs 2025' becomes an ordinary group-by over YEAR."""
    value_cols: dict = {}
    pivot_set = {c for cols in by_year.values() for c in cols}
    for year, cols in by_year.items():
        for c in cols:
            stem = _YEAR_IN_NAME_RE.sub("", str(c)).strip(" -_/").strip()
            value_cols[str(c)] = {"year": int(year), "measure": stem or "VALUE"}
    id_vars = [c for c in columns if c not in pivot_set]
    if len(value_cols) < 2 or not id_vars:
        return {}
    return {"id_vars": id_vars, "value_cols": value_cols}


# --------------------------------------------------------------------------- #
# Relationship (FK-like) detection                                            #
# --------------------------------------------------------------------------- #
def _norm_value_set(series, cap: int) -> Optional[set]:
    try:
        vals = series.dropna().astype(str).unique()
    except (TypeError, ValueError) as exc:
        cfg.dbg("semantic_model._norm_value_set", exc)
        return None
    if len(vals) == 0 or len(vals) > cap:
        return None
    return {norm_text(v) for v in vals}


def _detect_relationships(dataframes: dict, model: "SemanticModel") -> list[Relationship]:
    """Cross-table value containment: child column whose distinct values are
    (almost) all present in a near-unique parent column => FK-like link. Only
    links found here may be used by the query compiler for joins."""
    cap = cfg.RELATIONSHIP_MAX_DISTINCT
    parents: list[tuple[str, str, set]] = []
    children: list[tuple[str, str, set]] = []
    for tname, tprof in model.tables.items():
        df = dataframes.get(tname)
        if df is None:
            continue
        for col in tprof.columns:
            if col.semantic_type in ("empty",) or col.distinct < 3:
                continue
            vs = _norm_value_set(df[col.name], cap)
            if vs is None:
                continue
            if col.unique_ratio >= 0.95:
                parents.append((tname, col.name, vs))
            children.append((tname, col.name, vs))

    rels: list[Relationship] = []
    for ct, cc, cvals in children:
        for pt, pc, pvals in parents:
            if ct == pt:
                continue
            if (ct, cc) == (pt, pc):
                continue
            coverage = len(cvals & pvals) / len(cvals)
            if coverage >= cfg.RELATIONSHIP_MIN_COVERAGE:
                rels.append(Relationship(ct, cc, pt, pc, round(coverage, 4)))
    # deduplicate, keep best coverage per (child_table, child_column)
    best: dict[tuple, Relationship] = {}
    for r in rels:
        k = (r.child_table, r.child_column, r.parent_table)
        if k not in best or r.coverage > best[k].coverage:
            best[k] = r
    return list(best.values())


# --------------------------------------------------------------------------- #
# Build / persist                                                             #
# --------------------------------------------------------------------------- #
def build_semantic_model(dataframes: dict, meanings: Optional[dict] = None) -> SemanticModel:
    """Profile every table statistically (full scan) and merge in the LLM
    meanings/summaries produced at index time by column_types.profile_and_apply
    (``meanings`` is that profile dict; pass {} or None if unavailable)."""
    import pandas as pd

    meanings = meanings or {}
    model = SemanticModel()
    for name, df in (dataframes or {}).items():
        if df is None:
            continue
        rows = len(df)
        tprof = TableProfile(name=name, rows=rows)
        seen = set()
        for c in df.columns:
            cname = str(c)
            if cname in seen:
                continue
            seen.add(cname)
            series = df[c]
            if hasattr(series, "columns"):    # duplicated label -> DataFrame
                continue
            cp = _profile_column(cname, series, rows, pd) if rows else ColumnProfile(
                name=cname, dtype=str(series.dtype),
                dtype_kind=getattr(series.dtype, "kind", "O"), semantic_type="empty",
                role="empty", null_rate=1.0, distinct=0, unique_ratio=0.0)
            info = ((meanings.get(name) or {}).get("columns") or {}).get(cname) or {}
            cp.meaning = str(info.get("meaning") or "").strip()
            fmt = info.get("format")
            if isinstance(fmt, str) and fmt.strip() and fmt.strip().lower() != "null":
                cp.date_format = fmt.strip()
            tprof.columns.append(cp)
        tprof.summary = str((meanings.get(name) or {}).get("summary") or "").strip()
        if rows:
            tprof.summary_row_indices = _detect_summary_rows(df, pd)
            tprof.pivot_year_columns = _detect_pivot_years([str(c) for c in df.columns])
            if tprof.pivot_year_columns and cfg.VIRTUAL_LONG_VIEWS:
                tprof.long_view = _build_long_view(
                    [str(c) for c in df.columns], tprof.pivot_year_columns)
        model.tables[name] = tprof

    model.relationships = _detect_relationships(dataframes or {}, model)
    return model


def _model_path(directory: str) -> str:
    return os.path.join(directory, "semantic_model.json")


def save_semantic_model(directory: str, model: SemanticModel) -> None:
    try:
        os.makedirs(directory, exist_ok=True)
        payload = {
            "tables": {n: asdict(t) for n, t in model.tables.items()},
            "relationships": [asdict(r) for r in model.relationships],
        }
        with open(_model_path(directory), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
    except (OSError, TypeError, ValueError) as exc:
        cfg.dbg("semantic_model.save", exc)  # persistence is best-effort


def load_semantic_model(directory: str) -> Optional[SemanticModel]:
    path = _model_path(directory)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        model = SemanticModel()
        for n, t in (data.get("tables") or {}).items():
            cols = [ColumnProfile(**c) for c in t.get("columns", [])]
            model.tables[n] = TableProfile(
                name=t["name"], rows=t.get("rows", 0), columns=cols,
                summary=t.get("summary", ""),
                summary_row_indices=t.get("summary_row_indices", []),
                pivot_year_columns=t.get("pivot_year_columns", {}),
                long_view=t.get("long_view", {}),
            )
        model.relationships = [Relationship(**r) for r in data.get("relationships", [])]
        return model
    except (OSError, ValueError, TypeError, KeyError) as exc:
        cfg.dbg("semantic_model.load (stale/corrupt cache -> rebuild)", exc)
        return None


# --------------------------------------------------------------------------- #
# Prompt rendering                                                            #
# --------------------------------------------------------------------------- #
def render_for_prompt(model: SemanticModel, table_names: Optional[list[str]] = None,
                      max_dim_values: int = None, cards: Optional[dict] = None) -> str:
    """Compact, evidence-dense schema block for the planner/codegen prompts:
    roles, types, units, ranges, the top values of dimensions, meanings,
    relationships and pivot hints -- instead of hundreds of raw values."""
    max_vals = max_dim_values or cfg.PROMPT_DIM_MAX_VALUES
    names = table_names or model.table_names()
    # Sibling sheets share a schema, so the block above is identical for each
    # and the model has nothing to choose on. One line of what DIFFERS (period
    # covered, which entities) is what tells them apart.
    disc = table_discriminators(model) if len(names) > 1 else {}
    parts: list[str] = []
    for n in names:
        t = model.tables.get(n)
        if t is None:
            continue
        head = f"Table '{n}' ({t.rows} rows)"
        if t.summary:
            head += f": {t.summary}"
        lines = [head]
        if disc.get(n):
            lines.append(f"  Distinguishes from sibling sheets: {disc[n]}")
        for c in t.columns:
            if c.role == "empty":
                continue
            bits = [f"  - '{c.name}' [{c.role}, {c.semantic_type}"]
            if c.unit:
                bits.append(f", {c.unit}")
            bits.append("]")
            desc = "".join(bits)
            if c.semantic_type == "number" and c.vmin is not None:
                desc += f" range {c.vmin}..{c.vmax}"
            if c.semantic_type == "date" and c.vmin is not None:
                desc += f" from {c.vmin} to {c.vmax}"
            if cfg.SCHEMA_SAMPLES and c.samples and c.role in ("identifier", "text"):
                desc += " e.g. " + ", ".join(repr(s) for s in c.samples[:3])
            if c.role in ("dimension", "year") and c.top_values:
                vals = ", ".join(v for v, _ in c.top_values[:max_vals])
                more = c.distinct - min(len(c.top_values), max_vals)
                desc += f" values: [{vals}]" + (f" (+{more} more)" if more > 0 else "")
            elif c.role == "identifier":
                desc += f" ({c.distinct} distinct)"
            if c.meaning:
                m = c.meaning[:cfg.PROMPT_MEANING_MAX_CHARS]
                desc += f" -- {m}"
            lines.append(desc)
        if t.summary_row_indices:
            lines.append(f"  (note: {len(t.summary_row_indices)} Total/Subtotal row(s) "
                         "detected inside the data; aggregations exclude them)")
        if cards and cfg.TABLE_CARDS and n in cards:
            card = cards[n] or {}
            if card.get("purpose"):
                lines.append(f"  Purpose: {str(card['purpose'])[:160]}")
            if card.get("as_of"):
                lines.append(f"  This table is a SNAPSHOT as of "
                             f"{card['as_of']} -- 'as at {card['as_of']}' "
                             "questions need NO date filter; a different "
                             "as-at date is NOT covered by this file.")
            elif card.get("table_kind"):
                lines.append(f"  Kind: {card['table_kind']}")
            if card.get("grain"):
                lines.append(f"  Grain: {str(card['grain'])[:100]}")
            if card.get("entity_key"):
                lines.append("  Entity key: "
                             + ", ".join(str(k) for k in card["entity_key"][:2]))
            for vp in (card.get("validity") or [])[:2]:
                lines.append(
                    f"  Validity interval: {vp['start']} .. {vp['end']} "
                    f"(empty {vp['end']} = {vp.get('null_end_means', 'active')})"
                    " -- 'as at <date>' means "
                    f"{vp['start']} <= date AND ({vp['end']} >= date OR empty)")
            meas = card.get("measures") or {}
            if meas:
                ms = "; ".join(f"'{c}' = {str(mn)[:60]}"
                               for c, mn in list(meas.items())[:4])
                lines.append(f"  Measure meanings: {ms}")
            for cv in (card.get("caveats") or [])[:2]:
                lines.append(f"  Note: {str(cv)[:120]}")
        if t.pivot_year_columns and not t.long_view:
            ys = ", ".join(sorted(t.pivot_year_columns))
            lines.append(f"  (note: wide year layout -- separate columns per year: {ys}; "
                         "to compare years, aggregate the matching per-year columns)")
        parts.append("\n".join(lines))
        if t.long_view:
            lv = t.long_view
            measures = sorted({v["measure"] for v in lv["value_cols"].values()})
            years = sorted({v["year"] for v in lv["value_cols"].values()})
            vl = [f"Table '{n}#long' ({t.rows * len(lv['value_cols'])} rows): "
                  f"unpivoted view of '{n}' -- USE THIS for questions about or "
                  "across years"]
            for c in lv["id_vars"]:
                vl.append(f"  - '{c}' (same values as in '{n}')")
            vl.append(f"  - 'YEAR' [year, number] values: "
                      f"[{', '.join(str(y) for y in years)}]")
            vl.append(f"  - 'MEASURE' [dimension, text] values: "
                      f"[{', '.join(measures)}]")
            vl.append("  - 'VALUE' [measure, number] -- the figure for that "
                      "year and measure")
            parts.append("\n".join(vl))
    if model.relationships:
        rl = ["Verified relationships between tables (the ONLY joins allowed):"]
        for r in model.relationships:
            rl.append(f"  - {r.child_table}.'{r.child_column}' -> "
                      f"{r.parent_table}.'{r.parent_column}' (coverage {r.coverage:.0%})")
        parts.append("\n".join(rl))
    return "\n\n".join(parts) if parts else "none"


def table_discriminators(model, max_dims: int = 2, max_vals: int = 6) -> dict:
    """One short line per table capturing ONLY what differs between sibling
    sheets: row count, the span of each date column, and the distinct values of
    the lowest-cardinality dimensions. Built from stats already in the model --
    no data rescan, no LLM. Same-schema sheets score identically in the table
    ranker, so this is what tells them apart in the prompt."""
    out = {}
    for name in model.table_names():
        t = model.tables.get(name)
        if t is None:
            continue
        bits = [f"{t.rows} rows"]
        for c in t.columns:
            if c.role in ("date", "year") and c.vmin is not None:
                bits.append(f"{c.name} {c.vmin}..{c.vmax}")
        dims = [c for c in t.columns
                if c.role == "dimension" and c.top_values and c.distinct <= 40]
        dims.sort(key=lambda c: c.distinct)
        for c in dims[:max_dims]:
            vals = ", ".join(str(v) for v, _ in c.top_values[:max_vals])
            more = c.distinct - min(len(c.top_values), max_vals)
            bits.append(f"{c.name}=[{vals}{f' +{more}' if more > 0 else ''}]")
        out[name] = "; ".join(bits)
    return out


def content_route(question, model, max_n: int = 3) -> list:
    """Deterministic pre-filter: narrow to the sheet(s) whose DATA matches a
    snapshot date, period or entity named in the question. Returns ordered
    table names, or [] when the question carries no distinguishing signal.

    Exists because the capability ranker scores same-schema sheets identically
    (a dead tie), leaving the planner to break the tie on prompt position --
    which always picks the same file. What actually separates monthly snapshots
    is the DATA they hold, not their columns."""
    import re
    from jarvisman.semantics.table_cards import filename_asof
    ql = (question or "").lower()

    # 'as at <date>' against a date-named snapshot: ONLY that file is valid --
    # a different snapshot does not cover that date.
    m = re.search(r"\b(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})\b", ql)
    if m:
        q_asat = f"{int(m.group(1)):02d}.{int(m.group(2)):02d}.{m.group(3)}"
        exact = [n for n in model.table_names() if filename_asof(n) == q_asat]
        if exact:
            return exact[:max_n]

    q_years = set(re.findall(r"\b(19\d{2}|20\d{2})\b", ql))
    months = ("jan", "feb", "mar", "apr", "may", "jun",
              "jul", "aug", "sep", "oct", "nov", "dec")
    q_months = {i + 1 for i, mo in enumerate(months) if mo in ql}
    # words of the question, for matching against table NAMES ('LTL', 'CY01',
    # 'deposit') -- a user who names the file/table must be routed to it even
    # when no date or cell value matches.
    q_words = set(re.findall(r"[a-z0-9]+", ql))
    _noise = {"data", "sheet", "sheet1", "table", "xlsx", "xls", "file",
              "the", "and", "all", "for"}
    scored = []
    for name in model.table_names():
        t = model.tables.get(name)
        if t is None:
            continue
        s = 0.0
        for nt in set(re.findall(r"[a-z0-9]+", str(name).lower())):
            if len(nt) >= 3 and nt not in _noise and nt in q_words:
                s += 2.0
        for c in t.columns:
            if c.role in ("date", "year") and c.vmin is not None:
                span = f"{c.vmin} {c.vmax}".lower()
                s += 3.0 * sum(1 for y in q_years if y in span)
                if c.role == "date":
                    s += 3.0 * sum(1 for mn in q_months
                                   if f"-{mn:02d}-" in str(c.vmin)
                                   or f"-{mn:02d}-" in str(c.vmax))
            if c.role == "dimension" and c.top_values:
                for v, _ in c.top_values:
                    if v and len(str(v)) >= 3 and str(v).lower() in ql:
                        s += 4.0
        if s:
            scored.append((s, name))
    scored.sort(key=lambda x: -x[0])
    return [n for _, n in scored[:max_n]]
