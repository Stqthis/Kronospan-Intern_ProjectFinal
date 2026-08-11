"""Resolve a company CODE typed in a question into the company NAME that the
data tables actually use, so the normal query path (plan OR codegen) can
answer it.

Exact normalized match is auto-applied. A fuzzy match (short names / variants)
is returned flagged so the caller can confirm before trusting it on money data.
"""
import re
from difflib import SequenceMatcher
from typing import Optional, Tuple


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s).strip().upper())


def looks_like_code(tok: str) -> bool:
    """A short letters+digits token like NT01, ABC1234 -- not a normal word."""
    return bool(re.fullmatch(r"[A-Za-z]{1,4}\d{1,5}[A-Za-z]?", tok.strip()))


def _company_mapping_columns(dataframes: dict):
    """Yield (df, code_col, name_col) for tables that map a COMPANY code to a
    COMPANY name. Deliberately company-specific: a table's code column must
    mention COMPANY (so CONTRACT_CODE/TRANCHE_NAME mapping tables are ignored),
    and its name column must be COMPANY_NAME or COMPANY."""
    for _, df in dataframes.items():
        cols = {c: str(c).strip().upper() for c in df.columns}
        code_c = next((c for c, u in cols.items()
                       if "CODE" in u and "COMPANY" in u), None)
        name_c = next((c for c, u in cols.items()
                       if u in ("COMPANY_NAME", "COMPANY")
                       or ("COMPANY" in u and "NAME" in u)), None)
        if code_c and name_c and code_c != name_c:
            yield df, code_c, name_c


def resolve_code(code: str, dataframes: dict) -> Optional[str]:
    """A company code (e.g. NT01) -> company name, via any company-mapping
    table. Returns None if not found."""
    target = _norm(code)
    for df, code_c, name_c in _company_mapping_columns(dataframes):
        try:
            hit = df[df[code_c].astype(str).map(_norm) == target]
        except Exception:
            continue
        if len(hit):
            return str(hit.iloc[0][name_c])
    return None


def find_company_column(df) -> Optional[str]:
    """The column in a data table that holds company names (not codes)."""
    for c in df.columns:
        u = str(c).strip().upper()
        if u == "COMPANY" or (("NAME" in u or "COMPANY" in u) and "CODE" not in u):
            return c
    return None


def match_name(name: str, df, company_col: str,
               threshold: float = 0.86) -> Tuple[Optional[str], bool, bool]:
    """Find `name` among df[company_col]. Returns (matched_value, is_exact,
    is_ambiguous). Exact-normalized first; else the best fuzzy candidate above
    `threshold`. is_ambiguous is True when the top two fuzzy scores are too
    close to call safely."""
    tn = _norm(name)
    try:
        vals = df[company_col].dropna().astype(str).unique()
    except Exception:
        return None, False, False
    norm_map = {v: _norm(v) for v in vals}
    for v, nv in norm_map.items():
        if nv == tn:
            return v, True, False
    tn_tokens = set(tn.split())
    scored = []
    for v, nv in norm_map.items():
        nv_tokens = set(nv.split())
        subset = tn_tokens <= nv_tokens or nv_tokens <= tn_tokens
        ratio = SequenceMatcher(None, tn, nv).ratio()
        score = max(ratio,
                    0.9 if subset and min(len(tn_tokens), len(nv_tokens)) >= 2 else 0.0)
        if score >= threshold:
            scored.append((score, v))
    scored.sort(reverse=True)
    if not scored:
        return None, False, False
    if len(scored) >= 2 and abs(scored[0][0] - scored[1][0]) < 0.05:
        return scored[0][1], False, True
    return scored[0][1], False, False


def resolve_in_question(question: str, dataframes: dict):
    """Rewrite any company code in `question` into the company name the data
    uses. Returns (new_question, notes, clarify) where:
      - new_question: the question with codes replaced by names (may equal input)
      - notes: list of human-readable strings describing what was resolved
      - clarify: None, or a string to ASK the user when a fuzzy match needs
        confirmation / is ambiguous (caller should surface it and stop).
    Exact matches are applied silently; fuzzy/ambiguous ones return a clarify."""
    notes: list = []
    for tok in re.findall(r"[A-Za-z]{1,4}\d{1,5}[A-Za-z]?", question):
        if not looks_like_code(tok):
            continue
        resolved = resolve_code(tok, dataframes)
        if not resolved:
            continue
        # verify the name against the data tables; decide exact vs fuzzy
        is_exact = False
        fuzzy_hit = None
        ambiguous = False
        for _, df in dataframes.items():
            col = find_company_column(df)
            if not col:
                continue
            matched, exact, amb = match_name(resolved, df, col)
            if matched and exact:
                is_exact = True
                break
            if matched and amb:
                ambiguous = True
            elif matched:
                fuzzy_hit = matched
        if ambiguous:
            return question, notes, (
                f"Code {tok} maps to '{resolved}', but I found more than one "
                f"close company name in the data. Which did you mean?")
        if not is_exact and fuzzy_hit is not None:
            return question, notes, (
                f"I read code {tok} as company '{resolved}', matched to "
                f"'{fuzzy_hit}' in the data. Is that correct?")
        # exact match (or resolved with a usable name) -> rewrite silently
        question = re.sub(re.escape(tok), resolved, question, count=1)
        notes.append(f"resolved company code {tok} to '{resolved}'")
    return question, notes, None