"""Resolve a company CODE typed in a question into the company NAME that the
data tables actually use, so the normal query path (plan OR codegen) can
answer it."""
import re
from difflib import SequenceMatcher
from typing import Optional, Tuple


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s).strip().upper())


def looks_like_code(tok: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]{1,4}\d{1,5}[A-Za-z]?", tok.strip()))


def _company_mapping_columns(dataframes: dict):
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
    for c in df.columns:
        u = str(c).strip().upper()
        if u == "COMPANY" or (("NAME" in u or "COMPANY" in u) and "CODE" not in u):
            return c
    return None


def match_name(name: str, df, company_col: str,
               threshold: float = 0.86) -> Tuple[Optional[str], bool, bool]:
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
    notes: list = []
    for tok in re.findall(r"[A-Za-z]{1,4}\d{1,5}[A-Za-z]?", question):
        if not looks_like_code(tok):
            continue
        resolved = resolve_code(tok, dataframes)
        if not resolved:
            continue
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
        question = re.sub(re.escape(tok), resolved, question, count=1)
        notes.append(f"resolved company code {tok} to '{resolved}'")
    return question, notes, None
