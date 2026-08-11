"""Interpretation ambiguity (Phase B+): ask the USER, in business language.

Two GENERAL patterns are detected deterministically (no LLM call), and each
produces ONE clarification with clickable options. Option labels NEVER show
raw column names -- they are phrased from the column MEANINGS captured at
index time (column_types / table cards), falling back to a humanised
description. The machine-readable part of each option (a `directive` for the
planner, or a column bind) is carried separately and never shown.

Pattern 1 -- value-vs-measure ("the EUR problem", generalised):
    A question term is BOTH a verified value of a dimension column (EUR in
    CURRENCY) AND part of a measure column of the same table (AMOUNT EURO).
    The question is genuinely two different questions:
      (a) restrict rows to that value, or
      (b) no restriction -- use the related (converted/equivalent) measure
          over ALL rows.
    Nothing in the text decides it, so the app must ask. The chosen option
    becomes a `directive` injected into the planner prompt on the re-run.
    The ask is SKIPPED when the question already decides it: it names the
    measure column itself ("total AMOUNT EUR of ..."), or uses deciding
    words ("equivalent", "converted", "denominated", ...).

Pattern 2 -- one value in several columns (CROATIA in COUNTRY and in
    COUNTRY OF BANK), asked ONLY when no other word of the question
    disambiguates. The chosen option is an ordinary column bind and re-uses
    the existing forced_bind machinery unchanged. Default OFF
    (RAG_COLUMN_CLARIFY=1 to enable): the long-standing policy of this app
    is that column choice belongs to the planner; this flag exists for
    deployments that prefer asking.

Both patterns are independently switchable and cost zero LLM calls.
"""

from __future__ import annotations

import re
from typing import Optional

from jarvisman import config as cfg
from jarvisman.semantics.value_index import norm_text, tokenize_unicode

# Words that mean the user ALREADY decided between the two readings of
# pattern 1, so asking would be noise. EN + EL (normalised: no accents).
# Words that mean the user ALREADY decided the "all rows, converted measure"
# reading -- sourced from config (RAG_DECIDED_WORDS), never hardcoded here.
# Whole-word OR prefix match (so 'converted'/'conversion' both hit). If the
# list is empty, the app always ASKS instead of auto-resolving.
def _build_word_re(words):
    parts = [re.escape(w) + r"\w*" for w in words if w]
    return re.compile(r"\b(" + "|".join(parts) + r")\b") if parts else None


_DECIDED_RE = _build_word_re(getattr(cfg, "INTERPRET_DECIDED_WORDS", []))

# A dimension column is "currency-like" when its name/meaning contains one of
# these words -- COSMETIC ONLY (nicer option labels). Sourced from config.
_CURRENCY_RE = _build_word_re(getattr(cfg, "INTERPRET_CURRENCY_WORDS", []))

_T = {
    "en": {
        "q": ("Your question can be read in two different ways -- "
              "which one do you mean?"),
        "only_ccy": "Only the amounts already in {v}",
        "all_ccy": "All amounts, converted to {v}",
        "only_gen": "Only the items whose {what} is {v}",
        "all_gen": "Everything, using the {what}",
        "as": "{v} as the {what}",
        "per_native": "Each {what}: original amounts, NOT converted",
        "per_conv": "All amounts converted to {m}",
        "q_col": ("'{v}' appears in more than one place in the data -- "
                  "which one do you mean?"),
    },
    "el": {
        "q": ("Η ερώτηση μπορεί να εννοεί δύο διαφορετικά πράγματα -- "
              "ποιο από τα δύο θέλετε;"),
        "only_ccy": "Μόνο τα ποσά που είναι ήδη σε {v}",
        "all_ccy": "Όλα τα ποσά, μετατρεμμένα σε {v}",
        "only_gen": "Μόνο τις εγγραφές όπου το {what} είναι {v}",
        "all_gen": "Όλα, με βάση το {what}",
        "as": "{v} ως {what}",
        "per_native": "Κάθε {what}: αρχικά ποσά, ΧΩΡΙΣ μετατροπή",
        "per_conv": "Όλα τα ποσά μετατρεμμένα σε {m}",
        "q_col": ("Το '{v}' υπάρχει σε περισσότερα από ένα σημεία στα "
                  "δεδομένα -- ποιο από τα δύο εννοείτε;"),
    },
}


def _t() -> dict:
    return _T["el"] if getattr(cfg, "UI_LANG", "en") == "el" else _T["en"]


def _col_profile(tprof, name: str):
    for cp in getattr(tprof, "columns", []) or []:
        if str(cp.name) == str(name):
            return cp
    return None


def _humanize(cp) -> str:
    """Business-language description of a column: its MEANING when the
    profile has one, otherwise the name itself made readable (lowercased,
    whitespace collapsed). Never empty."""
    meaning = (getattr(cp, "meaning", "") or "").strip()
    if meaning:
        meaning = re.split(r"[.;\n]", meaning)[0].strip()
        if meaning:
            return meaning[:70].rstrip()
    name = re.sub(r"\s+", " ", str(getattr(cp, "name", "")).strip())
    return name.lower() or "value"


def _short_label(cp) -> str:
    """A column's NAME made readable -- never its meaning sentence.

    _humanize returns the profile's MEANING, up to 70 characters. Dropped into
    "Each {what}'s own (original) amounts" that produced labels like
    "Each The currency in which the financial details are reported's own
    (original) amounts" -- unreadable, so the user could not see that the
    choice was local-currency vs converted, and picked the wrong reading.
    """
    n = re.sub(r"\s+", " ", str(getattr(cp, "name", "") or "")).strip()
    n = re.sub(r"\s*\((?:in\s+)?[A-Z]{3}\)\s*$", "", n)
    n = n.replace("_", " ").strip()
    if n.isupper() or n.islower():
        n = n.capitalize() if len(n) > 3 else n
    return (n[:32].rstrip() or "value")


def _is_currencyish(cp) -> bool:
    blob = norm_text(f"{getattr(cp, 'name', '')} {getattr(cp, 'meaning', '') or ''}")
    return bool(_CURRENCY_RE.search(blob)) if _CURRENCY_RE else False


def _question_names_measure(qn: str, mcol_name: str) -> bool:
    """True when the (normalised) question literally contains the measure
    column's own name ('total amount eur of ...' names 'AMOUNT EUR'):
    the user already pointed at the measure, so there is nothing to ask."""
    mn = norm_text(re.sub(r"\s+", " ", str(mcol_name)))
    if not mn:
        return False
    if mn in qn:
        return True
    # tolerate one-token prefix differences ('amount eur' vs 'amount euro')
    mtoks = tokenize_unicode(mn)
    qtoks = tokenize_unicode(qn)
    if len(mtoks) < 2:
        return False
    for i in range(len(qtoks) - len(mtoks) + 1):
        win = qtoks[i:i + len(mtoks)]
        if all(a == b or (min(len(a), len(b)) >= 3
                          and (a.startswith(b) or b.startswith(a)))
               for a, b in zip(win, mtoks)):
            return True
    return False


def _term_relates_to_measure(tn: str, cp) -> bool:
    """The question term ('eur') relates to a measure column ONLY when it is a
    genuine WHOLE-TOKEN of the column name ('eur' is a token of 'AMOUNT
    EURO'). The old version also matched the term as a bare substring of the
    column MEANING, which fired on coincidental letters -- e.g. a currency
    'MAD' or an ISO code 'ARE' appearing inside an unrelated meaning string,
    producing nonsense clarifications like 'company country ISO code is ARE'.
    Meaning-substring matching is removed; only a real name-token (with a
    light singular/plural and 'euro/eur' allowance) counts."""
    if len(tn) < 3:
        return False
    name_tokens = set(tokenize_unicode(norm_text(str(cp.name))))
    if tn in name_tokens:
        return True
    # tolerate euro/eur and trivial plural, but NOTHING looser
    variants = {tn, tn.rstrip("s"), tn + "s"}
    if tn.startswith("eur"):
        variants |= {"euro", "eur"}
    return bool(variants & name_tokens)


# --------------------------------------------------------------------------- #
# Pattern 1: value-vs-measure                                                 #
# --------------------------------------------------------------------------- #
def _value_vs_measure(question: str, evidence, model) -> Optional[dict]:
    qn = norm_text(question)
    decided = bool(_DECIDED_RE.search(qn)) if _DECIDED_RE else False
    T = _t()
    # STRICT gate (real-data find): the question term must BE a whole stored
    # value ('eur' == stored 'EUR'), found verbatim in the question. A mere
    # token hit ('bank' inside 'ECCM Bank PLC') produced nonsense options.
    hits = [h for h in (evidence.value_hits or [])
            if h.kind == "value_in_question"]
    for h in hits:
        tn = norm_text(h.matched_text)
        if len(tn) < 3 or norm_text(h.display) != tn:
            continue
        tprof = (model.tables or {}).get(h.table)
        if tprof is None:
            continue
        vcol = _col_profile(tprof, h.column)
        if vcol is None or getattr(vcol, "role", "") == "measure":
            continue
        for cp in tprof.columns:
            if str(cp.name) == str(h.column):
                continue
            is_measure = getattr(cp, "role", "") == "measure" or \
                getattr(cp, "semantic_type", "") == "number"
            if not is_measure or not _term_relates_to_measure(tn, cp):
                continue
            if _question_names_measure(qn, cp.name):
                return None  # the user already named the measure column
            v = str(h.display)
            ccy = _is_currencyish(vcol)
            what_v = _humanize(vcol)
            what_m = _humanize(cp)
            opt_only = {
                "label": (T["only_ccy"] if ccy else T["only_gen"]).format(
                    v=v, what=what_v),
                "table": h.table, "column": h.column, "display": v,
                "directive": (
                    f"treat '{h.matched_text}' as a FILTER: include ONLY rows "
                    f"where column '{h.column}' of table '{h.table}' has the "
                    f"value '{v}', then compute what the question asks over "
                    f"those rows."),
            }
            opt_all = {
                "label": (T["all_ccy"] if ccy else T["all_gen"]).format(
                    v=v, what=what_m),
                "table": h.table, "column": str(cp.name), "display": v,
                # machine-enforceable part of the choice: a plan that still
                # filters this column violates the user's explicit decision
                # and is corrected in CODE (reasoner), not trusted to the LLM
                "forbid_filter_column": h.column,
                "directive": (
                    f"'{h.matched_text}' does NOT restrict rows: do NOT "
                    f"filter column '{h.column}'. Use the measure column "
                    f"'{cp.name}' of table '{h.table}' over ALL rows."),
            }
            if decided:
                # 'equivalent'/'converted' already chose the 'all, converted'
                # reading. Do NOT ask -- but do NOT merely trust the prompt
                # either (the local model filtered CURRENCY anyway). Return an
                # AUTO-RESOLVED directive the reasoner enforces in code, with
                # the same forbid_filter_column guard as a user click.
                return {"kind": "auto_interpretation", "term": h.matched_text,
                        "directive": opt_all["directive"],
                        "forbid_filter_column": h.column}
            return {"kind": "interpretation", "term": h.matched_text,
                    "question": T["q"], "options": [opt_only, opt_all]}
    return None


# --------------------------------------------------------------------------- #
# Pattern 3: grouping BY a dimension whose values name a converted measure    #
# --------------------------------------------------------------------------- #
def _per_dimension_measure(question: str, evidence, model,
                           vindex) -> Optional[dict]:
    """'total funds per Currency': the table has BOTH a converted measure
    (a numeric column whose name/meaning embeds one of the dimension's own
    stored values -- AMOUNT EURO embeds the CURRENCY value EUR) and native
    measures. Grouping by that dimension is genuinely two questions:
      (a) each group's own (original) amounts, or
      (b) everything converted (the derived measure).
    Entirely data-driven: the dimension, its values, and the measures are all
    read from the live tables; nothing is named in code. A deciding word
    ('equivalent', 'converted' -- RAG_DECIDED_WORDS) auto-resolves to (b)."""
    if vindex is None:
        return None
    qn = norm_text(question)
    qtoks = set(tokenize_unicode(qn))
    # aggregation intent required, using the codebase's existing word set --
    # a plain listing ('show rows per currency') has no measure to choose
    from jarvisman.semantics.grounding import _AGG_SKIP
    if not (qtoks & set(_AGG_SKIP)):
        return None
    T = _t()
    decided = bool(_DECIDED_RE.search(qn)) if _DECIDED_RE else False
    for tname, tprof in (model.tables or {}).items():
        for dcol in tprof.columns:
            if getattr(dcol, "role", "") != "dimension":
                continue
            dn = norm_text(str(dcol.name))
            # the question must reference the dimension column by name
            if not dn or dn not in qn:
                continue
            dom = vindex.domains.get((tname, str(dcol.name))) or {}
            if not (2 <= len(dom) <= 200):
                continue
            converted, native = [], []
            for cp in tprof.columns:
                if str(cp.name) == str(dcol.name):
                    continue
                if getattr(cp, "role", "") != "measure":
                    continue
                # a measure is "converted for THIS dimension" only when its
                # NAME embeds one of the dimension's own values as a whole
                # token (CURRENCY value 'EUR' -> 'AMOUNT EURO'). This is the
                # currency-conversion shape; a category measure like
                # 'Finance/ECCM' matches nothing here.
                rel = any(len(v) >= 3 and _term_relates_to_measure(v, cp)
                          for v in dom)
                (converted if rel else native).append(cp)
            if not converted or not native:
                continue
            # extra safety: only offer this for a currency-like dimension
            # (its name or the converted measure says currency/ccy). Without
            # this the pattern speculates on unrelated dimensions.
            _blob = norm_text(str(dcol.name) + " " +
                              getattr(dcol, "meaning", "") + " " +
                              str(converted[0].name))
            if _CURRENCY_RE and not _CURRENCY_RE.search(_blob):
                continue
            mc = converted[0]
            what_d = _short_label(dcol)
            what_m = _short_label(mc)
            opt_native = {
                "label": T["per_native"].format(what=what_d),
                "table": tname, "column": str(dcol.name),
                "forbid_agg_column": str(mc.name),
                "directive": (
                    f"group by column '{dcol.name}' of table '{tname}' and "
                    f"aggregate each group's ORIGINAL amounts: do NOT "
                    f"aggregate the converted column '{mc.name}'; pick the "
                    f"measure that holds the original (unconverted) values."),
            }
            opt_conv = {
                "label": T["per_conv"].format(m=what_m),
                "table": tname, "column": str(mc.name),
                "directive": (
                    f"group by column '{dcol.name}' of table '{tname}' and "
                    f"aggregate the converted measure column '{mc.name}'."),
            }
            if decided:
                return {"kind": "auto_interpretation",
                        "term": str(dcol.name),
                        "directive": opt_conv["directive"]}
            return {"kind": "interpretation", "term": str(dcol.name),
                    "question": T["q"], "options": [opt_native, opt_conv]}
    return None


# --------------------------------------------------------------------------- #
# Pattern 2: one value, several columns (opt-in)                              #
# --------------------------------------------------------------------------- #
def _value_in_multiple_columns(question: str, evidence, model) -> Optional[dict]:
    T = _t()
    qtoks = set(tokenize_unicode(norm_text(question)))
    by_text: dict = {}
    for h in (evidence.value_hits or []):
        if h.kind != "value_in_question":
            continue
        by_text.setdefault(norm_text(h.matched_text), []).append(h)
    for tn, hs in by_text.items():
        if len(tn) < 4:
            continue
        cols: dict = {}
        for h in hs:
            cols.setdefault((h.table, h.column), h)
        if len(cols) < 2:
            continue
        colnames = [c for (_, c) in cols]
        # another word of the question already points at exactly one of the
        # candidate columns ('... bank ...' -> COUNTRY OF BANK): do not ask.
        term_toks = set(tokenize_unicode(tn))
        decided = False
        for tok in qtoks - term_toks:
            if len(tok) < 3:
                continue
            present = [c for c in colnames
                       if tok in tokenize_unicode(norm_text(c))]
            if len(present) == 1 and len(colnames) > 1:
                decided = True
                break
        if decided:
            continue
        options = []
        for (tbl, c), h in list(cols.items())[:4]:
            tprof = (model.tables or {}).get(tbl)
            cp = _col_profile(tprof, c) if tprof is not None else None
            what = _humanize(cp) if cp is not None else str(c).lower()
            options.append({"label": T["as"].format(v=h.display, what=what),
                            "table": tbl, "column": c,
                            "display": str(h.display)})
        if len(options) >= 2:
            return {"kind": "value_column", "term": hs[0].matched_text,
                    "question": T["q_col"].format(v=hs[0].display),
                    "options": options}
    return None


# --------------------------------------------------------------------------- #
def find_interpretation_ambiguity(question: str, evidence, model,
                                  vindex=None) -> Optional[dict]:
    """Entry point used by grounding.find_ambiguity. Deterministic, 0 LLM
    calls. Returns the same dict shape as the existing typo detector:
    {"kind", "term", "question", "options": [...]}, where options may carry
    a `directive` (interpretation choice) or a table/column (ordinary bind).
    """
    if model is None or not getattr(model, "tables", None):
        return None
    if not question or evidence is None:
        return None
    if getattr(cfg, "INTERPRET_CLARIFY", True):
        amb = _value_vs_measure(question, evidence, model)
        if amb:
            return amb
        amb = _per_dimension_measure(question, evidence, model, vindex)
        if amb:
            return amb
    if getattr(cfg, "COLUMN_CLARIFY", False):
        amb = _value_in_multiple_columns(question, evidence, model)
        if amb:
            return amb
    return None
