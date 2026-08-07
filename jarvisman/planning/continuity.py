"""Conversation continuity: carry the last company across implicit follow-ups
("what about this one?", "and their interest rate?"), and answer meta questions
about the previous result ("are you sure?", "is that correct?") by
RE-EXPLAINING it rather than recomputing a possibly-different number.

Deliberately pattern-based (no extra LLM call, no risk of a doubt-prompt making
the model change a correct figure). It carries the SUBJECT forward; the existing
analysis pipeline does the actual computing.
"""
import re
from typing import Optional, Tuple

# "are you sure / is that right / verify" -> re-explain the last answer
_REEXPLAIN_RE = re.compile(
    r"\b(are you sure|are u sure|you sure|is (that|this|it) (correct|right|accurate)|"
    r"is it correct|double.?check|verify (that|this|it)|can you confirm|"
    r"is that accurate|really\?|prove it|how did you get)\b", re.I)

# implicit-subject follow-ups that should inherit the last company
_IMPLICIT_RE = re.compile(
    r"\b(what about|how about|and (for )?(this|that|them|it|those)|"
    r"same (for|question for)|for (this|that|it|them|those)( one)?|"
    r"and (their|its|the))\b", re.I)

_BARE_RE = re.compile(
    r"^(what about|how about|and)\s+(this|that|them|it|those)( one)?$", re.I)

_CODE_RE = re.compile(r"\b[A-Za-z]{1,4}\d{1,5}[A-Za-z]?\b")

# ---- requests to EDIT the previous result table --------------------------
# "remove the last row", "delete the first two lines", "drop the bottom row".
# A QueryPlan has no field that can express positional row removal (there is
# table/filters/group_by/aggregations/select/sort/limit and nothing else), so
# these fall through the planner into raw codegen and improvise. Catching them
# here turns an unpredictable failure into a clear, honest answer.
_EDIT_VERB_RE = re.compile(
    r"\b(remove|delete|drop|exclude|hide|omit|get rid of|take out|"
    r"αφαιρεσε|αφαιρεση|σβησε|διαγραψε|βγαλε)\b", re.I)

# Positional -- the row is identified by WHERE IT SITS, not by what it holds.
_POSITIONAL_ROW_RE = re.compile(
    r"\b(last|first|final|top|bottom|second|third|\d+(st|nd|rd|th))\s+"
    r"(\d+\s+)?(row|rows|line|lines|record|records|entry|entries|result|results)\b"
    r"|\b(row|rows|line|lines)\s+(number\s+)?\d+\b"
    r"|\b(τελευται\w*|πρωτ\w*)\s+(γραμμ\w*|σειρ\w*|εγγραφ\w*)\b", re.I)


def is_positional_row_edit(question: str) -> bool:
    """The user is asking to strike rows from the PREVIOUS result by position.

    Deliberately narrow: it needs BOTH a removal verb AND a positional row
    reference. 'remove the total row' or 'excluding Lignum' name a VALUE, are
    expressible as an ordinary filter, and must keep flowing to the planner.
    """
    q = question or ""
    return bool(_EDIT_VERB_RE.search(q) and _POSITIONAL_ROW_RE.search(q))


def positional_row_edit_reply(last_question: str = "") -> str:
    """Explain the limit and hand back a phrasing that DOES work."""
    base = (
        "I can't remove rows by position -- an answer here is computed from "
        "the source data by a query, not edited afterwards, so there is no "
        "'last row' for me to strike out. Row order also comes from the Excel "
        "file itself, so a position-based answer would change the moment the "
        "sheet is re-exported.\n\n"
        "I can exclude rows by what they CONTAIN, which is stable. For example "
        "'excluding the Total row', 'without Lignum', or 'only rows where "
        "AMOUNT EURO is above 0'.")
    if last_question:
        base += (f"\n\nYour previous question was: \"{last_question.strip()}\" "
                 "-- tell me which rows to leave out and I'll re-run it.")
    return base


# Columns whose equality-filter value is a plausible conversational SUBJECT.
# Matched against the bound plan's column names, so a hit means the pipeline
# genuinely filtered on that column -- not that the word appeared in the text.
_SUBJECT_COL_RE = re.compile(
    r"(name|company|customer|client|supplier|vendor|bank|account|entity|"
    r"partner|counterparty|branch|plant|site|debtor|creditor|"
    r"επωνυμ|ονομα|πελατ|προμηθευτ|τραπεζ)", re.I)

# Never carry a bare year/number/date forward as if it were a subject.
_NOT_A_SUBJECT_RE = re.compile(r"^[\d\s.,:/-]+$")


def subject_from_plan(plan: Optional[dict]) -> str:
    """Extract the conversational subject from the LAST EXECUTED plan.

    This is ground truth: the plan is stored in bound form, so the value here
    is a real cell value that the query actually filtered on -- far more
    reliable than regexing the user's raw text, and it costs nothing.

    Preference order: an equality/match filter on a name-ish column, then any
    equality/match filter carrying a non-numeric string.
    """
    if not plan:
        return ""
    best = ""
    for f in (plan.get("filters") or []):
        if f.get("op") not in ("eq", "match"):
            continue
        val = f.get("value")
        if isinstance(val, (list, tuple)):
            val = val[0] if len(val) == 1 else None
        if not isinstance(val, str):
            continue
        val = val.strip()
        if not val or _NOT_A_SUBJECT_RE.match(val):
            continue
        if _SUBJECT_COL_RE.search(str(f.get("column", ""))):
            return val          # name-ish column wins immediately
        if not best:
            best = val          # otherwise remember the first plausible one
    return best


def is_reexplain(question: str) -> bool:
    """The user is questioning/verifying the PREVIOUS answer, not asking a new
    data question."""
    return bool(_REEXPLAIN_RE.search(question or ""))


def _names_subject(question: str, last_company: Optional[str]) -> bool:
    ql = (question or "").lower()
    if last_company and last_company.lower() in ql:
        return True
    return bool(_CODE_RE.search(question or ""))


def wants_carryover(question: str, last_company: Optional[str]) -> bool:
    """An implicit follow-up with no company of its own -> needs the remembered
    one."""
    if not last_company:
        return False
    if _names_subject(question, last_company):
        return False
    return bool(_IMPLICIT_RE.search(question or ""))


def expand_followup(question: str, last_question: Optional[str],
                    last_company: str) -> str:
    """Rewrite an implicit follow-up so it carries the remembered company."""
    ql = (question or "").strip().rstrip("?")
    if _BARE_RE.match(ql) and last_question:
        # "what about this one?" -> re-run the last question for the same company
        if last_company.lower() in last_question.lower():
            return last_question
        return f"{last_question.rstrip('?')} for {last_company}?"
    # contentful follow-up ("and their interest rate") -> attach the subject
    cleaned = re.sub(r"\b(their|its|this|that|them|it)\b", "",
                     question, flags=re.I)
    cleaned = re.sub(r"^\s*(and|what about|how about)\s+", "",
                     cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().rstrip("?")
    return f"{cleaned} for {last_company}?"


def reexplain_prompt(last_question: str, last_answer: str) -> str:
    """Build the system+user text for a re-explanation. The model must NOT
    produce a new figure -- it explains and stands behind the one already
    computed."""
    return (
        "The user is asking you to confirm or double-check your PREVIOUS "
        "answer. Do NOT calculate anything new and do NOT change any number. "
        "Re-state the previous answer and explain clearly how it was reached "
        "(which company, which filter/date, what was summed), so the user can "
        "trust it. If they doubt it, invite them to ask the question a "
        "different way rather than inventing a different figure.\n\n"
        f"Your previous answer was:\n{last_answer}\n\n"
        f"It answered the question:\n{last_question}\n\n"
        "Reply in the same language as the user, in 2-4 sentences.")