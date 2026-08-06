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