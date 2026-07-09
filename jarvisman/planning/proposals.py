"""Alternative proposals: when the system cannot answer as asked, let the
UNDERSTANDING model propose a concrete way forward -- in the user's language,
phrased by the model, not from a template.

This fires only AFTER the deterministic tiers have failed (plan rejected,
a stated condition could not be applied, or codegen produced nothing). The
model is given:
  * the user's question,
  * a plain-language description of WHY it could not be answered
    (the validator's issues + the decision trace), and
  * the real schema (table/column names + meanings),
and is asked to return ONE short proposal as JSON:
    {"proposal": "<one question to the user, in their language>",
     "accept_directive": "<instruction to apply if the user accepts>"}

The proposal is shown to the user as a clarify with two options ("Yes, do
that" / "No"). On "yes", accept_directive is injected and the question is
re-run through the normal enforced-directive path -- the same machinery a
clicked interpretation uses. Nothing here is hardcoded to currencies, dates,
or any column: the model decides what to propose from the failure + schema.

Disabled with RAG_PROPOSE=0. Costs one understanding-model call, only on the
failure path (never on a normal answer).
"""

from __future__ import annotations

import json
from typing import Optional

from jarvisman import config as cfg
from jarvisman.llm.llm_json import extract_json


_PROPOSE_SYSTEM = (
    "You are helping a non-technical user query their spreadsheet data. A "
    "question could NOT be answered as asked. Propose ONE concrete way "
    "forward as a single short question the user can say yes/no to. Speak in "
    "the user's own language. Do NOT mention column names, tables, code, or "
    "internal mechanics -- talk about the DATA in business terms. If a "
    "reasonable assumption could rescue the question (e.g. the file may "
    "represent a single point in time, or a term might mean something "
    "slightly different), propose exactly that. Reply with ONLY a JSON "
    "object: {\"proposal\": \"<your yes/no question>\", "
    "\"accept_directive\": \"<a precise instruction to follow if the user "
    "says yes; name the concrete columns/tables here since this part is for "
    "the system, not the user>\"}. If there is genuinely no sensible "
    "alternative, reply {\"proposal\": \"\", \"accept_directive\": \"\"}."
)


def _reason_text(issues, trace) -> str:
    parts = []
    for i in (issues or []):
        msg = getattr(i, "message", None) or str(i)
        parts.append(f"- {msg}")
    if trace:
        parts.append("Decision trace:")
        parts.extend(f"  {t}" for t in trace[-8:])
    return "\n".join(parts) if parts else "- the question could not be matched to the data"


def propose_alternative(ollama, chat_model: str, question: str,
                        schema_block: str, issues, trace,
                        progress=None) -> Optional[dict]:
    """Return {"proposal", "accept_directive"} or None. Uses the
    understanding model. Never raises."""
    if ollama is None or not getattr(cfg, "PROPOSE_ALTERNATIVES", True):
        return None
    user = (f"User question:\n{question}\n\n"
            f"Why it could not be answered:\n{_reason_text(issues, trace)}\n\n"
            f"The available data:\n{schema_block}\n\n"
            "Your JSON proposal:")
    try:
        if progress:
            progress("Thinking of an alternative ...")
        raw = ollama.chat(
            cfg.model_for("propose", chat_model),
            [{"role": "system", "content": _PROPOSE_SYSTEM},
             {"role": "user", "content": user}],
            options={"temperature": 0.3, "num_predict": 300},
            format="json",
        )
    except Exception:
        return None
    parsed = extract_json(raw)
    if not isinstance(parsed, dict):
        return None
    proposal = str(parsed.get("proposal") or "").strip()
    directive = str(parsed.get("accept_directive") or "").strip()
    if not proposal or not directive:
        return None
    return {"proposal": proposal, "accept_directive": directive}
