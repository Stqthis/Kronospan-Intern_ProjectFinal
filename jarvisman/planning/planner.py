"""Planner: turn the question + semantic model + evidence into a QueryPlan.

One LLM call produces a small JSON plan; the PlanValidator/compiler then
binds and checks it. If validation finds hard issues, the model gets ONE
repair call whose content is the precise, structured problem ("value
'Croacia' not found in COUNTRY; closest actual values: 'CROATIA'") -- far
more repairable than a runtime traceback. Anything the schema cannot express
is routed to the codegen fallback via ``requires_code``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from jarvisman import config as cfg
from jarvisman.semantics.grounding import Evidence
from jarvisman.llm.llm_json import extract_json
from jarvisman.planning.query_plan import CodeBundle, QueryPlan, compile_plan
from jarvisman.semantics.semantic_model import SemanticModel, render_for_prompt
from jarvisman.semantics.value_index import ValueIndex

PLAN_GUIDE = """You convert a user question about spreadsheet data into a structured JSON query plan.

### REQUIRED OUTPUT SHAPE
The output JSON MUST contain every single top-level key shown below, even when empty or completely unutilized. Never omit keys.
Use:
- [] for empty arrays
- null for unassigned objects or variables
- false for unasserted booleans

{
  "table": "<exact table name>",
  "join": null or {"right_table": "...", "left_column": "...", "right_column": "..."},
  "filters": [
    {"column": "<exact column>", "op": "match|eq|ne|in|contains|gt|ge|lt|le|between|isnull|notnull", "value": <string|number|list>, "or_null": true|false}
  ],
  "group_by": [],
  "aggregations": [
    {"fn": "sum|mean|median|min|max|count|nunique", "column": "<exact column or null>", "alias": "<name>"}
  ],
  "having": [],
  "derived": [
    {"alias": "<name>", "kind": "pct_of_total|ratio", "num": "<agg_alias>", "den": "<agg_alias_for_ratio_only>"}
  ],
  "union": [],
  "compare": null or {
    "left_table": "...", "right_table": "...", "key": "<entity key column>", "measures": ["<measure>"], "left_label": "...", "right_label": "..."
  },
  "select": [],
  "sort": null or {"by": "<column or alias>", "desc": true|false},
  "limit": null or <int>,
  "requires_code": false,
  "clarification": null,
  "intent_summary": "<one short sentence summarizing user intent>"
}

### CRITICAL ACCURACY RULES
1. NEVER invent, infer, normalize, abbreviate, or modify table names or column names. Use ONLY names appearing exactly as written in the provided schema definitions.
2. NEVER guess the target table. If multiple tables could plausibly satisfy the request and the schema data does not provide a unique structural mapping, immediately stop and set "clarification": "<short precise question to user>".
3. APPLY EVERY condition the user stated. A dropped filter, date restriction, or parameter is an absolute wrong answer. You must account for all logic elements in multi-clause questions.
4. COLUMN SELECTION & EVIDENCE: You MUST select the target column based on its semantic meaning in the Schema block. Do not blindly filter a column just because a word was found there in the Evidence block. Use the Evidence ONLY to check how to spell the value. If the exact value is not in the Evidence, write the exact string the user typed so the downstream validator can resolve it.
5. PRIORITIZE PRE-CALCULATED COLUMNS: Read column descriptions carefully. If a column directly satisfies the requested currency, type, or unit (e.g., an 'AMOUNT EUR' column for a Euro calculation), you MUST select and aggregate that specific column directly over ALL rows. Do NOT create filters on a separate 'currency' column if a pre-converted measure column exists. If multiple parallel candidate measure columns exist (e.g., REVENUE, REVENUE EUR, REVENUE USD) and the user query is ambiguous, trigger a clarification question instead of guessing.
6. USER INTERPRETATIONS: If the Evidence contains a user-confirmed interpretation, treat it as authoritative and apply its instructions exactly. Enforced directives override all standard rules.

### DECISION PRIORITY HIERARCHY
When processing ambiguous, complex, or incomplete inputs, apply this absolute order of operations:
1. If vital information or specific structural clarity is missing -> populate "clarification" and leave all query metrics empty.
2. Else if the logical request is valid but cannot be represented within this strict JSON schema -> set "requires_code": true and leave query metrics empty.
3. Else -> generate the standard, complete JSON execution plan.
Never populate both "requires_code": true and a "clarification" string simultaneously.

### PLAN MINIMALITY & STRUCTURAL INTEGRITY CONTRACT
- All top-level keys specified in the REQUIRED OUTPUT SHAPE must ALWAYS exist in the response.
- Minimality constraints apply strictly to the CONTENT inside arrays and objects, never to the omission of schema keys.
- Do not populate optional parameters unless required by the request: No group_by entries unless grouping is explicitly stated; no sort/limit entries unless requested or required by highest/lowest logic.
- SELECT ARRAY RULES: An empty "select": [] array explicitly instructs the compiler to return all columns for the slice. Populate the "select" array with explicit strings ONLY if the user names specific fields to display.

### STRICT STRUCTURAL SYNTAX MAPPINGS

AGGREGATION VOCABULARY CONTRACT:
- total, sum -> "sum"
- average, avg, mean -> "mean"
- median -> "median"
- minimum, lowest, smallest -> "min"
- maximum, highest, largest -> "max"
- count, how many -> "count"
- unique count, distinct count -> "nunique"

HAVING FORMAT SPECIFICATION:
The "having" array filters aggregate results and must strictly use this inner condition structure:
[{"column": "<aggregation_alias>", "op": "gt|ge|lt|le|eq|ne", "value": <number>}]

DATE RULES:
- If the schema column description or label explicitly defines it as a date/timestamp field, convert referenced years, months, and quarters into date range barriers using the "between" operator.
- If the schema column description or label defines it as an integer numeric period profile (e.g., YEAR, FISCAL_YEAR), use exact equality comparisons ("eq").
- Match the schema profile types exactly as verified on load; do not infer type based solely on header string text guessing.

DERIVED METRICS & ALIAS SCOPE:
- For "share of total" operations: Declare a derived object with kind="pct_of_total". Note that the proportional division is calculated relative to the entire column's total sum unless a sub-grouping context is explicitly required by the text.
- ALIAS UNIQUENESS: All generated names inside "alias" strings must be unique across BOTH the "aggregations" array and the "derived" array to prevent data overwrites during execution.

JOIN SAFETY FRAMEWORK:
A join block may ONLY be emitted if both tables exist, the target joining key columns exist, and the relationship is explicitly stated and verified in the schema block. If an unverified join is required to answer the query, set "requires_code": true or request clarification.

### OUTPUT REQUIREMENTS CONTRACT
- Return EXACTLY one valid JSON object.
- Do NOT wrap the JSON in conversational text, trailing remarks, or structural prose.
- Do NOT include comments, explanations, or markdown code fences inside or outside the JSON string.

### MANDATORY PRE-FLIGHT SELF-CHECK
Before emitting the final JSON string, systematically verify:
1. Is every user condition and filter represented? (Missing elements = failure).
2. Do all referenced tables and columns exist verbatim in the schema?
3. Does the JSON contain every single schema output key, utilizing null or [] where empty?
4. Are all operators and functions strictly supported by the vocabulary contract?
5. Did you avoid inventing or inferring an unverified value or column relationship?

### EXAMPLES

User: "total <measure> for <category-value> items"  (a filter + an aggregate)
Read the schema: pick the column whose described meaning matches <measure> as
the aggregation target, and the column that contains <category-value> as the
filter. Copy the value spelling from the Evidence verbatim.
{
  "table": "<the table whose columns fit, from the schema>",
  "join": null,
  "filters": [{"column": "<the column that holds the category>", "op": "match", "value": "<value exactly as in the Evidence>", "or_null": false}],
  "group_by": [],
  "aggregations": [{"fn": "sum", "column": "<the measure column per its meaning>", "alias": "total"}],
  "having": [],
  "derived": [],
  "union": [],
  "compare": null,
  "select": [],
  "sort": null,
  "limit": null,
  "requires_code": false,
  "clarification": null,
  "intent_summary": "<one sentence: the measure summed and the filter applied>"
}

User: "total revenue"
{
  "table": null,
  "join": null,
  "filters": [],
  "group_by": [],
  "aggregations": [],
  "having": [],
  "derived": [],
  "union": [],
  "compare": null,
  "select": [],
  "sort": null,
  "limit": null,
  "requires_code": false,
  "clarification": "Which revenue measure should be used: REVENUE EUR or REVENUE USD?",
  "intent_summary": "User wants a revenue total but the measure is ambiguous."
}

User: "share of total revenue by country"
{
  "table": "sales.xlsx:Data",
  "join": null,
  "filters": [],
  "group_by": ["COUNTRY"],
  "aggregations": [{"fn": "sum", "column": "REVENUE", "alias": "sum_revenue"}],
  "having": [],
  "derived": [{"alias": "revenue_share", "kind": "pct_of_total", "num": "sum_revenue", "den": null}],
  "union": [],
  "compare": null,
  "select": [],
  "sort": null,
  "limit": null,
  "requires_code": false,
  "clarification": null,
  "intent_summary": "Calculate the proportional share of total revenue contributed by each individual country."
}
"""

@dataclass
class PlanResult:
    status: str                       # 'ok' | 'clarify' | 'fallback'
    bundle: Optional[CodeBundle] = None
    plan: Optional[QueryPlan] = None
    issues: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    clarification: Optional[str] = None
    raw: str = ""
    llm_calls: int = 0


class Planner:
    _cards = None
    _coverage_anchors = None
    def __init__(self, ollama, chat_model: str) -> None:
        self.ollama = ollama
        self.chat_model = chat_model

    # ------------------------------------------------------------------ #
    def _user_prompt(self, question: str, schema_block: str,
                     evidence: Evidence, extra_evidence: str = "") -> str:
        ev = extra_evidence or (evidence.to_prompt_block() if evidence else "")
        from jarvisman.runtime import house_rules
        hr = house_rules.prompt_block()
        return (f"Schema of the available tables:\n{schema_block}\n\n"
                f"{ev}\n"
                f"{hr}"
                f"User question: {question}\n\n"
                "JSON plan:")

    def _chat(self, system: str, user: str) -> str:
        return self.ollama.chat(
            cfg.model_for("plan", self.chat_model),
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            options={"temperature": cfg.PLAN_TEMPERATURE},
        )

    # ------------------------------------------------------------------ #
    def make_plan(self, question: str, model: SemanticModel, dataframes: dict,
                  vindex: ValueIndex, evidence: Evidence,
                  progress: Optional[Callable[[str], None]] = None,
                  extra_evidence: str = "") -> PlanResult:
        schema_block = render_for_prompt(model, list(dataframes.keys()),
                                          cards=self._cards)
        user = self._user_prompt(question, schema_block, evidence, extra_evidence)
        calls = 0
        raw = ""
        try:
            if progress:
                progress("Planning the query ...")
            raw = self._chat(PLAN_GUIDE, user)
            calls += 1
        except Exception as exc:
            return PlanResult(status="fallback", issues=[f"planner error: {exc}"],
                              llm_calls=calls)

        parsed = extract_json(raw)
        if parsed is None:
            # one re-ask, then give up to the codegen fallback
            try:
                raw = self._chat(PLAN_GUIDE, user + "\n\nReply with ONLY the JSON object.")
                calls += 1
                parsed = extract_json(raw)
            except Exception:
                parsed = None
        if parsed is None:
            return PlanResult(status="fallback", raw=raw, llm_calls=calls,
                              issues=["model did not return a parseable JSON plan"])

        plan = QueryPlan.from_dict(parsed)
        if plan.clarification:
            return PlanResult(status="clarify", plan=plan, raw=raw,
                              clarification=plan.clarification, llm_calls=calls)
        if plan.requires_code:
            return PlanResult(status="fallback", plan=plan, raw=raw, llm_calls=calls,
                              notes=["planner chose codegen (requires_code)"])

        bundle, issues, notes = compile_plan(plan, model, dataframes, vindex, cards=self._cards, coverage_anchors=self._coverage_anchors)
        repairs = 0
        while bundle is None and repairs < cfg.PLAN_REPAIR_ATTEMPTS:
            repairs += 1
            if progress:
                progress("Repairing the query plan ...")
            issue_text = "\n".join(f"- {i.message}" for i in issues)
            note_text = ("\n".join(f"- {n}" for n in notes)) if notes else ""
            repair_user = (
                f"{user}\n\nYour previous plan was:\n{raw}\n\n"
                f"It has these problems:\n{issue_text}\n"
                + (f"Resolution notes:\n{note_text}\n" if note_text else "")
                + "\nReturn a corrected JSON plan only. Use the exact column names "
                  "and the closest actual values suggested above. If it cannot be "
                  "fixed within the schema, set \"requires_code\": true.")
            try:
                raw = self._chat(PLAN_GUIDE, repair_user)
                calls += 1
            except Exception as exc:
                return PlanResult(status="fallback", plan=plan, issues=issues,
                                  notes=notes, raw=raw, llm_calls=calls)
            parsed = extract_json(raw)
            if parsed is None:
                break
            plan = QueryPlan.from_dict(parsed)
            if plan.clarification:
                return PlanResult(status="clarify", plan=plan, raw=raw,
                                  clarification=plan.clarification, llm_calls=calls)
            if plan.requires_code:
                return PlanResult(status="fallback", plan=plan, raw=raw,
                                  llm_calls=calls,
                                  notes=["planner chose codegen on repair"])
            bundle, issues, notes = compile_plan(plan, model, dataframes, vindex, cards=self._cards, coverage_anchors=self._coverage_anchors)

        if bundle is None:
            return PlanResult(status="fallback", plan=plan, issues=issues,
                              notes=notes, raw=raw, llm_calls=calls)
        return PlanResult(status="ok", bundle=bundle, plan=plan, issues=issues,
                          notes=notes, raw=raw, llm_calls=calls)
