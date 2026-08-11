"""TableReasoner: the reasoning pipeline for questions over the tables.

    follow-up rewrite -> clarify check -> ground -> plan -> compile ->
    execute (sandbox) -> deterministic answer (+optional LLM phrasing)

Grounding, validation, follow-up rewriting and clarification detection are
all deterministic; the LLM is used exactly where judgement is needed (the
plan, one optional repair, optional phrasing). Charts ride the same pipeline:
the data part is planned and compiled identically, then a deterministic
matplotlib block renders it through the existing plot sandbox.

If the plan tier cannot express or fix the question, the reasoner returns a
fallback signal carrying the evidence block, and the Agent runs its existing
codegen path with that evidence injected -- the fallback is strictly
better-informed than the old system, never worse.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from jarvisman import config as cfg
from jarvisman.planning import followup
from jarvisman.planning import tier05
from jarvisman.semantics.grounding import coverage_anchors
from jarvisman.semantics.grounding import Evidence, find_ambiguity, ground
from jarvisman.planning.planner import Planner
from jarvisman.planning.query_plan import QueryPlan, compile_plan
from jarvisman.runtime.sandbox import run_query, run_sandboxed, validate_code
from jarvisman.semantics.semantic_model import SemanticModel
from jarvisman.semantics.value_index import ValueIndex, norm_text

_PROV_RE = re.compile(r"^PROV (.+?): (\d+) rows?", re.MULTILINE)
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_CHART_WORD_RE = re.compile(
    r"\b(plot|chart|graph|draw|visuali[sz]e|γραφημα|γράφημα|διαγραμμα|"
    r"διάγραμμα|πιτα|πίτα)\b", re.IGNORECASE)


_PROSE_INTENT_RE = re.compile(
    r"\b(compare[ds]?|comparison|versus|vs\.?|differenc\w*|"
    r"explain|why|reason\w*|"
    r"opinion|think|thoughts|view|recommend\w*|suggest\w*|advi[sc]e|"
    r"assess\w*|evaluate|analys[ei]s|interpret\w*|"
    r"summar\w+|overview|insight\w*|takeaway\w*|"
    r"better|best|worse|worst|strongest|weakest|riskiest|safest|"
    r"trend\w*|pattern\w*|notable|unusual|stands? ?out|concerning|"
    r"should (we|i|they)|what do you)\b", re.I)

_LOOSE_RE = re.compile(r"[^0-9a-z\u0370-\u03ff]+")


def sibling_choice(question, model, dataframes, max_opts: int = 12):
    """No period/entity signal + several SAME-SCHEMA sheets -> ask which one.

    content_route() settles any question that names a date, period or entity.
    What is left is genuinely ambiguous ("what is the total?" across 12 monthly
    files) and must be asked about, not guessed -- guessing silently is the bug
    this whole path exists to kill.

    -> (clarify_question, options) or (None, None)."""
    from jarvisman.semantics.semantic_model import content_route
    from jarvisman.semantics.table_cards import filename_asof
    names = [n for n in model.table_names() if n in dataframes]
    if len(names) < 2 or content_route(question, model):
        return None, None                # a signal exists -> routing handles it
    sig = {}
    for n in names:
        t = model.tables.get(n)
        if t is None:
            continue
        key = tuple(sorted(str(c.name) for c in t.columns))
        sig.setdefault(key, []).append(n)
    if not sig:
        return None, None
    group = max(sig.values(), key=len)
    if len(group) < 2:
        return None, None                # schemas differ -> not this problem
    opts = []
    for n in group[:max_opts]:
        d = filename_asof(n)
        opts.append({"label": d or n.split(":")[0], "table": n,
                     "suffix": f" as at {d}" if d else f" in file {n.split(':')[0]}"})
    opts.sort(key=lambda o: o["label"])
    return ("Those sheets all hold the same kind of data. "
            "Which one do you mean?", opts)


@dataclass
class ReasonerOutcome:
    kind: str                       # 'ok' | 'clarify' | 'fallback'
    text: str = ""
    table_html: Optional[str] = None
    code: Optional[str] = None
    image: Optional[bytes] = None
    options: list = field(default_factory=list)       # clarification labels
    provenance: list = field(default_factory=list)
    plan_json: Optional[str] = None
    evidence_block: str = ""
    notes: list = field(default_factory=list)
    trace: list = field(default_factory=list)    # human-readable decision log
    prose: bool = False          # answer is a judgement; the UI must show text
    row_count: Optional[int] = None   # TRUE rows, before any display cap


class TableReasoner:
    def __init__(self, ollama, chat_model: str) -> None:
        self.ollama = ollama
        self.chat_model = chat_model
        self.planner = Planner(ollama, chat_model)
        self.model: Optional[SemanticModel] = None
        self.vindex: Optional[ValueIndex] = None
        self.dataframes: dict = {}
        # conversation state (Phase A) and pending clarification (Phase B)
        self.last_plan: Optional[dict] = None
        self.pending_clarify: Optional[dict] = None
        self.cards: dict = {}
        self._plan_cache: dict = {}
        self._index_version: str = "0"
        self._validator = None

    # ------------------------------------------------------------------ #
    def attach(self, dataframes: dict, model: Optional[SemanticModel],
               vindex: Optional[ValueIndex], cards: Optional[dict] = None,
               index_version: Optional[str] = None) -> None:
        self.dataframes = dataframes or {}
        self.model = model
        self.vindex = vindex
        self.cards = cards or {}
        self.last_plan = None
        self.pending_clarify = None
        self._index_version = index_version or str(len(self.dataframes))
        self._plan_cache = self._load_plan_cache()
        from jarvisman.planning.query_plan import PlanValidator
        if model is not None and vindex is not None:
            self._validator = PlanValidator(self.dataframes, model, vindex) \
                if False else PlanValidator(model, self.dataframes, vindex)
            self._validator.attach_context(self.cards, [])
        else:
            self._validator = None
        # share cards with the planner for schema rendering + compile context
        self.planner._cards = self.cards
        self.planner._coverage_anchors = []

    @property
    def ready(self) -> bool:
        return bool(self.dataframes) and self.model is not None \
            and self.vindex is not None

    # ------------------------------------------------------------------ #
    # Clarification state machine (Phase B)                              #
    # ------------------------------------------------------------------ #
    def consume_option(self, message: str) -> Optional[dict]:
        """If a clarification is pending and the message selects one of its
        options (by label or by 1-based number), return
        {"question": original, "bind": {...}} and clear the pending state."""
        p = self.pending_clarify
        if not p:
            return None
        mn = norm_text(message)
        # Punctuation-insensitive comparison. Option labels carry apostrophes
        # and parentheses ("Each {what}'s own (original) amounts"); a smart
        # quote from the UI, or an apostrophe stripped by _clean_columns, is
        # enough to break strict equality. A failed match used to fall through
        # to the planner, which then answered the CHIP TEXT as if it were a
        # data question -- real pandas, wrong question, plausible numbers.
        _loose = lambda s: _LOOSE_RE.sub("", norm_text(s))
        ml = _loose(message)
        chosen = None
        if mn.isdigit() and 1 <= int(mn) <= len(p["options"]):
            chosen = p["options"][int(mn) - 1]
        else:
            for opt in p["options"]:
                if ml == _loose(opt["label"]) \
                        or (opt.get("column") and ml == _loose(opt["column"])):
                    chosen = opt
                    break
        if chosen is None:
            # Keep the clarification ALIVE. Clearing it here destroyed the
            # pending state AND let the message through as a fresh question.
            return {"question": p["question"], "bind": None, "term": "",
                    "unmatched": True}
        self.pending_clarify = None       # one clarification max, ever
        if p.get("kind") == "table_choice":
            # Rewrite the question with the chosen period and re-run:
            # content_route() then resolves it deterministically, so there is
            # exactly one place that decides which sheet answers a question.
            q = p["question"].rstrip("? ")
            return {"question": q + chosen["suffix"] + "?",
                    "bind": None, "term": ""}
        if p.get("kind") == "proposal":
            # yes -> re-run the original question with the model's
            # accept_directive enforced; no/anything else -> stop gracefully
            d = chosen.get("directive") or ""
            if not d:
                return {"question": p["question"], "bind": None,
                        "term": "", "directive": "", "declined": True}
            return {"question": p["question"], "bind": None, "term": "",
                    "directive": d}
        if p.get("kind") == "interpretation":
            # the user chose a READING of the question, not a value or a
            # column: re-run the original question with the chosen reading
            # injected into the planner prompt as a confirmed directive
            constraint = {}
            if chosen.get("forbid_filter_column"):
                constraint["forbid_filter_column"] = \
                    chosen["forbid_filter_column"]
            if chosen.get("forbid_agg_column"):
                constraint["forbid_agg_column"] = \
                    chosen["forbid_agg_column"]
            return {"question": p["question"], "bind": None,
                    "term": p["term"], "directive": chosen.get("directive", ""),
                    "constraint": constraint}
        if p.get("kind") == "value_typo":
            # the user confirmed the real value: substitute it into the
            # ORIGINAL question and re-run normally -- the value index then
            # matches exactly and the usual rebind machinery does the rest
            new_q = re.sub(re.escape(p["term"]), chosen["display"],
                           p["question"], count=1, flags=re.IGNORECASE)
            return {"question": new_q, "bind": None, "term": p["term"]}
        return {"question": p["question"], "bind": chosen, "term": p["term"]}

    @staticmethod
    def _ground_summary(evidence) -> str:
        names = sorted({h.display for h in evidence.value_hits
                        if h.kind in ("value_in_question", "value_token")})[:6]
        asat = f", as-at {evidence.asat[0]}" if evidence.asat else ""
        return (f"grounded values: {', '.join(repr(n) for n in names) or 'none'}"
                + asat) if (names or asat) else "no verified values grounded"

    # ------------------------------------------------------------------ #
    def answer(self, question: str,
               progress: Optional[Callable[[str], None]] = None,
               forced_bind: Optional[dict] = None,
               bind_term: str = "",
               directive: str = "",
               constraint: Optional[dict] = None) -> ReasonerOutcome:
        if not self.ready:
            return ReasonerOutcome(kind="fallback")
        trace: list = []

        # ---- Phase A: deterministic follow-up (zero LLM calls) ----------
        fu_plan = None if directive else \
            followup.rewrite(question, self.last_plan, self.model, self.vindex)
        if fu_plan is not None:
            if progress:
                progress("Resolving follow-up from the previous question ...")
            out = self._compile_and_run(QueryPlan.from_dict(fu_plan), question,
                                        evidence_block="", plan_raw="(follow-up rewrite)",
                                        notes=["follow-up resolved without an LLM call"],
                                        progress=progress)
            if out is not None:
                out.trace = ["follow-up rewrite of the previous question"]
                if followup.wants_chart(question) and out.kind == "ok":
                    chart = self.chart(question, progress)
                    if chart.kind == "ok":
                        chart.trace = ["follow-up rewrite -> chart"]
                        return chart
                return out
            trace.append("follow-up rewrite attempted but did not validate")

        if progress:
            progress("Grounding the question in the data ...")
        evidence = ground(question, self.model, self.vindex)
        ev_block = evidence.to_prompt_block()
        trace.append(self._ground_summary(evidence))

        # ---- auto-interpretation: a deciding word ('EUR equivalent',
        # 'converted to EUR') already chose the 'use the converted measure
        # over ALL rows' reading. Apply it as an ENFORCED directive without
        # asking -- and crucially BEFORE the as-at/clarify gates, so a date
        # in the question cannot suppress it. (Skipped on a forced re-run or
        # when a directive is already in force.)
        if forced_bind is None and not directive and cfg.INTERPRET_CLARIFY:
            try:
                from jarvisman.semantics.ambiguity import find_interpretation_ambiguity
                auto = find_interpretation_ambiguity(
                    question, evidence, self.model, self.vindex)
            except Exception:
                auto = None
            if auto and auto.get("kind") == "auto_interpretation":
                directive = auto["directive"]
                if auto.get("forbid_filter_column"):
                    constraint = {"forbid_filter_column":
                                  auto["forbid_filter_column"]}
                ev_block += ("INTERPRETATION (decided by the wording of the "
                             "question -- follow it EXACTLY): "
                             + directive + "\n")
                trace.append("auto-interpretation from question wording: "
                             + directive[:120])
        anchors = coverage_anchors(evidence, self.model)
        self.planner._coverage_anchors = anchors
        if self._validator is not None:
            self._validator.attach_context(self.cards, anchors)

        # ---- plan cache: a normalized question already answered this index
        if cfg.PLAN_CACHE and forced_bind is None and not directive:
            ckey = (self._index_version, norm_text(question))
            cached = self._plan_cache.get(ckey)
            if cached is not None:
                if progress:
                    progress("Reusing the previous plan for this question ...")
                out = self._compile_and_run(
                    QueryPlan.from_dict(cached), question, ev_block,
                    "(plan cache)", ["answered from the plan cache (0 LLM calls)"],
                    progress)
                if out is not None:
                    out.trace = trace + ["answered from plan cache (0 LLM calls)"]
                    return out

        # ---- Tier 0.5: deterministic plan for simple shapes (0 LLM calls)
        if forced_bind is None and not directive:
            t05 = tier05.try_plan(question, evidence, self.model, self._validator)
            if t05 is not None:
                if progress:
                    progress("Planning directly (no model call needed) ...")
                out = self._compile_and_run(
                    QueryPlan.from_dict(t05), question, ev_block,
                    "(tier 0.5)", ["planned deterministically (0 LLM calls)"],
                    progress, cache_plan=t05)
                if out is not None:
                    out.trace = trace + ["Tier 0.5 deterministic plan (0 LLM calls)"]
                    return out
            trace.append("Tier 0.5 declined (not a simple single-anchor shape)")

        # ---- Phase B: ask ONE clarifying question when truly ambiguous --
        if forced_bind is None and not directive and self.pending_clarify is None:
            amb = find_ambiguity(evidence, self.model, question, self.vindex)
            if amb:
                self.pending_clarify = {"question": question,
                                        "term": amb["term"],
                                        "kind": amb.get("kind", ""),
                                        "options": amb["options"]}
                labels = [o["label"] for o in amb["options"]]
                return ReasonerOutcome(kind="clarify", text=amb["question"],
                                       options=labels, evidence_block=ev_block)
        if forced_bind is not None:
            ev_block += (f"USER CLARIFIED: '{bind_term}' refers to column "
                         f"'{forced_bind['column']}' of table "
                         f"'{forced_bind['table']}'. Use exactly that column.\n")
            evidence_forced = evidence
        if directive:
            ev_block += ("INTERPRETATION (confirmed by the user -- follow it "
                         "EXACTLY, it overrides any other reading): "
                         + directive + "\n")
            trace.append("user chose an interpretation: " + directive[:120])
        # one clarification max per question
        self.pending_clarify = None

        # Nothing in the question says WHICH sibling sheet is meant. Ask --
        # picking one silently is the original bug. content_route() has already
        # had its chance, so anything reaching here is genuinely ambiguous.
        if forced_bind is None and not directive:
            cq, copts = sibling_choice(question, self.model, self.dataframes)
            if cq:
                self.pending_clarify = {"question": question, "term": "",
                                        "kind": "table_choice",
                                        "options": copts}
                return ReasonerOutcome(
                    kind="clarify", text=cq,
                    options=[o["label"] for o in copts],
                    evidence_block=ev_block, trace=trace)

        tables = self._planner_tables(evidence)

        extra = ev_block if (forced_bind or directive) else ""
        if self.last_plan and followup._is_followup_shaped(
                norm_text(question), len(question.split())):
            import json as _json
            extra = (ev_block + "Previous plan (the user may be refining it):\n"
                     + _json.dumps(self.last_plan)[:600] + "\n")
        trace.append("called the planner (1 LLM call)")
        plres = self.planner.make_plan(question, self.model, tables,
                                       self.vindex, evidence, progress,
                                       extra_evidence=extra)
        if plres.status == "clarify":
            return ReasonerOutcome(kind="clarify", text=plres.clarification or "",
                                   evidence_block=ev_block, notes=plres.notes,
                                   trace=trace + ["planner asked for clarification"])
        # ---- enforce the user's chosen interpretation IN CODE: a planner
        # that ignored the directive and still filtered the forbidden column
        # gets that filter removed and the plan recompiled. The user's click
        # is a hard constraint, not a suggestion.
        forb_agg = (constraint or {}).get("forbid_agg_column")
        if forb_agg and plres.status == "ok" and plres.plan is not None \
                and any(a.column == forb_agg for a in plres.plan.aggregations):
            kept = [a for a in plres.plan.aggregations
                    if a.column != forb_agg]
            trace.append(f"planner aggregated '{forb_agg}' despite the "
                         "user's choice; aggregation removed")
            if kept:
                plres.plan.aggregations = kept
                b3, i3, n3 = compile_plan(plres.plan, self.model,
                                          self.dataframes, self.vindex,
                                          cards=self.cards,
                                          coverage_anchors=None)
                if b3 is not None:
                    plres.bundle = b3
                    plres.notes = list(plres.notes) + n3
                else:
                    trace.append("plan unusable after the correction -> codegen")
                    return ReasonerOutcome(kind="fallback",
                                           evidence_block=ev_block,
                                           notes=plres.notes, trace=trace)
            else:
                # the ONLY aggregation was the forbidden one: give the
                # planner ONE corrective re-plan with the precise violation
                # before surrendering to codegen
                trace.append("re-planning once with the violation spelled out")
                extra2 = (ev_block +
                          f"REJECTED: your previous plan aggregated column "
                          f"'{forb_agg}', which the user explicitly excluded. "
                          f"Do NOT use '{forb_agg}' anywhere. Group as asked "
                          f"and aggregate the measure that holds the ORIGINAL "
                          f"(unconverted) amounts.\n")
                plres2 = self.planner.make_plan(question, self.model,
                                                tables, self.vindex,
                                                evidence, progress,
                                                extra_evidence=extra2)
                ok2 = (plres2.status == "ok" and plres2.bundle is not None
                       and plres2.plan is not None
                       and not any(a.column == forb_agg
                                   for a in plres2.plan.aggregations))
                if ok2:
                    plres = plres2
                    trace.append("corrected plan accepted")
                else:
                    trace.append("re-plan still violated or failed -> codegen "
                                 "(directive carried in the evidence)")
                    return ReasonerOutcome(kind="fallback",
                                           evidence_block=ev_block,
                                           notes=plres.notes, trace=trace)

        forb = (constraint or {}).get("forbid_filter_column")
        if forb and plres.status == "ok" and plres.plan is not None \
                and any(f.column == forb for f in plres.plan.filters):
            plres.plan.filters = [f for f in plres.plan.filters
                                  if f.column != forb]
            trace.append(f"planner filtered '{forb}' despite the user's "
                         "choice; filter removed and plan recompiled")
            b2, i2, n2 = compile_plan(plres.plan, self.model,
                                      self.dataframes, self.vindex,
                                      cards=self.cards,
                                      coverage_anchors=None)
            if b2 is not None:
                plres.bundle = b2
                plres.notes = list(plres.notes) + n2
            else:
                trace.append("plan unusable after the correction -> codegen")
                return ReasonerOutcome(kind="fallback",
                                       evidence_block=ev_block,
                                       notes=plres.notes, trace=trace)

        if plres.status != "ok" or plres.bundle is None:
            issue_lines = [getattr(i, "message", str(i)) for i in plres.issues]
            # This is the decision that demotes a question to codegen. Make it
            # legible: WHAT the planner proposed and WHY it was rejected, so a
            # wrong answer is traceable instead of a mystery.
            trace.append("PLAN REJECTED -> falling through to codegen")
            for ln in issue_lines:
                trace.append(f"  reason: {ln}")
            if plres.raw:
                trace.append(f"  planner proposed: {str(plres.raw)[:300]}")
            return ReasonerOutcome(kind="fallback", evidence_block=ev_block,
                                   notes=plres.notes + issue_lines, trace=trace)

        out = self._execute(plres.bundle, plres.plan, question, ev_block,
                            plres.raw, plres.notes, progress)
        if out is not None and out.kind == "ok" and cfg.PLAN_CACHE \
                and forced_bind is None and self.last_plan:
            self._plan_cache[(self._index_version, norm_text(question))] = \
                dict(self.last_plan)
            self._save_plan_cache()
        if out is not None:
            out.trace = trace + ["plan compiled and ran"]
            return out
        trace.append("compiled plan produced no usable outcome -> codegen")
        return ReasonerOutcome(kind="fallback", evidence_block=ev_block,
                               notes=plres.notes, trace=trace)

    # ------------------------------------------------------------------ #
    # Charts (Phase C): same grounding/planning, deterministic rendering #
    # ------------------------------------------------------------------ #
    def chart(self, question: str,
              progress: Optional[Callable[[str], None]] = None) -> ReasonerOutcome:
        if not self.ready or not cfg.CHART_PLAN_ENABLED:
            return ReasonerOutcome(kind="fallback")
        qn = norm_text(question)
        # explicit type words win; otherwise 'auto' and we decide from the
        # data shape after the group-by result is known (few categories -> bar
        # or pie, many -> horizontal bar, temporal -> line).
        if "pie" in qn or "πιτα" in qn:
            ctype = "pie"
        elif any(w in qn for w in ("line", "trend", "γραμμη", "εξελιξη",
                                   "πορεια", "over time", "by year",
                                   "by month", "per year", "per month")):
            ctype = "line"
        elif "horizontal" in qn or "οριζοντ" in qn:
            ctype = "barh"
        else:
            ctype = "auto"
        data_q = _CHART_WORD_RE.sub(" ", question).strip() or question

        if progress:
            progress("Grounding the chart request ...")
        evidence = ground(data_q, self.model, self.vindex)
        ev_block = evidence.to_prompt_block()
        tables = self._planner_tables(evidence)

        plres = self.planner.make_plan(data_q, self.model, tables,
                                       self.vindex, evidence, progress)
        plan = plres.plan
        if plres.status != "ok" or plres.bundle is None or plan is None \
                or not plan.group_by or not plan.aggregations:
            return ReasonerOutcome(kind="fallback", evidence_block=ev_block,
                                   notes=(plres.notes if plres else []))

        g0 = plan.group_by[0]
        alias0 = self._first_alias(plan)
        code = self._chart_code(plres.bundle.code, g0, alias0, ctype, question)
        ok, why = validate_code(code)
        if not ok:
            return ReasonerOutcome(kind="fallback", evidence_block=ev_block,
                                   notes=[f"chart code rejected: {why}"])
        if progress:
            progress("Rendering the chart ...")
        exec_tables = {n: tables[n] for n in plres.bundle.tables_used if n in tables}
        result = run_sandboxed(code, exec_tables, timeout=cfg.SANDBOX_TIMEOUT)
        if not result.get("ok") or not result.get("image"):
            return ReasonerOutcome(kind="fallback", evidence_block=ev_block,
                                   notes=[f"chart failed: {result.get('error')}"])
        provenance = self._provenance(result.get("stdout") or "",
                                      plres.bundle.explain)
        caption = f"Chart of {alias0} by {g0}."
        self.last_plan = self._plan_state(plan)
        return ReasonerOutcome(kind="ok", text=caption, image=result["image"],
                               code=code, provenance=provenance,
                               plan_json=plres.raw, evidence_block=ev_block,
                               notes=plres.notes)

    def _planner_tables(self, evidence: Evidence) -> dict:
        """Tables exposed to the planner.

        When the question names a snapshot date, period or entity value, expose
        ONLY the sheet(s) that hold it: the capability ranker scores
        same-schema sheets identically, so without this the planner breaks the
        tie on prompt position and always picks the same file.

        When it names none of those, fall back to the ORIGINAL behaviour: for a
        small workbook expose every table in stable insertion order, so the
        schema block is byte-identical across questions and llama.cpp
        prompt-prefix caching can skip re-evaluating it. Routing costs that
        cache (the block now varies per question) -- but a routed block is one
        table instead of six, and a wrong answer is worse than a slow one."""
        all_names = list(self.dataframes)
        if len(all_names) <= 1:
            return dict(self.dataframes)

        from jarvisman.semantics.semantic_model import content_route
        cap = cfg.MAX_ANALYSIS_TABLES_REASONER

        routed = content_route(evidence.question, self.model, max_n=cap)
        value_tables = [n for n in dict.fromkeys(
            h.table for h in evidence.value_hits) if n in self.dataframes]
        lead = list(dict.fromkeys(
            n for n in (routed + value_tables) if n in self.dataframes))
        if lead:
            chosen = self._ensure_column_coverage(evidence.question, lead[:cap], cap)
            return {n: self.dataframes[n] for n in chosen}

        # No signal -> keep the cacheable stable-order block for small workbooks
        if len(all_names) <= 6:
            return dict(self.dataframes)
        ranked = evidence.ranked_tables(cap)
        chosen = [n for n in ranked if n in self.dataframes] or all_names[:cap]
        return {n: self.dataframes[n] for n in chosen}



    def _ensure_column_coverage(self, question: str, chosen: list,
                                cap: int) -> list:
        """Value/date routing can pick sheets that merely CONTAIN a
        mentioned value while lacking the COLUMN the question needs
        ('Italy' appears in one sheet, but only another has Address -- the
        address question then dead-ends with 'column does not exist').
        If the question names a real column that none of the chosen
        tables carries, append the best table that has it."""
        ql = (question or "").lower()
        col_tables: dict = {}
        for name, df in self.dataframes.items():
            try:
                for c in df.columns:
                    cs = str(c).strip().lower()
                    if len(cs) >= 4 and cs in ql:
                        col_tables.setdefault(cs, []).append(name)
            except Exception:
                continue
        out = list(chosen)
        for cs, tables in col_tables.items():
            if any(t in out for t in tables):
                continue
            try:
                extra = max(tables, key=lambda t: self.dataframes[t].shape[0])
            except Exception:
                extra = tables[0]
            if extra not in out:
                out.append(extra)
        return out[: cap + 2]
        
    @staticmethod
    def _first_alias(plan: QueryPlan) -> str:
        from jarvisman.planning.query_plan import PlanValidator
        return PlanValidator._alias(plan.aggregations[0])

    @staticmethod
    def _chart_code(data_code: str, g0: str, alias0: str, ctype: str,
                    question: str) -> str:
        title = re.sub(r"\s+", " ", question).strip()[:80]
        # All decisions that depend on the DATA shape are made at runtime
        # inside the sandbox (category count, ordering), so a single emitted
        # program adapts: 'auto' becomes bar / barh / pie based on cardinality.
        L = [data_code, ""]
        L.append(f"_ct = {ctype!r}")
        L.append(f"result = result.dropna(subset=[{g0!r}])")
        # sort: temporal/line keep natural key order; else by value desc
        L.append(f"if _ct == 'line':")
        L.append(f"    result = result.sort_values({g0!r})")
        L.append(f"else:")
        L.append(f"    result = result.sort_values({alias0!r}, ascending=False)")
        L.append(f"_labels = result[{g0!r}].astype(str).tolist()")
        L.append(f"_values = [float(v) for v in result[{alias0!r}].tolist()]")
        L.append("_n = len(_labels)")
        # auto type from cardinality
        L.append("if _ct == 'auto':")
        L.append("    _ct = 'bar' if _n <= 8 else 'barh'")
        # cap categories: keep top N, fold the rest into 'Other' (bar/barh),
        # or top slices for pie; line keeps all points
        L.append("_CAP = 15")
        L.append("if _ct in ('bar', 'barh', 'pie') and _n > _CAP:")
        L.append("    _keep = _CAP - 1")
        L.append("    _other = sum(_values[_keep:])")
        L.append("    _labels = _labels[:_keep] + ['Other']")
        L.append("    _values = _values[:_keep] + [_other]")
        L.append("    _n = len(_labels)")
        # figure size scales with category count for horizontal bars
        L.append("if _ct == 'barh':")
        L.append("    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.4 * _n + 1)))")
        L.append("else:")
        L.append("    fig, ax = plt.subplots(figsize=(8, 4.5))")
        L.append("_color = '#4c9aff'")
        # ---- render per type ----
        L.append("if _ct == 'pie':")
        L.append("    ax.pie(_values, labels=_labels, autopct='%1.1f%%',")
        L.append("           startangle=90, counterclock=False,")
        L.append("           textprops={'fontsize': 9})")
        L.append("    ax.axis('equal')")
        L.append("elif _ct == 'line':")
        L.append("    ax.plot(_labels, _values, marker='o', color=_color, lw=2)")
        L.append("    ax.grid(True, axis='y', alpha=0.25)")
        L.append("    ax.tick_params(axis='x', rotation=45)")
        L.append("elif _ct == 'barh':")
        L.append("    _y = range(_n)")
        L.append("    ax.barh(list(_y), _values, color=_color)")
        L.append("    ax.set_yticks(list(_y)); ax.set_yticklabels(_labels)")
        L.append("    ax.invert_yaxis()")
        L.append("    _vmax = max(_values) if _values else 0")
        L.append("    for _i, _v in enumerate(_values):")
        L.append("        ax.text(_v + _vmax*0.01, _i, _fmt_val(_v),")
        L.append("                va='center', fontsize=8)")
        L.append("    ax.grid(True, axis='x', alpha=0.25)")
        L.append("else:")  # vertical bar
        L.append("    _x = range(_n)")
        L.append("    ax.bar(list(_x), _values, color=_color)")
        L.append("    ax.set_xticks(list(_x))")
        L.append("    ax.set_xticklabels(_labels, rotation=45, ha='right')")
        L.append("    _vmax = max(_values) if _values else 0")
        L.append("    for _i, _v in enumerate(_values):")
        L.append("        ax.text(_i, _v + _vmax*0.01, _fmt_val(_v),")
        L.append("                ha='center', va='bottom', fontsize=8)")
        L.append("    ax.grid(True, axis='y', alpha=0.25)")
        # axis labels + thousands formatting on the value axis
        L.append(f"ax.set_title({title!r}, fontsize=12, fontweight='bold')")
        L.append("if _ct not in ('pie',):")
        L.append("    _axis = ax.xaxis if _ct == 'barh' else ax.yaxis")
        L.append("    _axis.set_major_formatter(plt.FuncFormatter(lambda v, p: _fmt_val(v)))")
        L.append("if _ct == 'bar':")
        L.append(f"    ax.set_xlabel({g0!r}); ax.set_ylabel({alias0!r})")
        L.append("elif _ct == 'barh':")
        L.append(f"    ax.set_ylabel({g0!r}); ax.set_xlabel({alias0!r})")
        L.append("elif _ct == 'line':")
        L.append(f"    ax.set_xlabel({g0!r}); ax.set_ylabel({alias0!r})")
        L.append("for _s in ('top', 'right'):")
        L.append("    ax.spines[_s].set_visible(False)")
        L.append("fig.tight_layout()")
        # helper for compact value labels (1.2M / 3.4k / 56)
        helper = (
            "def _fmt_val(v):\n"
            "    try: v = float(v)\n"
            "    except Exception: return str(v)\n"
            "    a = abs(v)\n"
            "    if a >= 1e9: return f'{v/1e9:.1f}B'\n"
            "    if a >= 1e6: return f'{v/1e6:.1f}M'\n"
            "    if a >= 1e3: return f'{v/1e3:.1f}k'\n"
            "    return f'{v:.0f}'\n")
        return helper + "\n" + "\n".join(L)

    # ------------------------------------------------------------------ #
    # Shared execution path                                              #
    # ------------------------------------------------------------------ #
    def _compile_and_run(self, plan: QueryPlan, question: str,
                         evidence_block: str, plan_raw: str, notes: list,
                         progress, cache_plan: Optional[dict] = None
                         ) -> Optional[ReasonerOutcome]:
        # deterministic paths (follow-up, tier 0.5, cache) are already built
        # from verified values -> do not re-run the dropped-condition guard
        bundle, issues, vnotes = compile_plan(
            plan, self.model, self.dataframes, self.vindex,
            cards=self.cards, coverage_anchors=None)
        if bundle is None:
            return None
        out = self._execute(bundle, plan, question, evidence_block,
                            plan_raw, notes + vnotes, progress)
        if out is not None and out.kind == "ok" and cache_plan is not None \
                and cfg.PLAN_CACHE:
            self._plan_cache[(self._index_version, norm_text(question))] = cache_plan
            self._save_plan_cache()
        return out

    def _execute(self, bundle, plan, question: str, ev_block: str,
                 plan_raw: str, notes: list, progress) -> Optional[ReasonerOutcome]:
        if progress:
            progress("Running the query ...")
        exec_tables = {n: self.dataframes[n] for n in bundle.tables_used
                       if n in self.dataframes}
        result = run_query(bundle.code, exec_tables, timeout=cfg.SANDBOX_TIMEOUT)
        if not result.get("ok"):
            return ReasonerOutcome(
                kind="fallback", evidence_block=ev_block,
                notes=notes + [f"compiled plan failed: {result.get('error')}"])

        provenance = self._provenance(result.get("stdout") or "", bundle.explain)
        zero = [p for p in provenance if p.endswith(": 0 rows") and "filter" in p]
        body = (result.get("text") or "").strip()

        if zero:
            note = ("No rows match all the conditions together "
                    f"({'; '.join(z[5:] for z in zero)}).")
            self.last_plan = self._plan_state(plan)
            return ReasonerOutcome(kind="ok", text=note, code=bundle.code,
                                   provenance=provenance, plan_json=plan_raw,
                                   evidence_block=ev_block, notes=notes)

        text = body or "(empty result)"
        # Speed: a multi-row table speaks for itself -- phrasing it with an LLM
        # call adds latency and no clarity. Synthesize ONLY for compact scalar/
        # few-row results, where a one-line answer genuinely helps. Controlled
        # by SYNTHESIZE_ANSWER (off entirely) and SYNTHESIZE_MAX_ROWS.
        rows = result.get("row_count")
        if rows is None:
            rows = (body.count("\n") if body else 0)
        # Gate on INTENT as well as size. A figure question is answered by its
        # table, so phrasing it is latency for nothing. A comparison question is
        # NOT answered by a table at all -- it needs words -- so allow the
        # phrasing call on a larger result.
        wants_prose = self._wants_prose(question)
        cap = (getattr(cfg, "SYNTHESIZE_PROSE_MAX_ROWS", 40) if wants_prose
               else getattr(cfg, "SYNTHESIZE_MAX_ROWS", 3))
        prose_out = False
        if cfg.SYNTHESIZE_ANSWER and body and rows <= cap:
            if progress:
                progress("Writing the answer ...")
            synth = self._synthesize(question, body, provenance,
                                     prose=wants_prose)
            if synth:
                text, prose_out = synth, wants_prose
            else:
                text = body
        elif body:
            how = "; ".join(e for e in bundle.explain[:4])
            if how:
                text = f"{body}\n\n(How: {how})"

        self.last_plan = self._plan_state(plan)
        return ReasonerOutcome(kind="ok", text=text,
                               table_html=result.get("table_html"),
                               code=bundle.code, provenance=provenance,
                               plan_json=plan_raw, evidence_block=ev_block,
                               notes=notes, prose=prose_out,
                               row_count=result.get("row_count"))

    @staticmethod
    def _plan_state(plan: QueryPlan) -> dict:
        """Bound plan -> plain dict for the follow-up rewriter (Phase A)."""
        try:
            d = dataclasses.asdict(plan)
            for f in d.get("filters") or []:
                f.pop("_resolved", None)
            return d
        except Exception:
            return {}

    # ------------------------------------------------------------------ #
    @staticmethod
    def _provenance(stdout: str, explain: list) -> list:
        steps = [f"PROV {m.group(1)}: {m.group(2)} rows"
                 for m in _PROV_RE.finditer(stdout)]
        return list(explain) + steps

    def _load_plan_cache(self) -> dict:
        """Read persisted plans for the current index version (0 model calls)."""
        if not getattr(cfg, "PLAN_CACHE_PERSIST", True):
            return {}
        path = getattr(cfg, "PLAN_CACHE_PATH", "")
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh) or {}
        except Exception:
            return {}
        iv = self._index_version
        out = {}
        for q, plan in (data.get(iv, {}) or {}).items():
            if isinstance(plan, dict):
                out[(iv, q)] = plan
        return out

    def _save_plan_cache(self) -> None:
        """Write the current index version's plans atomically; keep the file
        small by retaining only the few most-recent index versions."""
        if not getattr(cfg, "PLAN_CACHE_PERSIST", True):
            return
        path = getattr(cfg, "PLAN_CACHE_PATH", "")
        if not path:
            return
        iv = self._index_version
        bucket = {q: plan for (v, q), plan in self._plan_cache.items() if v == iv}
        try:
            existing = {}
            try:
                with open(path, encoding="utf-8") as fh:
                    existing = json.load(fh) or {}
            except Exception:
                existing = {}
            existing[iv] = bucket
            if len(existing) > 3:
                for k in list(existing.keys())[:-3]:
                    existing.pop(k, None)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(existing, fh)
            os.replace(tmp, path)
        except Exception as exc:
            cfg.dbg("reasoner._save_plan_cache", exc)

    @staticmethod
    def _wants_prose(question: str) -> bool:
        """True when the user asked for a judgement/comparison rather than a
        figure. Deliberately narrow: it must never fire on 'total funds per
        currency' or 'sort interest rates' -- those already work and must not
        pay for an extra LLM call."""
        return bool(_PROSE_INTENT_RE.search(question or ""))

    def _synthesize(self, question: str, result_text: str,
                    provenance: list, prose: bool = False) -> Optional[str]:
        """One short call to phrase the computed result -- with an echo check:
        every number in the phrased answer must exist in the computed result,
        otherwise the raw result is returned instead. The model is never
        allowed to 'improve' a figure."""
        capped = result_text[:1500]
        prov = "\n".join(provenance[-6:])
        if prose:
            # _numbers_echo_ok rejects ANY number not present in the result, so
            # a derived figure ("EUR 200 more", "23% higher") would silently
            # sink the whole answer back to a bare table. Demand comparative
            # WORDS instead -- which is what the user asked for anyway.
            head = (
                "Answer the user's question in 2-5 sentences, in the same "
                "language as the question, using ONLY the computed result "
                "below. They want a judgement, not a data dump: say which is "
                "largest and smallest, what stands out, and what it means for "
                "them. Quote figures EXACTLY as they appear. NEVER calculate a "
                "new number -- no differences, percentages, ratios or totals of "
                "your own. Use words instead: 'the largest by some margin', "
                "'roughly double', 'far behind'. Do not mention code or "
                "DataFrames.")
        else:
            head = (
                "Write a direct 1-3 sentence answer to the user's question, in "
                "the same language as the question, using ONLY the computed "
                "result below. Quote the numbers exactly as they appear; do not "
                "round, convert, or add any figure that is not in the result. "
                "Do not mention code or DataFrames.")
        prompt = (f"{head}\n\nQuestion: {question}\n\n"
                  f"Computed result:\n{capped}\n\n"
                  f"How it was computed:\n{prov}\n\nAnswer:")
        try:
            out = self.ollama.chat(
                cfg.model_for("synthesize", self.chat_model),
                [{"role": "user", "content": prompt}],
                options={"temperature": 0.0,
                         "num_predict": (
                             getattr(cfg, "SYNTH_PROSE_NUM_PREDICT", 420) if prose
                             else getattr(cfg, "SYNTH_NUM_PREDICT", 220))},
            ).strip()
        except Exception:
            return None
        if not out:
            return None
        if self._numbers_echo_ok(out, result_text):
            return out
        return None

    @staticmethod
    def _numbers_echo_ok(answer: str, source: str) -> bool:
        from jarvisman.runtime import numfmt
        src = numfmt.canon(source)
        src_nums = set(_NUM_RE.findall(src))
        for raw in _NUM_RE.findall(numfmt.canon(answer)):
            if raw in src_nums:
                continue
            try:
                v = float(raw)
            except ValueError:
                continue
            ok = False
            for s in src_nums:
                try:
                    if abs(float(s) - v) <= max(abs(float(s)) * 1e-6, 1e-9) or \
                            round(float(s), 2) == round(v, 2):
                        ok = True
                        break
                except ValueError:
                    continue
            if not ok:
                return False
        return True