from __future__ import annotations

import difflib
import json
import re
from typing import Callable, Optional

from jarvisman import config as cfg
from jarvisman.runtime import house_rules
from jarvisman.semantics.grounding import ground
from jarvisman.llm.llm_json import extract_json as _extract_json_shared
from jarvisman.llm.ollama_client import OllamaClient
from jarvisman.retrieval.rag import RAGPipeline
from jarvisman.planning.reasoner import TableReasoner
from jarvisman.planning import code_resolver
from jarvisman.planning import continuity
from jarvisman.retrieval.retrieval import tokenize
from jarvisman.runtime.sandbox import run_query, run_sandboxed
from jarvisman.semantics.semantic_model import build_semantic_model, load_semantic_model
from jarvisman.semantics.value_index import ValueIndex
from jarvisman.runtime import timing
timing.reset()

def _yes_label() -> str:
    return "Ναι, κάνε το" if getattr(cfg, "UI_LANG", "en") == "el" else "Yes, do that"


def _no_label() -> str:
    return "Όχι" if getattr(cfg, "UI_LANG", "en") == "el" else "No"


_VALID_TOOLS = {"answer_docs", "analyze", "plot", "clarify", "chat"}
_PLOT_HINTS = ("plot", "chart", "graph", "visuali", "histogram", "bar ", "scatter",
               "trend", "distribution", "pie", "line chart", "boxplot")
_ANALYZE_HINTS = ("total", "sum", "average", "mean", "median", "count", "how many",
                  "how much", "maximum", "minimum", "value of", "amount", "equivalent",
                  "as at", "as of", "per ", "breakdown", "group by", "subtotal",
                  "balance", "aggregate")
_LOOKUP_HINTS = ("list", "show me", "show all", "which ", "find ", "filter",
                 "rows where", "row where", "all rows", "how many rows", "records",
                 "record for", "value of", "values of", "for company", "for each",
                 "details of", "details for", "information about", "info on",
                 "look up", "lookup", "entries", "entry for", "in the table",
                 "in the spreadsheet", "from the table", "from the spreadsheet")
_DOC_SIGNALS = ("document", "pdf", "report", "contract", "agreement", "page ",
                "section", "clause", "paragraph", "says", "say ", "stated",
                "describe", "explain", "summar", "what does the", "according to",
                "mentioned in", "the text")
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_KEYERROR_RE = re.compile(r"KeyError: ['\"](.+?)['\"]")
# a quoted string immediately followed by ']' -> almost always a column reference
# (df['X'], df.loc[mask, 'X'], dfs['file:sheet']); value strings sit inside (...) instead
_COL_REF_RE = re.compile(r"""['"]([^'"]+)['"]\s*\]""")


def _norm_col(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _match_column(token: str, norm_map: dict) -> Optional[str]:
    """Find the real column that 'token' most likely meant, or None. Conservative:
    exact-normalised, then substantial substring, then close fuzzy match."""
    nb = _norm_col(token)
    if not nb:
        return None
    if nb in norm_map:                                      # exact (case/space-insensitive)
        return norm_map[nb]
    for nc, c in norm_map.items():                          # substantial containment
        if (nb in nc or nc in nb) and min(len(nb), len(nc)) / max(len(nb), len(nc)) >= 0.6:
            return c
    cand = difflib.get_close_matches(nb, list(norm_map.keys()), n=1, cutoff=0.84)
    return norm_map[cand[0]] if cand else None


_READ_CALL_RE = re.compile(r"(?:pd|pandas)\s*\.\s*read_[a-z_]+\s*\(")
_ASSIGN_RE = re.compile(r"^(\s*)([A-Za-z_]\w*)\s*=")


def _sanitize_code(code: str) -> str:
    """Make generated code safe to run against the provided tables: drop import
    lines (pd/np/df/dfs are already supplied) and neutralise any attempt to
    re-read a file from disk. A line like `df = pd.read_excel('x (2).xlsx')`
    becomes `df = df`, so the real, pre-loaded table is used instead. Done
    line-by-line so it is robust to parentheses inside the file name."""
    kept = []
    for ln in code.splitlines():
        s = ln.strip()
        if s.startswith("import ") or s.startswith("from "):
            continue
        if _READ_CALL_RE.search(ln):  # a file re-read: pd.read_excel/read_csv/...
            m = _ASSIGN_RE.match(ln)
            if m:
                kept.append(f"{m.group(1)}{m.group(2)} = df")  # X = pd.read_excel(...) -> X = df
            # a bare read call with no assignment is simply dropped
            continue
        kept.append(ln)
    out = "\n".join(kept)
    return _salvage_trailing_prose(out)


def _salvage_trailing_prose(code: str) -> str:
    """Models sometimes append an explanation after the code (e.g. 'This returns
    the total...'), which lands inside the block and makes it unparseable partway
    down. If the code doesn't parse, drop trailing lines one at a time until it
    does -- but never discard a line that assigns `result`, so the answer is
    preserved. If trimming can't produce valid code, return it unchanged and let
    the normal syntax-error path handle it."""
    try:
        ast.parse(code)
        return code
    except SyntaxError:
        pass
    lines = code.splitlines()
    for cut in range(len(lines) - 1, 0, -1):
        tail = "\n".join(lines[cut:])
        if re.search(r"(?m)^\s*result\s*=", tail):
            break                      # would throw away the answer -> stop
        head = "\n".join(lines[:cut])
        try:
            ast.parse(head)
            return head
        except SyntaxError:
            continue
    return code


def _norm_map(tables: dict) -> dict:
    nm: dict[str, str] = {}
    for df in tables.values():
        for c in df.columns:
            nm.setdefault(_norm_col(c), str(c))
    return nm


def _validate_and_fix_columns(code: str, tables: dict) -> tuple[str, list]:
    """BEFORE running, scan the generated code for column references that don't
    exist and remap each to the closest real column. Returns (fixed_code,
    missing) where 'missing' are referenced columns with no confident match.
    Works for any workbook -- no knowledge of the data required."""
    real = {str(c) for df in tables.values() for c in df.columns}
    keys = set(tables.keys())
    nm = _norm_map(tables)
    new_code = code
    missing: list[str] = []
    seen: set[str] = set()
    for tok in _COL_REF_RE.findall(code):
        if tok in seen or tok in real or tok in keys:
            continue
        seen.add(tok)
        best = _match_column(tok, nm)
        if best and best != tok:
            new_code = new_code.replace(f"'{tok}'", f"'{best}'").replace(f'"{tok}"', f'"{best}"')
        elif best is None:
            missing.append(tok)
    return new_code, missing


def _repair_columns(code: str, error: str, tables: dict) -> Optional[str]:
    """Reactive fix: on a KeyError, remap just the offending column. No LLM."""
    m = _KEYERROR_RE.search(error or "")
    if not m:
        return None
    bad = m.group(1)
    best = _match_column(bad, _norm_map(tables))
    if best is None:
        return None
    new = code.replace(f"'{bad}'", f"'{best}'").replace(f'"{bad}"', f'"{best}"')
    return new if new != code else None


def _extract_json(raw: str) -> Optional[dict]:
    # Centralised in llm_json: strips <think> blocks (reasoning models such as
    # Qwen3 emit them) and parses the first balanced JSON object.
    return _extract_json_shared(raw)


def _extract_code(raw: str) -> str:
    m = _CODE_BLOCK_RE.search(raw)
    if m:
        return m.group(1).strip()
    return raw.strip() if "plt" in raw else ""


def _extract_code_any(raw: str) -> str:
    """Like ``_extract_code`` but for analysis code (no matplotlib expected)."""
    m = _CODE_BLOCK_RE.search(raw)
    if m:
        return m.group(1).strip()
    stripped = raw.strip()
    if "result" in stripped or "df" in stripped or "dfs" in stripped:
        return stripped
    return ""


class Agent:
    def __init__(self, ollama: OllamaClient, rag: RAGPipeline, chat_model: str) -> None:
        self.ollama = ollama
        self.rag = rag
        self.chat_model = chat_model
        self.semantic_model = None
        self.value_index = None
        self.reasoner = TableReasoner(ollama, chat_model)
        self._dataframes: dict = {}
        # conversation memory: a bounded list of past turns so follow-ups and
        # references ("that company", "compare to the previous") work and the
        # chat path has continuity like a normal chatbot.
        self.history: list = []          # [{"q":..., "answer":..., "tool":...}]
        self.max_history_turns = 12
        self._last_company: str = ""      # last company asked about (continuity)
        self._last_analysis_q: str = ""   # last data question (for follow-ups)
        self._last_answer_text: str = ""  # last answer text (for 're-explain')

    # ------------------------------------------------------------------ #
    # Tables + semantic layer                                            #
    # ------------------------------------------------------------------ #
    # ``dataframes`` stays a plain attribute from the outside (the GUI and
    # the eval harness assign it directly); the setter (re)builds the
    # semantic layer so the reasoner is always in sync with the tables.
    @property
    def dataframes(self) -> dict:
        return self._dataframes

    @dataframes.setter
    def dataframes(self, dfs: dict) -> None:
        self._dataframes = dfs or {}
        self._rebuild_semantics()

    def _rebuild_semantics(self) -> None:
        dfs = self._dataframes
        if not dfs:
            self.semantic_model = None
            self.value_index = None
            self.reasoner.attach({}, None, None)
            return
        # Prefer the model built at index time (it carries the LLM meanings);
        # fall back to the persisted copy, then to a fresh statistical build.
        sm = getattr(self.rag, "semantic_model", None)
        if sm is None or set(sm.tables.keys()) != set(dfs.keys()):
            sm = load_semantic_model(cfg.INDEX_DIR)
        if sm is None or set(sm.tables.keys()) != set(dfs.keys()):
            sm = build_semantic_model(
                dfs, meanings=getattr(self.rag, "table_profile", {}) or {}
            )
        self.semantic_model = sm
        self.value_index = ValueIndex.build(dfs)
        # table cards: prefer freshly built (rag), else persisted; validated
        # against the live data inside active_cards(); absence is harmless.
        cards = {}
        try:
            from jarvisman.semantics import table_cards as tc
            store = getattr(self.rag, "table_cards", None) or tc.load_cards(cfg.INDEX_DIR)
            cards = tc.active_cards(store, dfs, sm)
        except Exception:
            cards = {}
        iv = str(getattr(self.rag, "index_version", len(dfs)))
        self.reasoner.attach(dfs, sm, self.value_index, cards=cards,
                             index_version=iv)

    # ------------------------------------------------------------------ #
    # Schemas                                                            #
    # ------------------------------------------------------------------ #
    def _schema_text(self) -> str:
        if not self.dataframes:
            return "none"
        parts = []
        for name, df in self.dataframes.items():
            cols = ", ".join(str(c) for c in df.columns)
            parts.append(f"{name} (rows={len(df)}, columns=[{cols}])")
        return "; ".join(parts)

    def _schema_detail(self, tables: Optional[dict] = None) -> str:
        """Rich schema for code generation: dtype, numeric ranges, and the
        distinct values of low-cardinality columns (so the model can match
        names/currencies exactly). Sent to the LOCAL model only."""
        tables = self.dataframes if tables is None else tables
        if not tables:
            return "none"
        parts = []
        for name, df in tables.items():
            lines = []
            for c in df.columns:
                series = df[c]
                info = f"{c} ({series.dtype})"
                try:
                    kind = series.dtype.kind
                    if kind in "iuf":
                        info += f" range=[{series.min()}, {series.max()}]"
                    else:
                        nunique = int(series.nunique(dropna=True))
                        uniq = series.dropna().unique()
                        if nunique <= cfg.CATEGORICAL_MAX_UNIQUE:
                            vals = ", ".join(str(v) for v in uniq[: cfg.CATEGORICAL_MAX_UNIQUE])
                            info += f" values=[{vals}]"
                        else:
                            ex = ", ".join(str(v) for v in uniq[: cfg.QUERY_SAMPLE_ROWS])
                            info += f" e.g. {ex}"
                except Exception:
                    pass
                lines.append("    - " + info)
            parts.append(f"DataFrame '{name}' (rows={len(df)}):\n" + "\n".join(lines))
        return "\n\n".join(parts)

    def _exact_columns_block(self, tables: Optional[dict] = None) -> str:
        tables = self.dataframes if tables is None else tables
        if not tables:
            return ""
        lines = ["Exact column names you may use (copy verbatim):"]
        for name, df in tables.items():
            cols = ", ".join(repr(str(c)) for c in df.columns)
            lines.append(f"  {name}: [{cols}]")
        return "\n".join(lines) + "\n"

    def _table_meaning_block(self, tables: Optional[dict] = None) -> str:
        """What each selected table and column MEANS, as understood by the model
        when the data was indexed (see column_types.profile_and_apply). Lets the
        model map a question to the right column by meaning rather than by a word
        that merely resembles a column name. Returns '' if no profile is
        available, so the prompt simply falls back to the raw schema."""
        tables = self.dataframes if tables is None else tables
        profile = getattr(self.rag, "table_profile", {}) or {}
        if not tables or not profile:
            return ""
        parts = []
        for name in tables:
            entry = profile.get(name)
            if not entry:
                continue
            summary = (entry.get("summary") or "").strip()
            cols = entry.get("columns") or {}
            head = f"Table '{name}': {summary}" if summary else f"Table '{name}':"
            lines = [head]
            for c in tables[name].columns:
                info = cols.get(str(c)) or {}
                meaning = (info.get("meaning") or "").strip()
                ctype = (info.get("type") or "").strip()
                if meaning or ctype:
                    tag = f" ({ctype})" if ctype else ""
                    lines.append(f"    - {c}{tag}: {meaning}".rstrip())
            if len(lines) > 1:
                parts.append("\n".join(lines))
        if not parts:
            return ""
        return ("What each table and column means (use this to map the question "
                "to the correct column by MEANING, not by a similar-looking "
                "name):\n" + "\n\n".join(parts) + "\n")

    _STOP = {"the", "of", "a", "an", "is", "are", "in", "for", "to", "and", "as",
             "at", "all", "what", "which", "please", "me", "give", "with", "on",
             "by", "this", "that", "value", "values", "total", "type", "company"}

    def _score_tables(self, query: str) -> list:
        """Rank tables by whether they CAN ANSWER the question, BEFORE any
        filter is chosen: column/subject fit is the primary signal; a
        capitalised entity actually appearing in a table's cells is only a
        SECONDARY tie-breaker (capped), never the lead term. (Choosing the
        table by where a value sits would let a coincidental cell match in
        the wrong sheet decide the table -- the opposite of what we want.)"""
        q = {t for t in tokenize(query) if t not in self._STOP and len(t) > 2}
        caps = re.findall(r"[A-Z][\w&.\-]+(?:\s+[A-Z][\w&.\-]+)*", query)
        entity = max(caps, key=len) if caps else None
        scored: list[tuple[float, str]] = []
        for name, df in self.dataframes.items():
            col_tokens = set(tokenize(" ".join(str(c) for c in df.columns)))
            # Table NAME tokens matter too: 'LTL', 'CY01', 'deposit report'
            # live in the file/sheet name, not in any column. Without this a
            # question that names the table by its code scores 0 for it.
            name_tokens = {t for t in tokenize(str(name)) if len(t) >= 2}
            score = 0.0
            for qt in q:                                   # STRONG: named table
                for nt in name_tokens:
                    if qt == nt or (len(qt) >= 3 and len(nt) >= 3
                                    and (qt in nt or nt in qt)):
                        score += 2.0
                        break
            for qt in q:                                   # PRIMARY: column fit
                for ct in col_tokens:
                    if qt == ct or (len(qt) >= 3 and len(ct) >= 3 and (qt in ct or ct in qt)):
                        score += 1.0
                        break
            if entity and len(entity) > 3:                 # SECONDARY: value present
                try:
                    obj_cols = [c for c in df.columns if df[c].dtype.kind not in "iufcMb"]
                    for c in obj_cols[:20]:
                        s = df[c].astype(str).head(5000)
                        if s.str.contains(re.escape(entity), case=False, na=False).any():
                            score += 1.5   # capped tie-breaker, not dominant
                            break
                except Exception:
                    pass
            scored.append((score, name))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def _llm_pick_table(self, query: str, candidates: list) -> Optional[str]:
        """Ask the model to choose the ONE table that answers the question, given
        a compact catalogue (name + columns) of the shortlisted candidates."""
        if self.ollama is None or not candidates:
            return None
        lines = []
        for name, df in candidates:
            cols = ", ".join(str(c) for c in list(df.columns)[:40])
            lines.append(f"- {name}: columns = [{cols}]")
        prompt = (
            "Choose the ONE table that can answer the user's question. "
            "Reply with ONLY the exact table name from the list (copy it verbatim), "
            "nothing else.\n\n"
            f"Tables:\n" + "\n".join(lines) + f"\n\nQuestion: {query}\n\nTable name:"
        )
        try:
            raw = self.ollama.chat(
                cfg.model_for("pick_table", self.chat_model),
                [{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
            ).strip().strip("`'\"")
        except Exception:
            return None
        names = [n for n, _ in candidates]
        if raw in names:
            return raw
        nr = _norm_col(raw)
        for n in names:                                    # normalised match
            nn = _norm_col(n)
            if nn == nr or (nr and (nr in nn or nn in nr)):
                return n
        tail = raw.split(":")[-1].strip()                  # model returned sheet part only
        for n in names:
            if tail and tail in n:
                return n
        return None

    # How many of the most-relevant tables to expose to the analysis step.
    # The answer may span more than one sheet, so we do NOT collapse to a
    # single table: we surface the best few (best one first, so `df` is still
    # the most relevant) and let the generated code use `dfs` to reach the rest.
    MAX_ANALYSIS_TABLES = 4

    def _select_tables(self, query: str) -> dict:
        """Pick the most relevant table(s) for the question, BEST FIRST.

        Returns an ORDERED dict whose first entry is the single most relevant
        table (so `df` = that table, as before), followed by the next most
        relevant ones up to MAX_ANALYSIS_TABLES. Exposing a few candidates -
        instead of collapsing to exactly one - is what lets cross-sheet
        questions work: if the figure the user asked for lives on another
        sheet, the model can still reach it through `dfs`. With a single loaded
        table this is unchanged (it just returns that one)."""
        if len(self.dataframes) <= 1:
            return self.dataframes

        # Content routing FIRST. _score_tables ranks same-schema sheets as a
        # dead tie, and this is a SECOND, independent table selector (the plan
        # tier has its own): a fix applied only there leaves codegen still
        # reaching into the wrong sheet.
        if self.semantic_model is not None:
            try:
                from jarvisman.semantics.semantic_model import content_route
                routed = content_route(query, self.semantic_model,
                                       max_n=self.MAX_ANALYSIS_TABLES)
                if routed:
                    return {n: self.dataframes[n] for n in routed
                            if n in self.dataframes}
            except Exception:
                pass          # routing is an optimisation; never break the path

        scored = self._score_tables(query)
        best_score, best_name = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0.0

        # Tables with any positive relevance, best first; fall back to the top
        # few by score if nothing matched on keywords/entity.
        ranked = [n for s, n in scored if s > 0] or [n for _, n in scored]

        confident = best_score > 0 and (best_score - second >= 2.0)
        if confident:
            # Clear winner: lead with it, but still include the next best few
            # as context in case a sub-part of the answer lives elsewhere.
            ordered = [best_name] + [n for n in ranked if n != best_name]
        else:
            # Ambiguous: let the model name the single best, then lead with it;
            # the remaining candidates follow so nothing needed is dropped.
            shortlist = [(n, self.dataframes[n]) for n in ranked[:10]]
            picked = self._llm_pick_table(query, shortlist) or best_name
            ordered = [picked] + [n for n in ranked if n != picked]

        chosen_names = ordered[: self.MAX_ANALYSIS_TABLES]
        # Preserve order (Python dicts keep insertion order) so the first key
        # becomes `df` in the sandbox.
        return {n: self.dataframes[n] for n in chosen_names}

    def _mentions_known_column(self, ql: str) -> bool:
        for df in self.dataframes.values():
            for c in df.columns:
                cs = str(c).strip().lower()
                if len(cs) >= 3 and cs in ql:
                    return True
        return False

    # generic words that appear in table names but carry no routing signal
    _NAME_NOISE = {"data", "sheet", "sheet1", "sheet2", "table", "xlsx", "xls",
                   "file", "final", "new", "copy", "main", "the", "and"}

    def _mentions_known_table(self, ql: str) -> bool:
        """True when the question names a loaded TABLE (by code or by a word
        of its file/sheet name): 'the LTL balances', 'in CY01', 'the daily
        deposit report'. Such questions are tabular even when they also
        contain doc-sounding words like 'report'."""
        q_tokens = set(tokenize(ql))
        for name in self.dataframes:
            for t in tokenize(str(name)):
                if len(t) < 3 or t in self._NAME_NOISE:
                    continue
                if t in q_tokens or (len(t) >= 4 and t in ql):
                    return True
        return False

    # ------------------------------------------------------------------ #
    # Routing (cascade: heuristic first, LLM only when unsure)            #
    # ------------------------------------------------------------------ #
    def _route_fast(self, query: str) -> Optional[str]:
        """Decide the obvious cases without an LLM call. Returns a tool name,
        or None when the decision is ambiguous and the LLM router is needed."""
        ql = query.lower()

        # Greetings / thanks / chitchat -> conversational reply, never the data
        # pipeline. Exact-match (after stripping trailing !.?) so a real
        # question like "high interest accounts" is never swallowed.
        _q = ql.strip().rstrip("!.?")
        _greetings = {
            "hi", "hello", "hey", "yo", "hiya", "howdy", "sup",
            "good morning", "good afternoon", "good evening",
            "hi jarvis", "hello jarvis", "hey jarvis",
            "how are you", "hows it going", "how's it going",
            "whats up", "what's up",
            "thanks", "thank you", "thx", "cheers", "ok thanks",
            "who are you", "what can you do", "help",
        }
        if _q in _greetings:
            return "chat"

        has_tables = bool(self.dataframes)
        has_index = self.rag.vector_store.count > 0

        # questions ABOUT the conversation itself go to chat (where history is
        # available): "what did I ask before", "summarize our chat", etc.
        if self.history and any(p in ql for p in (
                "what did i ask", "what did we", "my last question",
                "my first question", "previous question", "earlier question",
                "our conversation", "our chat", "repeat your", "what was my",
                "summarize our", "what have i asked", "what did you say",
                "your last answer", "the last answer", "remind me what")):
            return "chat"

        if not has_tables and not has_index:
            return "chat"  # nothing loaded

        if has_tables and any(h in ql for h in _PLOT_HINTS):
            return "plot"

        if has_index and any(s in ql for s in _DOC_SIGNALS) \
                and not self._mentions_known_column(ql) \
                and not (has_tables and self._mentions_known_table(ql)):
            # 'what does the report say about X' is a prose question even when
            # it contains words like 'total' -- unless a table column OR a
            # loaded table itself is named ('the LTL report', 'the daily
            # deposit report' are tabular questions, not PDF ones).
            return "answer_docs"

        # Strong data signal -> analyze (this runs BEFORE the conversational
        # fallback, so a real question like "total funds for X" always wins).
        if has_tables and (
            any(h in ql for h in _ANALYZE_HINTS)
            or any(h in ql for h in _LOOKUP_HINTS)
            or self._mentions_known_column(ql)
            or self._mentions_known_table(ql)
        ):
            return "analyze"

        # No data signal at all? Catch conversational sentences the exact-match
        # greeting set misses ("Hello Jarvis, I have some questions", "can you
        # help me"). Only fires when nothing above matched.
        _has_data_signal = (
            any(h in ql for h in _ANALYZE_HINTS)
            or any(h in ql for h in _LOOKUP_HINTS)
            or any(h in ql for h in _DOC_SIGNALS)
            or self._mentions_known_column(ql)
        )
        if not _has_data_signal:
            _greet_words = ("hi", "hello", "hey", "greetings", "good morning",
                            "good afternoon", "good evening", "thanks", "thank")
            _looks_conversational = (
                any(_q.startswith(w) for w in _greet_words)
                or "how are you" in _q
                or "who are you" in _q
                or "what can you do" in _q
                or ("i have" in _q and "question" in _q)
            )
            if _looks_conversational:
                return "chat"

        if has_tables and not has_index:
            return "analyze"  # only tables exist -> query them
        if has_index and not has_tables:
            return "answer_docs"  # only prose exists -> retrieve
        return None  # both present, no strong signal -> ask the LLM router

    def route(self, query: str) -> dict:
        system = (
            "You are a router for an offline document assistant. Choose the single "
            "best tool for the user's message and reply with ONLY a JSON object, no "
            "prose, no code fences.\n"
            'Schema: {"tool": "answer_docs|analyze|plot|clarify|chat", "clarification": "<question, only if tool=clarify>"}\n'
            "Guidance: 'analyze' for ANY question about the spreadsheet/table data -- "
            "looking up a specific value, finding/listing/searching/filtering rows, or "
            "any total, average, count, or figure 'as at' a date. Use 'answer_docs' for "
            "questions answered from document PROSE (e.g. a PDF's narrative text). "
            "'plot' when the user wants a chart/graph of the data; 'clarify' when the "
            "request is too vague to act on; 'chat' for general conversation unrelated "
            "to the documents.\n"
            f"State: indexed_chunks={self.rag.vector_store.count}; "
            f"available_dataframes={self._schema_text()}"
        )
        raw = self.ollama.chat(
            cfg.model_for("route", self.chat_model),
            [{"role": "system", "content": system},
             {"role": "user", "content": (
                 (("Recent conversation:\n" + self._history_block(4) + "\n\n")
                  if self.history else "") + "Current message: " + query)}],
            options={"temperature": 0.0, "num_predict": 200},
            format="json",
        )
        parsed = _extract_json(raw)
        if parsed and parsed.get("tool") in _VALID_TOOLS:
            return parsed
        return {"tool": self._heuristic_tool(query)}

    def _heuristic_tool(self, query: str) -> str:
        return self._route_fast(query) or ("answer_docs" if self.rag.vector_store.count else "chat")

    # ------------------------------------------------------------------ #
    # Dispatch                                                           #
    # ------------------------------------------------------------------ #
    def handle(self, query: str, *args, **kwargs):

        try:
            res = self._handle_inner(query, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            # A question the pipeline cannot serve must not reach the user as a
            # raw traceback. The full detail is written to an error log, so
            # nothing is lost for debugging -- only the presentation changes.
            self._log_error(query, exc)
            return self._text_result(
                "I hit an internal error working on that one, so I have no "
                "answer I'd trust enough to show you. Try asking it a "
                "different way -- naming the sheet or the column usually "
                "helps.\n\n"
                f"(Technical detail for the log: {type(exc).__name__})",
                streamed=False)
        try:
            opts = res.get("options") or []
            if opts:
                from jarvisman.semantics.value_index import norm_text as _nt
                reg = getattr(self, "_recent_option_labels", [])
                reg.extend(_nt(o) for o in opts)
                self._recent_option_labels = reg[-24:]
        except Exception:
            pass
        # record the turn for conversational continuity (skip pure
        # clarification prompts -- they are not a completed answer)
        try:
            if not (res.get("options")):
                self._record_turn(query, res)
        except Exception:
            pass
        return res

    @staticmethod
    def _log_error(query: str, exc: BaseException) -> None:
        """Append a failed turn to errors.log. Never raises -- a logging
        problem must not replace the error the user already hit."""
        import datetime as _dt
        import os as _os
        import traceback as _tb
        try:
            d = _os.path.join(cfg.DATA_DIR, "audit")
            _os.makedirs(d, exist_ok=True)
            with open(_os.path.join(d, "errors.log"), "a",
                      encoding="utf-8") as fh:
                fh.write(f"\n=== {_dt.datetime.now():%Y-%m-%d %H:%M:%S} ===\n"
                         f"question: {query}\n{_tb.format_exc()}")
        except Exception:
            pass
    
    def _record_turn(self, query: str, res: dict) -> None:
        """Store a compact summary of this turn so later questions can refer
        back to it. We keep the question plus a short text rendering of the
        answer (and the matched value/table when present)."""
        answer = (res.get("text") or "").strip()
        # if the answer is a table, summarise from table_html text content
        if not answer and res.get("table_html"):
            import re as _re
            from html import unescape
            cells = _re.findall(r"<td[^>]*>(.*?)</td>", res["table_html"], _re.S)
            answer = ", ".join(unescape(_re.sub(r"<[^>]+>", "", c)).strip()
                               for c in cells[:12])
        answer = answer[:600]
        self.history.append({"q": query, "answer": answer,
                             "tool": res.get("tool")})
        if res.get("tool") == "analyze":
            self._last_analysis_q = query
            self._last_answer_text = answer
            # Learn the conversational subject from the plan that actually
            # ran. Previously _last_company was only ever set as a side effect
            # of code->name resolution, so a question that named the company
            # directly ("total for Acme Bank") left it empty and every
            # follow-up afterwards silently failed to carry over.
            try:
                subj = continuity.subject_from_plan(
                    getattr(self.reasoner, "last_plan", None))
                if subj:
                    self._last_company = subj
            except Exception:
                pass
        if len(self.history) > self.max_history_turns:
            self.history = self.history[-self.max_history_turns:]

    def _history_block(self, max_turns: int = 6) -> str:
        """Compact recent conversation for prompt injection."""
        if not self.history:
            return ""
        lines = []
        for t in self.history[-max_turns:]:
            q = (t.get("q") or "").strip()
            a = (t.get("answer") or "").strip().replace("\n", " ")
            if a and len(a) > 220:
                a = a[:220] + "..."
            lines.append(f"User: {q}\nAssistant: {a}")
        return "\n".join(lines)

    def clear_history(self) -> None:
        """Start a genuinely fresh conversation. Clearing only ``history`` left
        the remembered subject and last answer alive, so 'New conversation'
        could still answer a follow-up with the previous session's company."""
        self.history = []
        self._last_company = ""
        self._last_analysis_q = ""
        self._last_answer_text = ""
        self._recent_option_labels = []
        try:
            self.reasoner.last_plan = None
        except Exception:
            pass

    def _handle_inner(
        self,
        query: str,
        progress_callback: Optional[Callable[[str], None]] = None,
        token_callback: Optional[Callable[[str], None]] = None,
    ) -> dict:
        # Phase B: a pending clarification's option click arrives as the next
        # message -- bind it and re-run the ORIGINAL question, skipping routing.
        if self.dataframes and self.reasoner.pending_clarify:
            forced = self.reasoner.consume_option(query)
            if forced is not None:
                if forced.get("unmatched"):
                    # A pending clarification is open and this message is not
                    # one of its options. Never route it to the planner: a chip
                    # label is not a question, and answering it produces a real
                    # figure for the wrong thing.
                    result = self._text_result(
                        "I still need to know which reading you meant -- "
                        "please pick one of the options above, or rephrase "
                        "the question if neither fits.", streamed=False)
                    result["tool"] = "chat"
                    return result
                if forced.get("declined"):
                    result = self._text_result(
                        "No problem -- I won't make that assumption. "
                        "Feel free to rephrase or ask something else.",
                        streamed=False)
                    result["tool"] = "analyze"
                    return result
                result = self._run_analysis(
                    forced["question"], progress_callback,
                    forced_bind=forced["bind"], bind_term=forced.get("term", ""),
                    directive=forced.get("directive", ""),
                    constraint=forced.get("constraint"))
                result["tool"] = "analyze"
                return result
        from jarvisman.semantics.value_index import norm_text as _nt
        if _nt(query) in getattr(self, "_recent_option_labels", []):
            # a chip from an EARLIER exchange clicked after the conversation
            # moved on must never be treated as a data question (real-usage
            # find: "Each Bank name's own ..." was typo-scanned into nonsense)
            return self._text_result(
                "That choice belongs to an earlier question and has expired "
                "-- please ask the question again.", streamed=False)

        # ---- Conversation continuity -------------------------------------
        # (0) "remove the last row" -> the plan schema cannot express positional
        #     row removal, so this used to fall through to raw codegen and fail
        #     unpredictably. Answer it honestly instead of improvising.
        if continuity.is_positional_row_edit(query):
            result = self._text_result(
                continuity.positional_row_edit_reply(self._last_analysis_q),
                streamed=False)
            result["tool"] = "chat"
            return result
        # (a) "are you sure / is that right" -> re-explain the LAST answer,
        #     never recompute (a doubt-prompt must not change a correct figure).
        if self._last_answer_text and continuity.is_reexplain(query):
            try:
                prompt = continuity.reexplain_prompt(self._last_analysis_q,
                                                     self._last_answer_text)
                out = self.ollama.chat(
                    cfg.model_for("synthesize", self.chat_model),
                    [{"role": "user", "content": prompt}],
                    options={"temperature": 0.0, "num_predict": 320}).strip()
                if out:
                    result = self._text_result(out, streamed=False)
                    result["tool"] = "chat"
                    return result
            except Exception:
                pass
        # (b) implicit follow-up ("what about this?") -> carry the last company
        if self._last_company and continuity.wants_carryover(query, self._last_company):
            expanded = continuity.expand_followup(query, self._last_analysis_q,
                                                  self._last_company)
            if expanded and expanded != query:
                if progress_callback:
                    progress_callback(f"Continuing with {self._last_company} ...")
                result = self._run_analysis(expanded, progress_callback)
                # make the carried-over subject VISIBLE, never silent.
                # _run_analysis returns a dict; guard defensively regardless.
                if isinstance(result, dict):
                    if result.get("text"):
                        result["text"] = (f"(For {self._last_company})\n\n"
                                          + result["text"])
                    result["tool"] = "analyze"
                return result

        decision: Optional[dict] = None
        tool = self._route_fast(query)
        if tool is None:
            if progress_callback:
                progress_callback("Routing request ...")
            decision = self.route(query)
            tool = decision.get("tool", "answer_docs")

        if tool == "clarify":
            text = (decision or {}).get("clarification") or "Could you give me a bit more detail about what you need?"
            result = self._text_result(text, streamed=False)
        elif tool == "plot":
            if not self.dataframes:
                result = self._text_result(
                    "There is no tabular (Excel) data loaded to plot. Add an Excel file "
                    "and build the index first.",
                    streamed=False,
                )
            else:
                result = self._make_plot(query, progress_callback)
        elif tool == "analyze":
            if not self.dataframes:
                result = self._text_result(
                    "There is no tabular (Excel) data loaded to analyse. Add an Excel "
                    "file and build the index first.",
                    streamed=False,
                )
            else:
                result = self._run_analysis(query, progress_callback)
        elif tool == "chat":
            if progress_callback:
                progress_callback("Thinking ...")
            msgs = []
            hist = self._history_block()
            if hist:
                msgs.append({"role": "system",
                             "content": "Recent conversation for context "
                             "(use it to resolve references like 'that', "
                             "'the previous one', 'my last question'):\n" + hist})
            msgs.append({"role": "user", "content": query})
            text = self.ollama.chat(
                cfg.model_for("chat", self.chat_model),
                msgs,
                options={"temperature": 0.4},
                stream=bool(token_callback),
                on_token=token_callback,
            )
            result = self._text_result(text, streamed=bool(token_callback))
        else:  # answer_docs
            if self.rag.vector_store.count == 0:
                result = self._text_result(
                    "No documents have been indexed yet. Add PDF or Excel files and build the index.",
                    streamed=False,
                )
            else:
                if progress_callback:
                    progress_callback("Retrieving and answering ...")
                rag_result = self.rag.answer(query, token_callback=token_callback)
                answer = rag_result.get("text", "")
                sources = rag_result.get("sources", [])
                result = self._text_result(answer, streamed=bool(token_callback), sources=sources)
                # Add metadata to result
                result["retrieval_time"] = rag_result.get("retrieval_time", 0)
                result["llm_time"] = rag_result.get("llm_time", 0)
                result["chunks_used"] = rag_result.get("chunks_used", 0)
                result["confidence"] = rag_result.get("confidence", 0)

        result["tool"] = tool
        return result

    # ------------------------------------------------------------------ #
    # Plot tool                                                          #
    # ------------------------------------------------------------------ #
    def _plot_prompt(self, query: str, error: Optional[str] = None) -> str:
        prompt = (
            "Write Python code that creates ONE matplotlib figure answering the user's "
            "request about their data.\n"
            "Available variables (do NOT import anything): "
            "`pd` (pandas), `np` (numpy), `plt` (matplotlib.pyplot), "
            "`dfs` (dict of DataFrames by name), `df` (the first DataFrame).\n"
            f"DataFrames: {self._schema_text()}\n"
            "Rules: no imports; no file or network access; do not call plt.show() "
            "or save the figure to a file (no savefig); "
            "add a title and axis labels; leave the figure as the active figure.\n"
            "Reply with ONLY a Python code block.\n\n"
            f"User request: {query}"
        )
        if error:
            prompt += (
                f"\n\nThe previous attempt failed with this error:\n{error}\n"
                "Return corrected code following the same rules."
            )
        return prompt

    def _generate_and_run(self, prompt: str) -> tuple[dict, str]:
        raw = self.ollama.chat(
            cfg.model_for("codegen", self.chat_model),
            [{"role": "user", "content": prompt}], options={"temperature": 0.0}
        )
        code = _extract_code(raw)
        if not code:
            return {"ok": False, "error": "the model did not return runnable code.", "image": None, "stdout": ""}, ""
        code = _sanitize_code(code)  # same hygiene as the analysis path:
        # drop imports, neutralise pd.read_* re-reads of files from disk
        return run_sandboxed(code, self.dataframes, timeout=cfg.SANDBOX_TIMEOUT), code

    def _make_plot(self, query: str, progress_callback: Optional[Callable[[str], None]]) -> dict:
        # Phase C: plan-tier chart -- grounded, validated, deterministic
        # matplotlib over the compiled group-by result. Falls back to the
        # legacy codegen plot when the request doesn't fit a plan.
        if self.reasoner.ready and cfg.CHART_PLAN_ENABLED:
            outcome = self.reasoner.chart(query, progress_callback)
            if outcome.kind == 'ok' and outcome.image:
                res = self._text_result(outcome.text, streamed=False, code=outcome.code)
                res['type'] = 'plot'
                res['image'] = outcome.image
                res['provenance'] = outcome.provenance
                return res

        if progress_callback:
            progress_callback("Generating plot code ...")
        result, code = self._generate_and_run(self._plot_prompt(query))

        if not result["ok"] and code:
            if progress_callback:
                progress_callback("First attempt failed; retrying ...")
            result, code = self._generate_and_run(self._plot_prompt(query, error=result["error"]))

        if result["ok"]:
            res = self._text_result("Here is the chart:", streamed=False)
            res.update({"type": "plot", "image": result["image"], "code": code})
            return res
        return self._text_result(
            f"I could not generate that plot.\nReason: {result['error']}", streamed=False, code=code
        )

    # ------------------------------------------------------------------ #
    # Analyse tool (NL -> pandas over full tables, with self-debugging)  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _format_trace(trace: list) -> str:
        lines = "\n".join(f"  {t}" for t in trace)
        return "[decision trace]\n" + lines

    def _analysis_prompt(
        self, query: str, tables: Optional[dict] = None,
        error: Optional[str] = None, previous_code: Optional[str] = None,
        evidence: str = ""
    ) -> str:
        prompt = (
            "Write Python (pandas) code that answers the user's question about their "
            "tabular data. This may be a value lookup, filtering or searching rows, "
            "listing matching records, or an aggregation. The code runs over the FULL "
            "tables (every row), never a preview.\n"
            "Available variables (do NOT import anything): `pd` (pandas), `np` (numpy), "
            "`dfs` (dict of DataFrames keyed by 'file:sheet'), `df` (the first/most "
            "relevant DataFrame). The answer MAY require more than one of the tables in "
            "`dfs`: if the figure asked for is not in `df`, look through the other tables "
            "in `dfs` (iterate `dfs.items()`), and combine across them if needed.\n"
            f"Data:\n{self._schema_detail(tables)}\n"
            f"{self._table_meaning_block(tables)}"
            f"{evidence}"
            f"{self._exact_columns_block(tables)}"
            "Rules:\n"
            "- Use column names EXACTLY as written above (same case, spaces and "
            "punctuation). Do NOT invent, abbreviate, or reformat a column name.\n"
            "- To filter by a company/entity/name, first find which TEXT column actually "
            "contains that value (search the object/text columns if unsure); do not "
            "assume the column is named 'Company'.\n"
            "- CRITICAL - text matching: NEVER use exact equality (==) to filter a text "
            "column, because the spelling/case/spacing in the data may differ from the "
            "question (case and surrounding spaces often differ, e.g. 'abc' vs 'ABC' "
            "vs 'abc '). ALWAYS match text "
            "case-insensitively and tolerant of surrounding spaces, e.g. "
            "df['COL'].astype(str).str.strip().str.contains('value', case=False, na=False). "
            "The schema above lists the distinct values of each text column - match "
            "against those ACTUAL values, do not guess a value that is not shown.\n"
            "- CRITICAL - only the user's conditions: filter ONLY by the conditions the "
            "user actually stated. Do NOT add, invent, or assume any extra filter "
            "(a bank name, a product, a type) that the user did not ask for. If the user "
            "asked for items of one category, filter by THAT category column only and "
            "add nothing else.\n"
            "- CRITICAL - 'as at / by the end of / on <date>' (validity at a date): a row "
            "is valid at the date when START <= date AND (END >= date OR END is empty/"
            "NaT). An EMPTY end date means STILL ACTIVE and MUST be included. Never write "
            "just END <= date or END >= date alone -- that drops the active rows, which "
            "are usually the answer. Example: m = df['END'].isna() | (df['END'] >= "
            "pd.Timestamp(d)); keep = (df['START'] <= pd.Timestamp(d)) & m. If the table "
            "has NO start/end columns, it is a snapshot: apply NO date filter at all.\n"
            "- No imports; no file or network access.\n"
            "- The data is ALREADY loaded in `df` / `dfs`. NEVER create your own data: "
            "do not write pd.DataFrame(...), pd.Series(...), pd.read_excel(...) or any "
            "literal values. Only read from the provided `df` / `dfs`.\n"
            "- Assign the final answer to a variable named `result`.\n"
            "- If the answer is a single value, make `result` a clear string, e.g. "
            'result = "Total: {:.2f} EUR".format(total)".\n'
            "- If the answer is one or more matching rows, make `result` that DataFrame "
            "(optionally selecting the relevant columns).\n"
            "- If the answer is a table of computed figures, make `result` the DataFrame "
            "or Series.\n"
            "- Coerce text to numbers where needed (pd.to_numeric(col, errors='coerce')) "
            "and handle NaN sensibly.\n"
            "- After filtering, if you expected matches, it is good practice to also "
            "print how many rows matched, so an empty result is visible.\n"
            "- You may print intermediate steps for transparency.\n"
            "- When the result lists entities (companies, persons, banks), ALWAYS "
            "include BOTH the name column AND its matching code/id column of the "
            "same table (e.g. COMPANY NAME together with COMPANY CODE) when such "
            "a column exists.\n"
            f"{house_rules.prompt_block()}"
            "Reply with ONLY a Python code block.\n\n"
            f"User question: {query}"
        )
        if previous_code:
            prompt += (
                f"\n\nYour previous attempt was:\n```python\n{previous_code}\n```\n"
                f"It failed with this error:\n{error}\n"
                "In a brief comment, note what went wrong, then return corrected code "
                "following the same rules."
            )
        elif error:
            prompt += (
                f"\n\nThe previous attempt failed with this error:\n{error}\n"
                "Return corrected code following the same rules."
            )
        return prompt

    def _generate_and_query(self, prompt: str, tables: dict,
                            rewrite_hits=None) -> tuple[dict, str, list]:
        raw = self.ollama.chat(
            cfg.model_for("codegen", self.chat_model),
            [{"role": "user", "content": prompt}], options={"temperature": 0.0}
        )
        code = _extract_code_any(raw)
        if not code:
            return (
                {"ok": False, "error": "the model did not return runnable code.",
                 "text": None, "table_html": None, "stdout": ""},
                "", [],
            )
        code = _sanitize_code(code)  # drop imports / neutralise file re-reads
        # deterministic: align string literals to verified stored spellings
        # so a 'Karanikolaos Panagiotis' literal becomes the data's
        # 'Karanikolaos, Panagiotis' even if the model ignored the evidence
        if rewrite_hits:
            code, _rw = self._rewrite_literals_to_verified(code, rewrite_hits)
        # reject fabricated data: model building its OWN df/dfs instead of using ours
        if re.search(r"(?m)^\s*(?:df|dfs)\s*=\s*pd\.(?:DataFrame|Series|read_)", code):
            return (
                {"ok": False, "text": None, "table_html": None, "stdout": "",
                 "error": ("the code tried to CREATE or READ its own data instead of using "
                           "the provided `df`/`dfs`. Use ONLY the provided variables; do not "
                           "write pd.DataFrame(...), pd.Series(...) or pd.read_*().")},
                code, [],
            )
        # remap any non-existent column references BEFORE running (general, data-agnostic)
        fixed, missing = _validate_and_fix_columns(code, tables)
        return run_query(fixed, tables, timeout=cfg.SANDBOX_TIMEOUT), fixed, missing

    def _unmatched_value_hint(self, query: str, code: str, tables: dict):
        """When a query returns nothing, find string literals in the generated
        code that the user filtered on but that do NOT occur in any indexed
        value. For each: offer a did-you-mean if a close stored value exists,
        else state plainly it was not found. Returns a result dict or None
        (None = stay silent, e.g. nothing suspicious to report)."""
        import re as _re
        from jarvisman.semantics.value_index import norm_text
        vindex = self.value_index
        if vindex is None or not code:
            return None
        # values the code FILTERED on -- only literals inside .contains(...),
        # .isin([...]), == '...', not the df['COLUMN'] references. This avoids
        # mistaking a column name for a searched value.
        lits = set()
        for m in _re.finditer(r"\.(?:str\.)?contains\(\s*['\"]([^'\"]{3,60})['\"]", code):
            lits.add(m.group(1))
        for m in _re.finditer(r"\.isin\(\s*\[([^\]]*)\]", code):
            for v in _re.findall(r"['\"]([^'\"]{3,60})['\"]", m.group(1)):
                lits.add(v)
        for m in _re.finditer(r"==\s*['\"]([^'\"]{3,60})['\"]", code):
            lits.add(m.group(1))
        # exclude anything that is an actual column name in any table
        colnames = set()
        for df in (tables or {}).values():
            for c in getattr(df, "columns", []):
                colnames.add(norm_text(str(c)))
        lits = {l for l in lits if norm_text(l) not in colnames}
        entries = getattr(vindex, "entries", {}) or {}
        tindex = getattr(vindex, "token_index", {}) or {}
        all_tokens = getattr(vindex, "_all_tokens", None)
        if all_tokens is None:
            all_tokens = set()
            from jarvisman.semantics.value_index import tokenize_unicode
            for nv in entries:
                all_tokens.update(tokenize_unicode(nv))
        suspicious = []
        for lit in lits:
            ln = norm_text(lit)
            if len(ln) < 3 or ln.isdigit():
                continue
            # skip things that look like code/columns, dates, or that DO occur
            if any(ch in lit for ch in "[]{}()=<>") or _re.search(r"\d{4}", lit):
                continue
            toks = ln.split()
            present = ln in entries or ln in tindex or \
                all(t in all_tokens or t in tindex for t in toks if len(t) >= 3)
            if not present:
                suspicious.append(lit)
        if not suspicious:
            return None
        import difflib
        from jarvisman.semantics.value_index import tokenize_unicode
        lines = []
        for lit in suspicious[:3]:
            ln = norm_text(lit)
            lit_toks = [t for t in tokenize_unicode(ln) if len(t) >= 3]
            # (a) token-subset match, ORDER-INDEPENDENT: every typed token is a
            # token of some stored value ('Spiros Spyrou' -> 'Spyrou, Spiros').
            # Catches names the user typed in natural order when the data
            # stores "Surname, Firstname".
            subset_hits = {}
            if lit_toks:
                seed = max(lit_toks, key=len)
                for nv, locs in (tindex.get(seed) or []):
                    nv_toks = set(tokenize_unicode(nv))
                    if set(lit_toks) <= nv_toks:
                        for (_t, _c, _n, disp) in locs[:1]:
                            subset_hits[str(disp)] = True
            if subset_hits:
                opts = ", ".join(list(subset_hits)[:4])
                if len(subset_hits) == 1:
                    lines.append(f"\"{lit}\" matches {opts} in the data — but "
                                 f"the query returned nothing, so re-run using "
                                 f"that exact stored spelling.")
                else:
                    lines.append(f"\"{lit}\" could mean any of: {opts}. "
                                 f"Which one do you mean?")
                continue
            # (b) close whole-value match -> did you mean
            keys = [k for k in entries if k[:2] == ln[:2]]
            close = difflib.get_close_matches(ln, keys, n=3, cutoff=0.6)
            if close:
                shown = []
                for k in close:
                    for (_t, _c, _n, disp) in entries[k][:1]:
                        shown.append(str(disp))
                opts = ", ".join(dict.fromkeys(shown))
                lines.append(f"I couldn't find \"{lit}\" in the data. "
                             f"Did you mean: {opts}?")
            else:
                lines.append(f"I couldn't find \"{lit}\" anywhere in the data, "
                             f"so there are no matching rows.")
        return self._text_result("\n".join(lines), streamed=False)

    @staticmethod
    def _looks_empty(result: dict) -> bool:
        """Heuristic: did a successful run produce an empty/zero answer that is
        most likely a filter-matched-nothing mistake? General - no knowledge of
        the data. Looks at the text/stdout for a bare 0, 0.0, empty table, or an
        explicit 'no rows' shape."""
        text = (result.get("text") or "").strip()
        stdout = (result.get("stdout") or "").strip()
        blob = (text + " " + stdout).strip().lower()
        if not blob:
            return True
        # A result that is just zero (with optional currency/formatting).
        bare = re.sub(r"[^0-9a-z]", "", blob)
        if bare in ("0", "00", "000"):
            return True
        # "total: 0.00 eur" style
        if re.fullmatch(r"(total|sum|result)?:?\s*0(\.0+)?\s*[a-z]{0,4}", blob):
            return True
        # An empty DataFrame string representation, or the friendly no-rows line.
        if "empty dataframe" in blob or "columns: []" in blob:
            return True
        if "no matching rows" in blob:
            return True
        return False

    @staticmethod
    def _rewrite_literals_to_verified(code: str, hits) -> tuple:
        """A model writing fallback code copies the USER'S spelling into
        string literals ('Karanikolaos Panagiotis'), but the data stores
        'Karanikolaos, Panagiotis' -- and even str.contains misses across
        the comma. Determinism, not model obedience: every string literal in
        the generated code whose TOKEN SET matches a verified stored value
        (but whose text differs) is rewritten to the data's exact spelling
        before execution. Returns (code, notes)."""
        import ast as _ast
        from jarvisman.semantics.value_index import norm_text as _nt, tokenize_unicode as _tk
        verified = {}
        for h in hits or []:
            d = str(h.display)
            toks = frozenset(t for t in _tk(_nt(d)) if len(t) >= 2)
            if len(toks) >= 2:
                verified.setdefault(toks, d)
        if not verified:
            return code, []
        try:
            tree = _ast.parse(code)
        except SyntaxError:
            return code, []
        notes, done = [], set()
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Constant) and isinstance(node.value, str):
                lit = node.value
                if len(lit) < 6 or lit in done:
                    continue
                ltoks = frozenset(t for t in _tk(_nt(lit)) if len(t) >= 2)
                stored = verified.get(ltoks)
                if stored and stored != lit and _nt(stored) != _nt(lit):
                    for q in (repr(lit),
                              '"' + lit + '"' if '"' not in lit else None):
                        if q and q in code:
                            code = code.replace(q, repr(stored))
                            notes.append(f"matched '{lit}' to the data's "
                                         f"spelling '{stored}'")
                            done.add(lit)
                            break
        return code, notes

    def _run_analysis(self, query: str, progress_callback: Optional[Callable[[str], None]] = None,
                      forced_bind: Optional[dict] = None, bind_term: str = "",
                      directive: str = "",
                      constraint: Optional[dict] = None) -> dict:
        # Company CODE -> NAME resolution, done ONCE here so BOTH the plan tier
        # and the codegen fallback see the rewritten query. (A rewrite inside
        # the reasoner would not reach codegen, which reads this `query`.)
        try:
            new_q, _rnotes, _clarify = code_resolver.resolve_in_question(
                query, self.dataframes)
            if _clarify:
                return self._text_result(_clarify, streamed=False)
            if new_q != query:
                if progress_callback:
                    progress_callback("Resolved company code to name ...")
                query = new_q
            for _n in (_rnotes or []):
                _m = _n.split("to '", 1)
                if len(_m) == 2:
                    self._last_company = _m[1].rstrip("'")
        except Exception:
            pass
        # Tier 1: structured plan over the semantic model -- grounded against
        # the actual cell values, validated and compiled BEFORE execution.
        evidence_block = ""
        fallback_trace_init: list = []
        if self.reasoner.ready:
            outcome = self.reasoner.answer(query, progress_callback,
                                           forced_bind=forced_bind,
                                           bind_term=bind_term,
                                           directive=directive,
                                           constraint=constraint)
            if outcome.kind == "ok":
                text = outcome.text or "(no result produced)"
                if cfg.EXPLAIN and outcome.trace:
                    text = self._format_trace(outcome.trace) + "\n\n" + text
                res = self._text_result(text, streamed=False, code=outcome.code)
                res["table_html"] = outcome.table_html
                res["provenance"] = outcome.provenance
                res["trace"] = outcome.trace
                res["prose"] = outcome.prose
                res["row_count"] = outcome.row_count
                if outcome.image:
                    res["type"] = "plot"
                    res["image"] = outcome.image
                return res
            if outcome.kind == "clarify" and outcome.text:
                res = self._text_result(outcome.text, streamed=False)
                res["options"] = outcome.options
                return res
            # fall through to codegen, carrying the grounding evidence AND
            # the trace that explains why the plan tier did not answer
            evidence_block = outcome.evidence_block
            fallback_trace_init = list(outcome.trace or [])
        if directive and not self.reasoner.ready:
            evidence_block = ("INTERPRETATION (confirmed by the user -- follow "
                              "it EXACTLY): " + directive + "\n") + evidence_block
        fallback_trace: list = list(fallback_trace_init or [])
        value_hits = []
        if self.semantic_model is not None and self.value_index is not None:
            _ev = ground(query, self.semantic_model, self.value_index)
            value_hits = _ev.value_hits
            if not evidence_block:
                evidence_block = _ev.to_prompt_block()

        # Tier 2: raw pandas codegen (the long tail), now evidence-fed.
        tables = self._select_tables(query)  # relevant sheet(s); df = most relevant
        real_cols = sorted({str(c) for df in tables.values() for c in df.columns})
        attempts = cfg.ANALYSIS_MAX_RETRIES + 1
        prev_code: Optional[str] = None
        error: Optional[str] = None
        result: dict = {}
        code = ""
        spelling_subs: list = []
        for attempt in range(attempts):
            if progress_callback:
                progress_callback(
                    "Writing analysis code ..." if attempt == 0
                    else f"Fixing analysis code (attempt {attempt + 1}/{attempts}) ..."
                )
            prompt = self._analysis_prompt(query, tables=tables, error=error,
                                           previous_code=prev_code, evidence=evidence_block)
            result, code, missing = self._generate_and_query(
                prompt, tables, rewrite_hits=value_hits)
            if result["ok"] and not self._looks_empty(result):
                break
            if result["ok"] and self._looks_empty(result):
                # Code ran but produced an empty/zero answer. FIRST try a
                # zero-LLM rescue: rewrite free-typed string literals to their
                # verified stored spellings ('Koutouvas Athanasios' ->
                # 'Koutouvas,Athanasios') and re-run. Only if that does not
                # help do we spend an LLM retry.
                if code and self.value_index is not None:
                    from jarvisman.semantics.value_index import resolve_code_literals
                    rc, rsubs = resolve_code_literals(code, self.value_index,
                                                      tables)
                    if rsubs:
                        if progress_callback:
                            progress_callback(
                                "Re-checking with the verified spelling ...")
                        rr = run_query(rc, tables, timeout=cfg.SANDBOX_TIMEOUT)
                        if rr.get("ok") and not self._looks_empty(rr):
                            result, code = rr, rc
                            spelling_subs = rsubs
                            break
                # Retry once with a hint instead of returning a misleading 0.
                if attempt < attempts - 1:
                    error = (
                        "Your code ran but the result was empty or zero, which usually "
                        "means a filter matched NO rows. Re-check every text filter: match "
                        "case-insensitively and strip spaces "
                        "(df['COL'].astype(str).str.strip().str.contains('value', case=False, "
                        "na=False)), and use only values that actually appear in the column's "
                        "listed distinct values. Do NOT add any filter the user did not ask "
                        "for. If after correct matching the result is genuinely zero, set "
                        "`result` to a string saying no matching rows were found."
                    )
                    prev_code = code
                    continue
                break
            raw_err = result.get("error") or ""
            # A syntax error (or a bare "did not return runnable code") won't be
            # helped by column or spelling repair -- the code never parsed. Give
            # the model the exact failure and demand a clean, prose-free block,
            # which is what it actually needs to correct itself.
            if ("syntax error" in raw_err.lower()
                    or "did not return runnable code" in raw_err.lower()) \
                    and attempt < attempts - 1:
                error = (
                    f"Your previous response was NOT valid Python and could not "
                    f"be parsed ({raw_err}). Return ONLY a single Python code "
                    "block that parses cleanly on its own: no sentences or "
                    "explanation before or after the code, nothing outside the "
                    "```python fence, no unfinished lines and no unbalanced "
                    "brackets or quotes. Use the provided `df`/`dfs` and assign "
                    "the final answer to `result`."
                )
                prev_code = code or prev_code
                continue
            # deterministic spelling rescue on the ERROR path too: a wrong
            # free-typed literal ('Koutouvas Athanasios') may sit alongside a
            # fixable error; rewriting it to the stored value and re-running
            # can resolve the attempt with zero LLM calls.
            if code and self.value_index is not None:
                from jarvisman.semantics.value_index import resolve_code_literals as _rcl
                _rc, _subs = _rcl(code, self.value_index, tables)
                if _subs:
                    _rr = run_query(_rc, tables, timeout=cfg.SANDBOX_TIMEOUT)
                    if _rr.get("ok") and not self._looks_empty(_rr):
                        result, code = _rr, _rc
                        spelling_subs = _subs
                        break
            # reactive single-column repair (e.g. a KeyError the pre-pass didn't catch)
            repaired = _repair_columns(code, raw_err, tables)
            if repaired and repaired != code:
                if progress_callback:
                    progress_callback("Auto-correcting a column name ...")
                fixed = run_query(repaired, tables, timeout=cfg.SANDBOX_TIMEOUT)
                if fixed["ok"]:
                    result, code = fixed, repaired
                    break
                raw_err = fixed.get("error") or raw_err
            # build a precise, general instruction for the next attempt
            ke = _KEYERROR_RE.search(raw_err)
            bad_list = sorted(set(missing) | ({ke.group(1)} if ke else set()))
            parts = []
            if bad_list:
                parts.append("These columns do NOT exist: " + ", ".join(repr(b) for b in bad_list) + ".")
            parts.append("Use ONLY these exact columns: " + ", ".join(repr(c) for c in real_cols) + ".")
            if self._looks_empty(result) and code and (
                    "Timestamp" in code or "datetime" in code
                    or "<= '" in code or ">= '" in code):
                parts.append("The result was EMPTY and your code filtered on a date: if "
                             "this is an 'as at <date>' question, you very likely "
                             "excluded rows whose END date is empty (still active). "
                             "Include empty/NaT end dates: END.isna() | (END >= date).")
            parts.append("Do NOT silently ignore any condition the user asked for (a "
                         "unit/currency, a date such as 'as at <date>', an "
                         "entity/company name). If a needed column genuinely does not "
                         "exist in ANY of the provided tables, do not drop the condition "
                         "and guess: instead set `result` to a short string that states "
                         "exactly which condition could not be applied and why, so the "
                         "user knows the figure is not filtered by it.")
            error = " ".join(parts)
            prev_code = code or prev_code

        if result.get("ok"):
            # deterministic spelling rescue: an empty result whose code
            # contains a free-typed literal that uniquely token-matches a
            # stored value ('Karanikolaos Panagiotis' vs stored
            # 'Karanikolaos,Panagiotis') is re-run ONCE with the verified
            # spelling -- zero LLM calls, substitution disclosed.
            subs = list(spelling_subs or [])
            txt0 = (result.get("text") or "")
            looks_empty = txt0.startswith("Empty DataFrame") or (
                not any(ch.isdigit() for ch in txt0)
                and not (result.get("stdout") or "").strip())
            if looks_empty and not subs and code and self.value_index is not None:
                from jarvisman.semantics.value_index import resolve_code_literals
                new_code, subs = resolve_code_literals(
                    code, self.value_index, tables)
                if subs:
                    if progress_callback:
                        progress_callback(
                            "Re-checking with the verified spelling ...")
                    retried = run_query(new_code, tables,
                                        timeout=cfg.SANDBOX_TIMEOUT)
                    if retried.get("ok") and not (retried.get("text") or "")\
                            .startswith("Empty DataFrame"):
                        result, code = retried, new_code
                    else:
                        subs = []
            answer = result.get("text")
            stdout = (result.get("stdout") or "").strip()
            if not answer and stdout:
                answer = stdout
            from jarvisman.runtime import numfmt
            answer = numfmt.reformat(answer or "(no result produced)")
            if subs:
                answer = "".join(
                    f"(interpreted {s!r} as the stored value {d!r})\n"
                    for s, d in subs) + answer
            tr = fallback_trace or None
            if cfg.EXPLAIN and tr:
                answer = self._format_trace(tr + ["answered via codegen tier"]) \
                    + "\n\n" + answer
            res = self._text_result(answer, streamed=False, code=code)
            res["table_html"] = result.get("table_html")
            res["trace"] = tr
            # Honest empty result: if the answer is empty AND a filter value the
            # user typed does not exist in the data, say so plainly (or offer a
            # did-you-mean when a close stored value exists) instead of
            # returning a bare empty table. Removes the asymmetry where a
            # near-miss name got a suggestion but a not-found name silently
            # produced nothing.
            if self._looks_empty(result) and self.value_index is not None:
                hint = self._unmatched_value_hint(query, code, tables)
                if hint is not None:
                    return hint
            return res

        # Everything failed (plan rejected AND codegen could not compute it).
        # Before dead-ending, let the UNDERSTANDING model propose ONE concrete
        # way forward as a yes/no question -- model-written, not templated.
        proposal = None
        if cfg.PROPOSE_ALTERNATIVES and self.ollama is not None \
                and not directive:
            try:
                from jarvisman.planning import proposals
                schema = self._table_meaning_block(tables) or self._schema_detail(tables)
                issues = list(self._fallback_trace) if hasattr(self, "_fallback_trace") else []
                proposal = proposals.propose_alternative(
                    self.ollama, self.chat_model, query, schema,
                    fallback_trace, fallback_trace, progress_callback)
            except Exception:
                proposal = None
        if proposal:
            # reuse the clarify machinery: a yes/no with the accept_directive
            self.reasoner.pending_clarify = {
                "question": query, "term": "", "kind": "proposal",
                "options": [
                    {"label": _yes_label(), "display": "",
                     "directive": proposal["accept_directive"]},
                    {"label": _no_label(), "display": "", "directive": ""},
                ]}
            res = self._text_result(proposal["proposal"], streamed=False)
            res["options"] = [_yes_label(), _no_label()]
            return res
        return self._text_result(
            f"I could not compute that after {attempts} attempt(s).\nReason: {result.get('error')}",
            streamed=False,
            code=code,
        )

    # ------------------------------------------------------------------ #
    # Result helper                                                      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _text_result(text: str, streamed: bool, sources=None, code=None) -> dict:
        return {
            "type": "text",
            "text": text,
            "streamed": streamed,
            "sources": sources or [],
            "image": None,
            "table_html": None,
            "code": code,
            "tool": None,
        }