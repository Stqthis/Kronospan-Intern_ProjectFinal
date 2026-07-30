#!/usr/bin/env python3
"""Fast smoke-test for JarvisMan: runs questions through the SAME path the UI
uses (Agent.handle) and reports the answer, the elapsed time, and -- crucially
-- WHICH TABLE the answer was computed from.

The old version called rag.answer(), which is the PDF/vector text path: it
never touches the spreadsheets, so every tabular question came back "the
indexed documents do not contain information relevant to that question" and
was still scored a success. Table questions must go through Agent.handle.

Usage:
    python test_queries.py                  # run every question
    python test_queries.py -k caixa         # only questions matching 'caixa'
    python test_queries.py -n 4             # only question 4
    python test_queries.py --list           # show questions, run nothing
    python test_queries.py -f mine.jsonl    # a different question file

Question file (JSONL, one object per line):
    {"q": "Total funds per currency with CaixaBank as at 31.07.2023?",
     "expect": ["EUR"],                 # substrings that must appear
     "expect_num": [19455244],          # numbers that must appear (0.1% tol;
                                        #   also accepts x1000 / /1000 scale
                                        #   for '000-denominated sheets)
     "expect_table": "31-07-2023"}      # substring of the table it must use
All expect fields are optional; a question with none is reported as
INFO (answered, nothing asserted) rather than pass or fail.

DGX Spark / remote Ollama:
    python test_queries.py --host http://localhost:11434
    OLLAMA_HOST=http://<host>:11434 python test_queries.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

os.environ.setdefault("RAG_EXPLAIN", "1")   # capture traces for failures

# NOTE: jarvisman imports are deliberately deferred into _boot() -- they pull in
# faiss/torch and take seconds. --help and --list must stay instant.

DEFAULT_QUESTIONS = "questions.jsonl"
RESULTS_FILE = "test_results.json"

# generated code addresses tables as dfs['<table name>'] (see sandbox._run_user_code)
_DFS_RE = re.compile(r"""dfs\[\s*['"]([^'"]+)['"]\s*\]""")


def load_questions(path: str) -> list:
    """Read JSONL, skipping blanks and # comments. Reports the offending line
    number on bad JSON rather than dying with a bare traceback."""
    cases = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as exc:
                sys.exit(f"{path}:{n}: bad JSON — {exc}")
            if not obj.get("q"):
                sys.exit(f"{path}:{n}: missing 'q'")
            cases.append(obj)
    if not cases:
        sys.exit(f"{path}: no questions found")
    return cases


def tables_used(result: dict) -> list:
    """Which tables the answer actually came from, read out of the generated
    code. This is the signal that catches wrong-sheet answers -- a plausible
    number from the wrong file looks perfect until you check this."""
    code = result.get("code") or ""
    seen, out = set(), []
    for name in _DFS_RE.findall(code):
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _numbers_in(text: str) -> list:
    """Every number in the text, commas stripped -> floats."""
    out = []
    for m in _NUM_RE.findall(text or ""):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _num_close(got: float, want: float) -> bool:
    """True when got matches want within 0.1% (or 1.0 absolute), at scale
    1x, 1000x or 1/1000 — WCR-style sheets are '000-denominated, so a
    correct answer may legitimately be a thousand times the sheet value."""
    for scale in (1.0, 1000.0, 0.001):
        w = want * scale
        if abs(got - w) <= max(1.0, abs(w) * 0.001):
            return True
    return False


def grade(case: dict, result: dict) -> tuple:
    """-> (status, reason). status is PASS / FAIL / INFO."""
    text = (result.get("text") or "")
    table_html = result.get("table_html") or ""
    haystack = (text + " " + table_html).lower()
    used = tables_used(result)

    problems = []

    want_table = case.get("expect_table")
    if want_table:
        if not used:
            problems.append(f"no table used (wanted ~{want_table})")
        elif not any(want_table.lower() in u.lower() for u in used):
            problems.append(f"used {used}, wanted ~{want_table}")

    for token in (case.get("expect") or []):
        if str(token).lower() not in haystack:
            problems.append(f"missing {token!r}")

    want_nums = case.get("expect_num") or []
    if want_nums:
        got_nums = _numbers_in(text + " " + table_html)
        for wn in want_nums:
            if not any(_num_close(g, float(wn)) for g in got_nums):
                problems.append(f"missing number {wn}")

    if not want_table and not case.get("expect") and not want_nums:
        return "INFO", "answered; nothing asserted"
    if problems:
        return "FAIL", "; ".join(problems)
    return "PASS", "ok"


def _boot():
    """Import the heavy stack only when we are really going to run questions."""
    from jarvisman import config as cfg
    from jarvisman.agent import Agent
    from jarvisman.llm.ollama_client import OllamaClient
    from jarvisman.retrieval.rag import RAGPipeline
    from jarvisman.retrieval.vector_store import VectorStore
    return cfg, Agent, OllamaClient, RAGPipeline, VectorStore


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-f", "--file", default=DEFAULT_QUESTIONS)
    ap.add_argument("-k", "--filter", help="only questions containing this text")
    ap.add_argument("-n", "--number", type=int, help="only question N (1-based)")
    ap.add_argument("--list", action="store_true", help="list questions and exit")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print the full answer and generated code")
    ap.add_argument("--host", help="Ollama host, e.g. http://localhost:11434 "
                    "(overrides OLLAMA_HOST for this run)")
    args = ap.parse_args()
    if args.host:
        os.environ["OLLAMA_HOST"] = args.host

    cases = load_questions(args.file)
    # narrow BEFORE listing, so `--list -k caixa` previews exactly what
    # `-k caixa` would run
    if args.number:
        if not 1 <= args.number <= len(cases):
            sys.exit(f"-n must be 1..{len(cases)}")
        cases = [cases[args.number - 1]]
    if args.filter:
        cases = [c for c in cases if args.filter.lower() in c["q"].lower()]
        if not cases:
            sys.exit(f"no question matches {args.filter!r}")
    if args.list:
        for i, c in enumerate(cases, 1):
            exp = []
            if c.get("expect_table"):
                exp.append(f"table~{c['expect_table']}")
            if c.get("expect"):
                exp.append(f"contains {c['expect']}")
            print(f"{i:2}. {c['q']}")
            if exp:
                print(f"    expects: {'; '.join(exp)}")
        return 0

    cfg, Agent, OllamaClient, RAGPipeline, VectorStore = _boot()
    if args.host:
        cfg.OLLAMA_HOST = args.host      # config was imported before the flag

    print(f"Connecting to Ollama at {cfg.OLLAMA_HOST} ...")
    oll = OllamaClient(cfg.OLLAMA_HOST)
    if not oll.is_alive():
        sys.exit(f"Ollama is not reachable at {cfg.OLLAMA_HOST}. "
                 f"Start it with 'ollama serve'.")
    store = VectorStore()
    rag = RAGPipeline(oll, store, cfg.DEFAULT_CHAT_MODEL, cfg.DEFAULT_EMBED_MODEL)

    print(f"Loading index from {cfg.INDEX_DIR} ...")
    dfs = rag.load_persisted(cfg.INDEX_DIR) or {}
    if not dfs:
        sys.exit("No tables in the index. Build it in the UI first "
                 "(Add files -> Build / Update Index).")
    agent = Agent(oll, rag, cfg.DEFAULT_CHAT_MODEL)
    agent.dataframes = dfs
    print(f"{len(dfs)} table(s) loaded. Running {len(cases)} question(s) "
          f"on {cfg.DEFAULT_CHAT_MODEL}.\n")

    results, tally = [], {"PASS": 0, "FAIL": 0, "INFO": 0, "ERROR": 0}
    t_all = time.monotonic()

    for i, case in enumerate(cases, 1):
        q = case["q"]
        print(f"[{i:2}/{len(cases)}] {q[:66]}{'...' if len(q) > 66 else ''}")
        t0 = time.monotonic()
        try:
            res = agent.handle(q)
            # Defensive: some code paths / older builds returned tuples of
            # (result, extra). Normalise so grading never crashes with
            # "'tuple' object has no attribute 'get'".
            if isinstance(res, tuple) and res and isinstance(res[0], dict):
                res = res[0]
            if not isinstance(res, dict):
                res = {"text": str(res)}
        except Exception as exc:                      # keep going; report at end
            secs = time.monotonic() - t0
            tally["ERROR"] += 1
            print(f"         ERROR  {secs:5.1f}s  {type(exc).__name__}: {exc}\n")
            results.append({"q": q, "status": "ERROR", "error": str(exc),
                            "seconds": round(secs, 2)})
            continue
        secs = time.monotonic() - t0
        status, why = grade(case, res)
        tally[status] += 1
        used = tables_used(res)
        print(f"         {status:5}  {secs:5.1f}s  "
              f"table={used[0] if used else '-'}"
              f"{f' (+{len(used) - 1})' if len(used) > 1 else ''}")
        if status != "PASS":
            print(f"                why: {why}")
        if status == "FAIL" or args.verbose:
            print(f"                got: {(res.get('text') or '')[:150]}")
        if args.verbose and res.get("code"):
            print("                --- code ---")
            for ln in (res["code"] or "").splitlines():
                print(f"                {ln}")
        print()
        results.append({"q": q, "status": status, "why": why,
                        "seconds": round(secs, 2), "tables_used": used,
                        "text": res.get("text"), "code": res.get("code"),
                        "tool": res.get("tool")})

    total = time.monotonic() - t_all
    print("=" * 62)
    print(f"PASS {tally['PASS']}   FAIL {tally['FAIL']}   "
          f"ERROR {tally['ERROR']}   INFO {tally['INFO']}   "
          f"[{total:.1f}s total, {total / max(len(cases), 1):.1f}s avg]")
    with open(RESULTS_FILE, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"Details written to {RESULTS_FILE}")
    try:
        from jarvisman.runtime import scoreboard
        scoreboard.record("answer_checks", dict(tally))
    except Exception:
        pass
    return 1 if (tally["FAIL"] or tally["ERROR"]) else 0


if __name__ == "__main__":
    sys.exit(main())
