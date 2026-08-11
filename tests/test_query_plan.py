"""The plan compiler is the trusted, deterministic core: it turns a validated
QueryPlan into pandas code. These tests pin three things end-to-end (compile
THEN run in the real sandbox):

  * a group-sum excludes Total rows and returns the right figures,
  * an unknown column is REJECTED (no bundle) rather than silently guessed,
  * every literal is emitted via repr() -- so a value that looks like code is
    inert (the injection-safety guarantee that lets the compiled code be
    trusted).
"""
from __future__ import annotations

import pandas as pd

from jarvisman.runtime import sandbox
from jarvisman.planning.query_plan import QueryPlan, compile_plan
from jarvisman.semantics.semantic_model import build_semantic_model
from jarvisman.semantics.value_index import ValueIndex


def _compile(plan_dict, tables, model=None, vindex=None):
    model = model or build_semantic_model(tables)
    vindex = vindex or ValueIndex.build(tables)
    plan = QueryPlan.from_dict(plan_dict)
    return compile_plan(plan, model, tables, vindex)


# --------------------------------------------------------------------------- #
# Correctness + provenance                                                   #
# --------------------------------------------------------------------------- #
def test_group_sum_excludes_total_row(tables, model, vindex):
    bundle, issues, notes = _compile(
        {
            "table": "book.xlsx:MATRIX",
            "group_by": ["COMPANY"],
            "aggregations": [{"fn": "sum", "column": "AMOUNT EURO"}],
        },
        tables, model, vindex,
    )
    assert bundle is not None, [i.message for i in issues]
    # provenance must record the exclusion
    assert any("Total" in e or "Subtotal" in e for e in bundle.explain)

    r = sandbox.run_query(bundle.code, tables, timeout=20)
    assert r["ok"], r.get("error")
    text = r.get("text") or ""
    assert "Lignum" in text and "Kronospan" in text
    assert "Total" not in text          # the summary row was dropped, not summed


def test_distinct_row_listing(tables, model, vindex):
    # CY05 has 'Spyrou,Spiros' twice; a select of DIRECTOR_NAME must be distinct
    bundle, issues, _ = _compile(
        {"table": "book.xlsx:CY05", "select": ["DIRECTOR_NAME"]},
        tables, model, vindex,
    )
    assert bundle is not None, [i.message for i in issues]
    r = sandbox.run_query(bundle.code, tables, timeout=20)
    assert r["ok"], r.get("error")
    # two unique directors across three rows
    assert (r.get("text") or "").count("Spyrou") == 1


# --------------------------------------------------------------------------- #
# Rejection (no silent guessing)                                             #
# --------------------------------------------------------------------------- #
def test_unknown_column_is_rejected(tables, model, vindex):
    bundle, issues, _ = _compile(
        {
            "table": "book.xlsx:MATRIX",
            "aggregations": [{"fn": "sum", "column": "DOES_NOT_EXIST"}],
        },
        tables, model, vindex,
    )
    assert bundle is None
    assert any(i.kind == "unknown_column" for i in issues)


def test_unknown_aggregation_is_rejected(tables, model, vindex):
    bundle, issues, _ = _compile(
        {
            "table": "book.xlsx:MATRIX",
            "aggregations": [{"fn": "hack", "column": "AMOUNT EURO"}],
        },
        tables, model, vindex,
    )
    assert bundle is None
    assert any(i.kind == "bad_agg" for i in issues)


# --------------------------------------------------------------------------- #
# Injection safety (literals are repr'd, never code)                         #
# --------------------------------------------------------------------------- #
def test_value_that_looks_like_code_is_inert():
    payload = "Acme'); import os; os.system('x'); ('"
    tables = {
        "x.xlsx:S": pd.DataFrame(
            {"COMPANY": [payload, "Normal"], "AMOUNT EURO": [10.0, 20.0]}
        )
    }
    bundle, issues, _ = _compile(
        {
            "table": "x.xlsx:S",
            "filters": [{"column": "COMPANY", "op": "eq", "value": payload}],
            "aggregations": [{"fn": "sum", "column": "AMOUNT EURO"}],
        },
        tables,
    )
    assert bundle is not None, [i.message for i in issues]
    # the compiled code must still pass the AST gate (the payload is a string
    # literal, not executable code) ...
    ok, reason = sandbox.validate_code(bundle.code)
    assert ok, reason
    # ... and running it yields the one matching row's sum, with no side effect
    r = sandbox.run_query(bundle.code, tables, timeout=20)
    assert r["ok"], r.get("error")
    assert "10" in (r.get("text") or "")
