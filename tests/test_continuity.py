"""Continuity: the remembered subject comes from the executed plan."""
from jarvisman.planning import continuity


def _plan(filters):
    return {"table": "accounts", "filters": filters}


def test_subject_prefers_name_like_column():
    plan = _plan([
        {"column": "YEAR", "op": "eq", "value": "2024"},
        {"column": "Customer Name", "op": "eq", "value": "Acme Bank"},
    ])
    assert continuity.subject_from_plan(plan) == "Acme Bank"


def test_subject_ignores_bare_years_and_numbers():
    plan = _plan([{"column": "YEAR", "op": "eq", "value": "2024"}])
    assert continuity.subject_from_plan(plan) == ""


def test_subject_falls_back_to_any_string_equality():
    plan = _plan([{"column": "Region", "op": "match", "value": "Cyprus"}])
    assert continuity.subject_from_plan(plan) == "Cyprus"


def test_subject_ignores_range_filters():
    plan = _plan([{"column": "Balance", "op": "gt", "value": "1000"}])
    assert continuity.subject_from_plan(plan) == ""


def test_subject_handles_missing_or_empty_plan():
    assert continuity.subject_from_plan(None) == ""
    assert continuity.subject_from_plan({}) == ""


def test_carryover_fires_once_a_subject_is_known():
    assert continuity.wants_carryover("what about this one?", "Acme Bank")
    assert not continuity.wants_carryover("what about this one?", "")


def test_carryover_skips_questions_that_name_their_own_subject():
    assert not continuity.wants_carryover(
        "what about Acme Bank's deposits?", "Acme Bank")


def test_expand_attaches_the_remembered_subject():
    out = continuity.expand_followup(
        "and their interest rate?", "total deposits for 2024?", "Acme Bank")
    assert "Acme Bank" in out and "interest rate" in out