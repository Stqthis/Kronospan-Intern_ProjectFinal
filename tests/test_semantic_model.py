"""The semantic model decides each column's MEANING (number vs date vs
dimension vs empty) and flags Total/Subtotal rows. Those classifications drive
every downstream tier, so they are pinned here against a small mixed-type
sheet.
"""
from __future__ import annotations

import pandas as pd

from jarvisman.semantics.semantic_model import build_semantic_model


def _profile(df, key="t.xlsx:S"):
    model = build_semantic_model({key: df})
    return model.tables[key]


def test_number_date_empty_classification():
    df = pd.DataFrame(
        {
            "AMOUNT EURO": [100.0, 200.0, 300.0],
            "AS_AT": pd.to_datetime(["2023-12-31", "2024-12-31", "2024-06-30"]),
            "CURRENCY": ["EUR", "USD", "EUR"],
            "EMPTY": [None, None, None],
        }
    )
    prof = _profile(df)
    assert prof.column("AMOUNT EURO").semantic_type == "number"
    assert prof.column("AS_AT").semantic_type == "date"
    assert prof.column("CURRENCY").semantic_type == "text"
    assert prof.column("EMPTY").semantic_type == "empty"


def test_year_column_gets_year_role():
    df = pd.DataFrame({"YEAR": [2022, 2023, 2024], "V": [1, 2, 3]})
    prof = _profile(df)
    assert prof.column("YEAR").role == "year"


def test_text_typed_numbers_profile_as_number():
    # a text-typed column that parses numerically above the 90% threshold must
    # be classified as a measure (the "measure-as-text" path)
    df = pd.DataFrame({"AMT_TXT": ["100", "200", "300", "400"]})
    prof = _profile(df)
    assert prof.column("AMT_TXT").semantic_type == "number"


def test_mostly_text_column_stays_text():
    # below the 90% numeric-parse threshold it must NOT be treated as a number
    df = pd.DataFrame({"MIXED": ["100", "200", "300", "oops"]})  # 75% numeric
    prof = _profile(df)
    assert prof.column("MIXED").semantic_type == "text"


def test_summary_row_detected(model):
    prof = model.tables["book.xlsx:MATRIX"]
    # the 'Total' row (index 3) must be flagged so aggregations exclude it
    assert 3 in prof.summary_row_indices


def test_no_false_summary_rows(cy05_df):
    prof = _profile(cy05_df, key="book.xlsx:CY05")
    assert prof.summary_row_indices == []
