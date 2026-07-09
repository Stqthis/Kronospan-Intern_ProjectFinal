"""Ingestion is the most Excel-messiness-prone layer, and its comments record
several "red-team finds". Those finds are turned into explicit regressions
here: the EU-number guard must not convert dotted CODES, the two-row header
flattener must not forward-fill a merged parent across a gap, and a leading
title row must be dropped without losing a numeric header row.
"""
from __future__ import annotations

import pandas as pd

from jarvisman.ingest import ingestion as ing


# --------------------------------------------------------------------------- #
# EU number parsing                                                          #
# --------------------------------------------------------------------------- #
def test_eu_money_column_is_converted():
    df = pd.DataFrame({"AMT": ["1.234,56", "2.000,00", "3.100,10"]})
    out = ing._apply_eu_numbers(df.copy())
    assert out["AMT"].dtype.kind in "if"
    assert abs(float(out["AMT"].iloc[0]) - 1234.56) < 1e-6
    assert abs(float(out["AMT"].iloc[1]) - 2000.0) < 1e-6


def test_dotted_codes_without_comma_are_left_alone():
    # red-team find: '1.234' matches the thousands pattern but is a CODE, not
    # money. Without a single decimal comma in the column, leave it as text.
    df = pd.DataFrame({"CODE": ["1.234", "5.678", "9.012"]})
    out = ing._apply_eu_numbers(df.copy())
    assert out["CODE"].dtype.kind == "O"
    assert out["CODE"].iloc[0] == "1.234"


def test_eu_parse_skips_when_below_threshold():
    # only 1 of 3 looks European -> conversion must NOT fire (conservative)
    df = pd.DataFrame({"X": ["1.234,56", "hello", "world"]})
    out = ing._apply_eu_numbers(df.copy())
    assert out["X"].dtype.kind == "O"


# --------------------------------------------------------------------------- #
# Two-row header flattening                                                  #
# --------------------------------------------------------------------------- #
def test_year_parent_forward_fills_across_gaps():
    top = [None, 2022, None, 2023, None]
    bottom = ["Company", "Rev", "Cost", "Rev", "Cost"]
    names = ing._flatten_two_row_header(top, bottom, ffill=True)
    assert names == ["Company", "2022 Rev", "2022 Cost", "2023 Rev", "2023 Cost"]


def test_merged_parent_does_not_fill_across_gap():
    # red-team find: with ffill=False the gap is a real boundary -- 'FUNDS'
    # must NOT be glued onto the neighbouring column.
    top = ["FUNDS", None, "TOTAL CO B/CE"]
    bottom = ["a", "b", "c"]
    names = ing._flatten_two_row_header(top, bottom, ffill=False)
    assert names == ["FUNDS a", "b", "TOTAL CO B/CE c"]


# --------------------------------------------------------------------------- #
# Header recovery + column cleaning                                          #
# --------------------------------------------------------------------------- #
def test_recover_header_drops_leading_title_row():
    raw = pd.DataFrame(
        [
            ["Bank funds report", None, None],   # title row
            ["COMPANY", "CURRENCY", "AMOUNT EURO"],  # real header
            ["Lignum", "EUR", 100],
            ["Kronospan", "EUR", 200],
        ]
    )
    out = ing._recover_header(raw)
    assert list(out.columns) == ["COMPANY", "CURRENCY", "AMOUNT EURO"]
    assert len(out) == 2
    assert out.iloc[0]["COMPANY"] == "Lignum"


def test_clean_columns_dedupes_and_strips_quotes():
    df = pd.DataFrame([[1, 2, 3]], columns=["BANK GROUPS", "BANK GROUPS", "A'MT \"E\""])
    out = ing._clean_columns(df)
    cols = list(out.columns)
    assert cols[0] == "BANK GROUPS"
    assert cols[1] == "BANK GROUPS.1"          # duplicate made unique
    assert "'" not in cols[2] and '"' not in cols[2]  # quotes stripped
