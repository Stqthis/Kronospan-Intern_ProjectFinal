"""Shared fixtures for the JarvisMan regression suite.

The fixtures build SMALL DataFrames that mirror the real Kronospan shapes the
app was hardened against -- a MATRIX-style snapshot sheet (many confusable EUR
columns, a Total summary row) and a CY05-style directorships registry
("Surname,First" names). Every deterministic module is exercised against these
instead of against live Excel/Ollama, so the suite runs offline in <1s.

sys.path is prepended with the repo root so `import jarvisman...` works when
pytest is launched from anywhere; the multiprocessing-`spawn` sandbox child
inherits this path, so it can re-import `jarvisman.runtime.sandbox`/`config`.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from jarvisman.semantics.semantic_model import build_semantic_model  # noqa: E402
from jarvisman.semantics.value_index import ValueIndex  # noqa: E402

MATRIX_KEY = "book.xlsx:MATRIX"
CY05_KEY = "book.xlsx:CY05"


@pytest.fixture
def matrix_df():
    """A snapshot sheet: the canonical figure is AMOUNT EURO; Finance/ECCM is a
    confusable decoy column; the last row is a Total that must never be summed
    into a per-company aggregation."""
    return pd.DataFrame(
        {
            "COMPANY": ["Lignum", "Lignum", "Kronospan", "Total"],
            "CURRENCY": ["EUR", "USD", "EUR", ""],
            "AMOUNT EURO": [100.0, 50.0, 200.0, 350.0],
            "FOREIGN CURRENCY": [100.0, 55.0, 200.0, None],
            "Finance/ECCM": [1.0, 2.0, 3.0, None],  # decoy: looks numeric/money
        }
    )


@pytest.fixture
def cy05_df():
    """A directorships registry: names stored 'Surname,First'; a company appears
    once per directorship row (so listings must be made distinct)."""
    return pd.DataFrame(
        {
            "DIRECTOR_NAME": [
                "Spyrou,Spiros",
                "Karanikolaos,Panagiotis",
                "Spyrou,Spiros",
            ],
            "COMPANY_NAME": ["Alpha", "Beta", "Gamma"],
            "COMPANY_CODE": ["A1", "B2", "G3"],
        }
    )


@pytest.fixture
def tables(matrix_df, cy05_df):
    return {MATRIX_KEY: matrix_df, CY05_KEY: cy05_df}


@pytest.fixture
def model(tables):
    return build_semantic_model(tables)


@pytest.fixture
def vindex(tables):
    return ValueIndex.build(tables)
