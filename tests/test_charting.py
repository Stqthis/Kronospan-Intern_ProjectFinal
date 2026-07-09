"""Charting logic is pure and backend-agnostic, so it is tested here with the
Agg backend (no Qt). Covers table->DataFrame coercion, role inference, default
spec selection, and that each chart type renders to a non-empty PNG without
raising on real-shaped data.
"""
from __future__ import annotations

import io
import os

import pandas as pd
import pytest

from jarvisman.ui import charting as ch


def _render_png(df, spec):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(5, 3))
    FigureCanvasAgg(fig)
    ch.render(fig, df, spec)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Data preparation                                                           #
# --------------------------------------------------------------------------- #
def test_frame_from_table_types():
    headers = ["COMPANY", "AMOUNT EURO"]
    rows = [["Lignum", "150"], ["Kronospan", "200"]]
    df = ch.frame_from_table(headers, rows)
    assert list(df.columns) == headers
    assert pd.api.types.is_numeric_dtype(df["AMOUNT EURO"])
    assert not pd.api.types.is_numeric_dtype(df["COMPANY"])
    assert df["AMOUNT EURO"].tolist() == [150.0, 200.0]


def test_frame_from_table_us_separators(monkeypatch):
    monkeypatch.setenv("RAG_NUM_FORMAT", "us")
    import jarvisman.runtime.numfmt as nf
    monkeypatch.setattr(nf, "_cached", "us", raising=False)
    df = ch.frame_from_table(["X"], [["1,234.50"], ["2,000.00"]])
    assert pytest.approx(df["X"].iloc[0]) == 1234.5


def test_codes_stay_text():
    df = ch.frame_from_table(["CODE", "V"], [["A1", "10"], ["B2", "20"]])
    assert not pd.api.types.is_numeric_dtype(df["CODE"])
    assert pd.api.types.is_numeric_dtype(df["V"])


def test_infer_roles_and_default_spec():
    df = ch.frame_from_table(["COMPANY", "AMOUNT EURO"],
                             [["A", "10"], ["B", "20"]])
    cats, nums = ch.infer_roles(df)
    assert cats == ["COMPANY"] and nums == ["AMOUNT EURO"]
    spec = ch.default_spec(df)
    assert spec.x == "COMPANY" and spec.y == ["AMOUNT EURO"]
    assert spec.chart_type == "bar"


# --------------------------------------------------------------------------- #
# Rendering smoke tests (every type produces a non-empty image)              #
# --------------------------------------------------------------------------- #
@pytest.fixture
def cat_df():
    return ch.frame_from_table(
        ["COMPANY", "AMOUNT EURO"],
        [["Lignum", "150"], ["Kronospan", "200"], ["Falco", "75"],
         ["Acme", "300"]],
    )


@pytest.mark.parametrize("ctype", ["bar", "barh", "line", "area", "pie"])
def test_each_chart_type_renders(cat_df, ctype):
    spec = ch.default_spec(cat_df)
    spec.chart_type = ctype
    png = _render_png(cat_df, spec)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"   # valid PNG header
    assert len(png) > 500


def test_grouped_bars_with_hue():
    df = ch.frame_from_table(
        ["COMPANY", "CURRENCY", "AMOUNT EURO"],
        [["Lignum", "EUR", "100"], ["Lignum", "USD", "50"],
         ["Kronospan", "EUR", "200"], ["Kronospan", "USD", "30"]],
    )
    spec = ch.ChartSpec(chart_type="bar", x="COMPANY", y=["AMOUNT EURO"],
                        hue="CURRENCY")
    png = _render_png(df, spec)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_top_n_folds_into_other():
    rows = [[f"C{i}", str(i + 1)] for i in range(30)]
    df = ch.frame_from_table(["NAME", "V"], rows)
    spec = ch.default_spec(df, top_n=5)
    png = _render_png(df, spec)          # must not raise on >top_n categories
    assert len(png) > 500


def test_render_never_raises_on_empty():
    df = pd.DataFrame()
    png = _render_png(df, ch.ChartSpec())
    assert png[:8] == b"\x89PNG\r\n\x1a\n"   # drew the fallback message, no crash
