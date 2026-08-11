"""Locale-aware number formatting for DISPLAYED results.

Display only: compiled pandas, PROV row counts, chart data and persisted
values all stay raw -- formatting is applied at the single point where a
result becomes text/HTML for the user, plus the two places that must READ
formatted text back (the synthesis echo check and the eval scorer), which use
``canon`` to undo it.

Style resolution (``RAG_NUM_FORMAT``):
  locale (default) -> ask the OS: decimal comma => 'eu', else 'us'
  us               -> 1,234,567.50
  eu               -> 1.234.567,50
  plain            -> str(x), byte-identical to the unformatted output

Integral values render without decimals ("12,345"); non-integral with two.
Integers in 1900-2100 stay plain so years never become "2,023" -- amounts
that happen to fall in that window lose their separator, which is the lesser
error and is documented in DECISIONS.md.
"""

from __future__ import annotations

from jarvisman import config as cfg

_cached: str | None = None


def style() -> str:
    """Resolved style, cached per process (the sandbox child re-imports this
    module and re-reads RAG_NUM_FORMAT from its inherited environment)."""
    global _cached
    s = getattr(cfg, "NUMBER_FORMAT", "locale")
    if s in ("us", "eu", "plain"):
        return s
    if _cached is None:
        try:
            import locale
            locale.setlocale(locale.LC_NUMERIC, "")
            dp = locale.localeconv().get("decimal_point", ".")
            _cached = "eu" if dp == "," else "us"
        except Exception:
            _cached = "us"
    return _cached


def fmt(x) -> str:
    """One value -> display string in the active style."""
    st = style()
    if st == "plain":
        return str(x)
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v or v in (float("inf"), float("-inf")):   # NaN/inf untouched
        return str(x)
    iv = round(v)
    if abs(v - iv) < 1e-9:
        iv = int(iv)
        if 1900 <= iv <= 2100:                          # years stay plain
            return str(iv)
        s = f"{iv:,d}"
    else:
        s = f"{v:,.2f}"
    if st == "eu":
        s = s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return s


def canon(text: str) -> str:
    """Undo the active style so numbers in formatted text parse as floats
    again ('1.234.567,50' -> '1234567.50'). Used by the echo check and the
    eval scorer; harmless on prose."""
    if not text:
        return text or ""
    if style() == "eu":
        return text.replace(".", "\x00").replace(",", ".").replace("\x00", "")
    return text.replace(",", "")


# --------------------------------------------------------------------------- #
# Interest-rate display                                                        #
# --------------------------------------------------------------------------- #
import re as _re

_RATE_COL_RE = _re.compile(r"rate|margin", _re.I)


def is_rate_col(name) -> bool:
    """A column whose values are interest rates / margins (by name)."""
    return bool(_RATE_COL_RE.search(str(name)))


def rate_formatter(values):
    """Build a per-COLUMN formatter that always renders rates as PERCENTAGES
    with three decimals and a % sign (use-case format: 4.236%).

    The data holds rates in two conventions: percent (RATE % = 4.236) and
    fraction (INTEREST RATE = 0.04236). Decided per column: when every
    non-zero value is <= 1 the column is fractional and is scaled x100 for
    display; otherwise values are already percent. Display-only — the
    underlying numbers are never changed."""
    nonzero = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == f and f != 0.0 and abs(f) != float("inf"):
            nonzero.append(abs(f))
    scale = 100.0 if nonzero and max(nonzero) <= 1.0 else 1.0

    def _f(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return str(v)
        if f != f or abs(f) == float("inf"):
            return str(v)
        s = f"{f * scale:.3f}"
        if style() == "eu":
            s = s.replace(".", ",")
        return s + "%"
    return _f


_REFMT_RE = _re.compile(r"(?<![\w.,])(-?\d{4,}(?:\.\d+)?)(?![\d,])(?!\.\d)(?!\w)")


def reformat(text: str) -> str:
    """Deterministically reformat bare numerics inside already-produced text
    (codegen-fallback stdout). Conservative: only standalone numbers with 4+
    integer digits, never tokens adjacent to other digits/punctuation (so
    dates like 31.12.2023, codes like C0011 and decimals already inside
    formatted numbers are untouched), and the 1900-2100 year window stays
    plain, same as fmt()."""
    if not text or style() == "plain":
        return text
    def _sub(m):
        raw = m.group(1)
        try:
            v = float(raw)
        except ValueError:
            return raw
        if "." not in raw and 1900 <= int(v) <= 2100:
            return raw
        return fmt(v)
    return _REFMT_RE.sub(_sub, text)
