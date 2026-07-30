"""Filename-vs-content date validation for snapshot workbooks.

The app routes "as at <date>" questions by the date in the FILE NAME, so a
mis-named export silently answers from the wrong period (found in the wild:
WCR_30_07_2024.xlsx containing the 30/12/2024 report and vice versa). This
check reads a small sample of each workbook, finds the dominant report date,
and compares it with the filename.
"""

from __future__ import annotations

import os
import re
from typing import Optional

_FN_DATE = re.compile(r"(\d{1,2})[-_.](\d{1,2})[-_.](\d{2,4})")
_DATE_COL_HINTS = ("report date", "date", "calc_date", "report_date")


def filename_date(name: str) -> Optional[str]:
    """dd.mm.yyyy parsed from the file name, or None."""
    m = _FN_DATE.search(os.path.basename(name))
    if not m:
        return None
    d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if y < 100:
        y += 2000
    if not (1 <= d <= 31 and 1 <= mo <= 12 and 2000 <= y <= 2100):
        return None
    return f"{d:02d}.{mo:02d}.{y}"


def content_date(path: str, nrows: int = 400) -> Optional[str]:
    """The workbook's REPORT date (dd.mm.yyyy), or None.

    Header-independent: reads raw cells (these exports have banner rows
    above the real header) and picks the datetime column whose values are
    the most CONSTANT — a report-date column repeats one value on every
    row, while review/maturity date columns spread across many values."""
    try:
        import warnings
        import pandas as pd
        warnings.filterwarnings(
            "ignore", message="Could not infer format")
        xl = pd.ExcelFile(path)
        best = None          # (mode_share, count, mode_str)
        for sheet in xl.sheet_names[:3]:
            try:
                df = xl.parse(sheet, header=None, nrows=nrows)
            except Exception:
                continue
            for c in list(df.columns)[:40]:
                s = pd.to_datetime(df[c], errors="coerce").dropna()
                s = s[(s.dt.year >= 2000) & (s.dt.year <= 2100)]
                if len(s) < 10:
                    continue
                vc = s.dt.strftime("%d.%m.%Y").value_counts()
                share = float(vc.iloc[0]) / float(len(s))
                cand = (share, int(len(s)), str(vc.index[0]))
                if best is None or cand[:2] > best[:2]:
                    best = cand
        if best is None or best[0] < 0.9:      # no near-constant date column
            return None
        return best[2]
    except Exception:
        return None


def check(path: str) -> Optional[dict]:
    """{'file', 'filename_date', 'content_date', 'mismatch'} or None when
    either date cannot be determined (nothing to compare)."""
    if not str(path).lower().endswith((".xlsx", ".xls")):
        return None
    fn = filename_date(path)
    if not fn:
        return None
    cd = content_date(path)
    if not cd:
        return None
    return {"file": os.path.basename(path), "filename_date": fn,
            "content_date": cd, "mismatch": fn != cd}
