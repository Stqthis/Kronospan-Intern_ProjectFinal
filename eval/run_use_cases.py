#!/usr/bin/env python3
"""Use-case regression harness.

Validates that the app's OWN ingestion pipeline, run on the real data files,
reproduces the ground-truth answers from the Kronospan use-case document. This
does NOT call the LLM -- it locks down the foundation (correct sheet, correct
columns, correct figures) so a change that breaks ingestion or the data
semantics is caught immediately, offline, in under a second.

Usage:
    python -m eval.run_use_cases --data /path/to/Kronospan/files
    RAG_UCTEST_DATA=/path/to/files python -m eval.run_use_cases

Files are located by filename pattern, so the exact folder layout doesn't
matter. Cases whose source file is absent are reported as SKIPPED, not failed.
"""
from __future__ import annotations

import argparse
import glob
import io
import contextlib
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jarvisman.ingest.ingestion import load_excel  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _find(data_dir: str, *patterns: str):
    for pat in patterns:
        hits = glob.glob(os.path.join(data_dir, "**", pat), recursive=True)
        hits = [h for h in hits if "~$" not in os.path.basename(h)]
        if hits:
            return sorted(hits)[0]
    return None


def _load_one(path: str) -> pd.DataFrame:
    """Ingest a workbook through the real pipeline and return its main table."""
    with contextlib.redirect_stdout(io.StringIO()):
        _, dfs = load_excel(path)
    if not dfs:
        raise RuntimeError(f"no tables ingested from {os.path.basename(path)}")
    # the largest table is the primary data sheet
    return max(dfs.values(), key=lambda d: d.shape[0] * d.shape[1])


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _real_companies(df: pd.DataFrame) -> pd.DataFrame:
    """Drop summary/total rows so aggregations match the business figures."""
    comp = df["COMPANY"].astype(str).str.strip().str.lower()
    return df[~comp.isin(["", "nan", "total", "grand total", "sum"])]


def _close(got, exp, tol) -> bool:
    return got is not None and abs(float(got) - float(exp)) <= tol


# --------------------------------------------------------------------------- #
# individual use cases -> list of (name, ok, detail)                          #
# --------------------------------------------------------------------------- #
def check_cy01(path) -> list:
    df = _real_companies(_load_one(path))
    eur = _num(df["AMOUNT EURO"])
    fx = _num(df["FOREIGN CURRENCY"])
    out = []

    m = df["COMPANY"].astype(str).str.contains("Lignum Technologies AG", case=False, na=False)
    out.append(("1.1 Lignum Technologies AG EUR total", _close(eur[m].sum(), 105247755.84, 0.5),
                f"{eur[m].sum():,.2f} (exp 105,247,755.84)"))

    m = df["COUNTRY"].astype(str).str.contains("Croatia", case=False, na=False)
    out.append(("1.2 Croatia deposits EUR total", _close(eur[m].sum(), 5346044, 1.0),
                f"{eur[m].sum():,.0f} (exp 5,346,044)"))

    m = df["BANK"].astype(str).str.contains("RBI", case=False, na=False)
    out.append(("2.1 RBI bank EUR total", _close(eur[m].sum(), 27010703, 1.0),
                f"{eur[m].sum():,.0f} (exp 27,010,703)"))

    cx = df["BANK"].astype(str).str.contains("CaixaBank", case=False, na=False)
    cur = df["CURRENCY"].astype(str)
    eur_caixa = eur[cx & (cur == "EUR")].sum()
    mad = fx[cx & (cur == "MAD")].sum()
    mxn = fx[cx & (cur == "MXN")].sum()
    ok = _close(eur_caixa, 19455244, 1) and _close(mad, 2158101, 1) and _close(mxn, 1833546, 1)
    out.append(("2.2 CaixaBank per currency (EUR/MAD/MXN)", ok,
                f"EUR {eur_caixa:,.0f} MAD {mad:,.0f} MXN {mxn:,.0f}"))
    return out


def check_cy05(path) -> list:
    df = _load_one(path)
    df.columns = [str(c).strip() for c in df.columns]
    out = []

    k = df[df["DIRECTOR_NAME"].astype(str).str.contains("Koutouvas", case=False, na=False)]
    codes = set(k["COMPANY_CODE"].astype(str).str.strip())
    expected = {"AT01", "CY09", "CY10", "CY14", "CY25", "CY44"}
    out.append(("4.1 Koutouvas directorships (snapshot in file)", codes == expected,
                f"{sorted(codes)}"))

    it = df[df["COMPANY_COUNTRY_NAME"].astype(str).str.strip() == "Italy"]
    by_addr = {}
    for _, r in it[["COMPANY_CODE", "Address"]].drop_duplicates().iterrows():
        by_addr.setdefault(str(r["Address"]).strip(), set()).add(str(r["COMPANY_CODE"]).strip())
    po01_po35_together = any({"PO01", "PO35"} <= codes for codes in by_addr.values())
    gn01_alone = any(codes == {"GN01"} for codes in by_addr.values())
    out.append(("7.1 Italy shared address (PO01&PO35 same, GN01 differs)",
                po01_po35_together and gn01_alone,
                f"groups={[sorted(c) for c in by_addr.values()]}"))
    return out


CASES = [
    ("CY01-DDR (funds)", ("CY01-DDR*DATA*.xlsx", "CY01*DATA*.xlsx"), check_cy01),
    ("CY05 (directorships/address)", ("CY05*DATA*.xlsx", "CY05*Group*Company*.xlsx"), check_cy05),
]


def run(data_dir: str) -> int:
    print(f"Use-case regression against: {data_dir}\n" + "-" * 68)
    total = passed = skipped = 0
    for label, patterns, fn in CASES:
        path = _find(data_dir, *patterns)
        if not path:
            print(f"SKIP  {label}: source file not found ({' / '.join(patterns)})")
            skipped += 1
            continue
        try:
            results = fn(path)
        except Exception as exc:
            print(f"ERROR {label}: {type(exc).__name__}: {exc}")
            total += 1
            continue
        for name, ok, detail in results:
            total += 1
            passed += ok
            print(f"{'PASS' if ok else 'FAIL'}  {name}\n        -> {detail}")
    print("-" * 68)
    print(f"{passed}/{total} checks passed, {skipped} case group(s) skipped")
    return 0 if passed == total and total > 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("RAG_UCTEST_DATA", "./demo_data"))
    args = ap.parse_args()
    if not os.path.isdir(args.data):
        print(f"Data folder not found: {args.data}\n"
              f"Point it at your Kronospan files with --data or RAG_UCTEST_DATA.")
        return 2
    return run(args.data)


if __name__ == "__main__":
    raise SystemExit(main())
