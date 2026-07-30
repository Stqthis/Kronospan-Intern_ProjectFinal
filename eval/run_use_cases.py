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


def check_wcr(path) -> list:
    """Use cases 9-10, verified against WCR_27_12_2023.xlsx."""
    df = _load_one(path)
    out = []
    bg = df[df["Company"].astype(str).str.contains("Bulgaria Eood", case=False, na=False)]
    av = _num(bg["Available Facility 000"]).round(0)
    ok = {29337.0, 838.0, 28499.0} <= set(av.dropna())
    out.append(("9.1 Bulgaria facilities '000 (29,337 / 838 / 28,499)", ok,
                f"available'000={sorted(av.dropna().unique())}"))

    teur = _num(bg["Total Funds Available TEUR"]).dropna()
    ok = _close(teur.iloc[0] if len(teur) else None, 18455.84, 1.0)
    out.append(("10.1 Bulgaria total funds available TEUR (18,456)", ok,
                f"{teur.iloc[0]:,.2f}" if len(teur) else "missing"))

    dk = df[df["Company Country"].astype(str).str.strip() == "Denmark"]
    ut = _num(dk["Utilised Facility 000"]).round(0)
    ok = len(dk) == 5 and 3187.0 in set(ut.dropna())
    out.append(("9.2 Denmark rows (5 lines, utilised 3,187 DKK)", ok,
                f"rows={len(dk)}, utilised'000={sorted(ut.dropna().unique())}"))
    return out


def check_ltl(path) -> list:
    """Use case 11, verified against LTL_Data.xlsx (see Demo Data PDF; the
    PDF total '31,499,9125.50' is a typo for 31,499,912.50)."""
    df = _load_one(path)
    out = []
    lender_col = next((c for c in df.columns if "LENDER" in str(c).upper()
                       and "CASEWHEN" in str(c).upper()), None)
    cr = df[(df["NAME"].astype(str).str.strip() == "Kronospan CR, spol s r.o.")
            & (df["FACILITYGROUPING"].astype(str) == "3rd Party")]
    snap = cr[pd.to_datetime(cr["CALC_DATE"], errors="coerce") == "2025-12-31"]
    total = _num(snap["OUTSTANDING_BCE"]).sum()
    out.append(("11.1 CR 3rd-party outstanding @31/12/2025 (31,499,912.50)",
                _close(total, 31499912.50, 1.0), f"{total:,.2f}"))
    if lender_col is not None and len(snap):
        per = snap.groupby(lender_col)["OUTSTANDING_BCE"].sum()
        ok = (_close(per.get("KBC Group"), 10499970.87, 1.0)
              and _close(per.get("Erste Group"), 10499970.82, 1.0)
              and _close(per.get("Societe Generale Group"), 10499970.82, 1.0))
        out.append(("11.1.2 split KBC/Erste/SocGen", ok,
                    ", ".join(f"{k}={v:,.2f}" for k, v in per.items())))
    undrawn = _num(snap["TOBEDRAWN"]).sum()
    out.append(("11.1.3 undrawn = 0", _close(undrawn, 0.0, 0.01),
                f"{undrawn:,.2f}"))
    ends = set(pd.to_datetime(cr["END_DATE"], errors="coerce").dropna()
               .dt.strftime("%Y-%m-%d"))
    out.append(("11.2 end date 30/09/2027", ends == {"2027-09-30"},
                f"{sorted(ends)}"))
    return out


CASES = [
    ("CY01-DDR (funds)", ("CY01-DDR*DATA*.xlsx", "CY01*DATA*.xlsx"), check_cy01),
    ("CY05 (directorships/address)", ("CY05*DATA*.xlsx", "CY05*Group*Company*.xlsx"), check_cy05),
    ("WCR (working capital, 27/12/2023 snapshot)", ("WCR_27_12_2023.xlsx",), check_wcr),
    ("LTL (long-term loans)", ("LTL_Data*.xlsx", "LTL*DATA*.xlsx"), check_ltl),
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
    try:
        from jarvisman.runtime import scoreboard
        scoreboard.record("data_checks", {"passed": int(passed),
                                          "total": int(total),
                                          "skipped": int(skipped)})
    except Exception:
        pass
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
