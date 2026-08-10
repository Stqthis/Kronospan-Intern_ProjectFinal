from __future__ import annotations

import os
import re
from typing import Tuple

import pandas as pd

from jarvisman import config as cfg


# --------------------------------------------------------------------------- #
# Loaders                                                                     #
# --------------------------------------------------------------------------- #
def load_pdf(path: str) -> Tuple[list[dict], dict]:
    """PDF -> prose records for retrieval AND ruled tables as DataFrames.

    Tables inside PDFs (e.g. the DDR 'pdf version' reports) are extracted with
    PyMuPDF's table finder and routed through the SAME cleaning pipeline as
    Excel sheets, so numeric questions against a PDF are answered by the table
    reasoner (exact arithmetic over a DataFrame) instead of text retrieval,
    which cannot sum figures scattered across chunks. Continuation pages are
    stitched two ways: a same-width page table whose first row is data (no
    repeated header) extends the previous table; tables whose recovered
    headers are identical are merged across pages. Everything is best-effort:
    a PDF with no detectable tables behaves exactly as before (prose only).
    """
    import fitz  # PyMuPDF; imported lazily to keep import time low

    name = os.path.basename(path)
    records: list[dict] = []
    page_grids: list = []          # [(page_no, raw grid DataFrame), ...]
    doc = fitz.open(path)
    try:
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                records.append({"text": text, "source": name, "location": f"p.{i + 1}"})
            try:
                found = page.find_tables()
                tabs = list(getattr(found, "tables", None) or [])
            except Exception as exc:
                cfg.dbg(f"ingestion.load_pdf find_tables p{i + 1}", exc)
                tabs = []
            for t in tabs:
                try:
                    rows = t.extract()
                except Exception as exc:
                    cfg.dbg(f"ingestion.load_pdf extract p{i + 1}", exc)
                    continue
                rows = [r for r in (rows or []) if any(_cell_filled(v) for v in r)]
                if len(rows) < 2 or max(len(r) for r in rows) < 2:
                    continue  # a 1-row/1-col "table" is layout noise, not data
                page_grids.append((i + 1, pd.DataFrame(rows)))
    finally:
        doc.close()

    if not page_grids:
        return records, {}

    # Pass 1 -- stitch continuation pages. A page grid continues the previous
    # table when it has the same width, does NOT open by repeating the
    # previous header row, and its column TYPE pattern (which columns hold
    # numbers) matches the previous table's data rows. The type signature --
    # not "does the first row look like a header" -- is what correctly keeps a
    # text-heavy data row (Company | Bank | Country | ...) as data.
    def _row_key(row) -> tuple:
        return tuple(str(v).strip().lower() if _cell_filled(v) else ""
                     for v in row)

    def _col_pattern(frame: "pd.DataFrame", skip_first: bool) -> list:
        body = frame.iloc[1:] if skip_first and len(frame) > 1 else frame
        pat = []
        for j in range(frame.shape[1]):
            vals = [v for v in body.iloc[:, j] if _cell_filled(v)]
            pat.append(bool(vals) and
                       sum(1 for v in vals if _numberish(v)) / len(vals) >= 0.5)
        return pat

    logical: list = []             # [width, first_page, last_page, [grids]]
    for pageno, grid in page_grids:
        w = grid.shape[1]
        cont = False
        if logical and logical[-1][0] == w:
            prev0 = logical[-1][3][0]
            header_repeat = _row_key(grid.iloc[0]) == _row_key(prev0.iloc[0])
            if not header_repeat:
                a = _col_pattern(prev0, skip_first=True)
                b = _col_pattern(grid, skip_first=False)
                agree = sum(1 for x, y in zip(a, b) if x == y)
                cont = agree >= 0.8 * w
        if cont:
            logical[-1][3].append(grid)
            logical[-1][2] = pageno
        else:
            logical.append([w, pageno, pageno, [grid]])

    # Pass 2 -- recover headers, clean like Excel, and merge tables whose
    # recovered headers are identical (header repeated on every page).
    dataframes: dict = {}
    by_header: dict = {}           # recovered column tuple -> table key
    for w, p0, p1, grids in logical:
        raw = pd.concat(grids, ignore_index=True) if len(grids) > 1 else grids[0]
        df = _recover_header(raw)
        if df is None or df.empty:
            continue
        df = _strip_text_cells(_clean_columns(df))
        if df.empty or _is_degenerate_table(df):
            continue
        sig = tuple(str(c) for c in df.columns)
        prev = by_header.get(sig)
        if prev is not None:
            dataframes[prev] = pd.concat([dataframes[prev], df],
                                         ignore_index=True)
            continue
        loc = f"p{p0}" if p0 == p1 else f"p{p0}-{p1}"
        key, n = f"{name}:{loc}", 2
        while key in dataframes:
            key = f"{name}:{loc}#{n}"
            n += 1
        by_header[sig] = key
        dataframes[key] = df
    # Coerce numbers AFTER all merging, so a short continuation page (too few
    # values to satisfy the coercion guards on its own) is typed together with
    # the rest of its table instead of leaving a mixed str/float column.
    for key in list(dataframes):
        dataframes[key] = _coerce_us_numbers(_apply_eu_numbers(dataframes[key]))
    # Quality gate: complex report layouts (multi-line headers, column spans)
    # can defeat the geometric table finder and come out under-segmented --
    # several figures glued into one cell. Feeding that to the reasoner would
    # produce WRONG answers, which is worse than none: reject it and keep the
    # page text searchable instead. The clean DATA workbooks remain the
    # authoritative source for those reports.
    rejected = 0
    for key in list(dataframes):
        if not _pdf_table_quality_ok(dataframes[key]):
            del dataframes[key]
            rejected += 1
    if dataframes:
        print(f"\n\U0001F4C4 {name}: extracted {len(dataframes)} "
              f"table(s) from the PDF")
    if rejected:
        print(f"\U0001F4C4 {name}: skipped {rejected} malformed table "
              f"extraction(s); those pages stay text-searchable")
    return records, dataframes


_MAX_HEADER_SCAN = 15  # how many top rows to consider when locating the header

# two or more separate number groups inside ONE cell = column under-segmentation
_MULTINUM_RE = None  # compiled below, after `re` usage is established


def _pdf_table_quality_ok(df) -> bool:
    """True only for a well-formed PDF table extraction.

    Rejects (a) headers that are mostly numbers (data mistaken for a header),
    and (b) tables where a meaningful share of body cells contain SEVERAL
    number groups glued together -- the signature of a geometric extraction
    that merged visual columns. Conservative: anything passing still goes
    through the normal degeneracy checks."""
    global _MULTINUM_RE
    import re as _re
    if _MULTINUM_RE is None:
        _MULTINUM_RE = _re.compile(r"\d[\d,.]*\s+[\d(]")
    try:
        cols = [str(c) for c in df.columns]
        if not cols:
            return False
        if sum(1 for c in cols if _numberish(c)) / len(cols) > 0.3:
            return False
        total = multi = 0
        sample = df.head(200)
        for col in sample.columns:
            for v in sample[col]:
                if not isinstance(v, str):
                    continue
                s = v.strip()
                if not s:
                    continue
                total += 1
                if _MULTINUM_RE.search(s) and \
                        sum(ch.isdigit() for ch in s) >= 6:
                    multi += 1
        if total and multi / total > 0.15:
            return False
        return True
    except Exception:
        return True


_YEAR_CELL_RE = None  # set below (re already imported at module top)


def _cell_filled(v) -> bool:
    """A cell counts as filled if it is not NaN/None and not an empty string."""
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return False
    try:
        return not pd.isna(v)
    except (TypeError, ValueError):
        return True


def _is_year_row(row) -> bool:
    """A sparse row whose filled cells are (almost all) 4-digit years -- the
    classic 'year headers placed rows above the data' layout. Such a row is a
    valid parent header even when it sits 2-3 rows above the sub-header."""
    import re as _re
    vals = [str(v).strip() for v in row if _cell_filled(v)]
    if len(vals) < 2:
        return False
    yearish = sum(1 for v in vals
                  if _re.fullmatch(r"(19|20)\d{2}(\.0)?", v))
    return yearish / len(vals) >= 0.8


def _is_numeric_row(row) -> bool:
    """A sparse row whose filled cells are mostly numbers (and not years) is
    DATA -- typically a grand-total line -- never a parent header. Without
    this guard, a totals row above the header gets flattened into column
    names ('307518243.8 TOTAL CO. B/CE': real-usage bug)."""
    vals = [v for v in row if _cell_filled(v)]
    if not vals:
        return False
    num = sum(1 for v in vals if _numberish(v))
    return num / len(vals) >= 0.5


def _find_parent_row(raw, header_i: int, anchor_filled: int):
    """Look up to 3 rows above the detected header for a merged parent row OR
    a sparse year row. Returns the row index or None. Searching beyond the
    immediately-previous row is what fixes year headers separated from the
    sub-header by a blank/title row."""
    for j in range(header_i - 1, max(-1, header_i - 4), -1):
        row = raw.iloc[j]
        if not any(_cell_filled(v) for v in row):
            continue  # blank spacer row between parent and sub-header
        if _is_year_row(row):
            return j           # sparse year row: the one numeric parent we accept
        if _is_numeric_row(row):
            return None        # totals/data row above the header: NOT a parent
        if _looks_like_parent_header(row, anchor_filled):
            return j
        return None  # first non-blank row above is something else: stop
    return None


def _looks_like_parent_header(row, anchor_filled: int) -> bool:
    """True if ``row`` (directly above the chosen header row) is a MERGED parent
    header rather than a title/subtitle: 2+ cells, fewer than the anchor row
    (the gaps left by horizontal merges), and at least one label that starts
    mid-row (an empty cell immediately to its left). This accepts
    ``_ | 2022 | _ | 2023`` while rejecting a left-packed title like
    ``Report | 2023``."""
    flags = [_cell_filled(v) for v in row]
    p_filled = sum(flags)
    if not (2 <= p_filled < anchor_filled):
        return False
    return any(flags[j] and not flags[j - 1] for j in range(1, len(flags)))


def _flatten_two_row_header(top_vals, bottom_vals, ffill: bool = True) -> list:
    """Combine a parent row with the sub-header row into flat names, e.g.
    parent ``[2022, _, 2023, _]`` + sub ``[Company, Rev, Cost, Rev, Cost]`` ->
    ``[Company, 2022 Rev, 2022 Cost, 2023 Rev, 2023 Cost]``.

    ``ffill`` forward-fills parent labels across gaps -- correct for sparse
    YEAR rows (the gap IS the span) but wrong for merged parents, whose true
    span was already materialised from openpyxl merge metadata: filling past
    it glued 'FUNDS' onto the neighbouring 'TOTAL CO. B/CE' (red-team find)."""
    filled_top, last = [], ""
    for v in top_vals:
        s = str(v).strip() if _cell_filled(v) else ""
        if s:
            last = s
        elif not ffill:
            last = ""
        filled_top.append(last)
    names = []
    for idx, (t, b) in enumerate(zip(filled_top, bottom_vals)):
        bs = str(b).strip() if _cell_filled(b) else ""
        name = f"{t} {bs}" if (t and bs and t != bs) else (bs or t)
        names.append(name or f"col_{idx}")
    return names


def _recover_header(raw: "pd.DataFrame") -> "pd.DataFrame":
    """Locate the real header in a sheet read with ``header=None`` and rebuild
    the table. Handles leading title/blank rows and single- OR two-row (merged)
    headers. A clean sheet whose header is already the first row comes out
    unchanged."""
    if raw is None or raw.empty:
        return raw if raw is not None else pd.DataFrame()

    n_scan = min(len(raw), _MAX_HEADER_SCAN)
    fills = [sum(1 for v in raw.iloc[i] if _cell_filled(v)) for i in range(n_scan)]
    width = max(fills) if fills else 0
    if width == 0:
        return pd.DataFrame()

    # The header is the first "full" row (>= 60% of the table's true width).
    # Scoring by fullness -- not by how much TEXT a row has -- skips leading
    # title/blank rows WITHOUT breaking a header row that happens to be numeric
    # (e.g. a row of years).
    header_i = 0
    fallback_i = None
    for i in range(n_scan):
        if fills[i] >= 0.6 * width:
            # a grand-total line can be the first "full" row, but a header is
            # text: skip numeric-dominant candidates (a sparse YEAR row never
            # reaches the fullness gate, so the year layout is unaffected)
            if _is_numeric_row(raw.iloc[i]):
                if fallback_i is None:
                    fallback_i = i
                continue
            header_i = i
            break
    else:
        header_i = fallback_i if fallback_i is not None else 0

    # A merged parent row that openpyxl filling made "full" (e.g.
    # [_, 2024, 2024]) can win the fullness gate over the REAL header below
    # it. Signature: duplicate runs in the chosen row + an at-least-as-full
    # row right after it -> demote the chosen row to parent.
    if header_i + 1 < n_scan and fills[header_i + 1] >= fills[header_i]:
        vals = [str(v).strip() if _cell_filled(v) else None
                for v in raw.iloc[header_i]]
        filled_vals = [v for v in vals if v is not None]
        has_run = any(vals[j] is not None and vals[j] == vals[j + 1]
                      for j in range(len(vals) - 1))
        if has_run and len(set(filled_vals)) < len(filled_vals):
            header_i += 1

    parent_i = _find_parent_row(raw, header_i, fills[header_i]) if header_i > 0 else None
    if parent_i is not None:
        header = _flatten_two_row_header(
            raw.iloc[parent_i].tolist(), raw.iloc[header_i].tolist(),
            ffill=_is_year_row(raw.iloc[parent_i]),
        )
    else:
        header = []
        for idx, v in enumerate(raw.iloc[header_i].tolist()):
            s = str(v).strip() if _cell_filled(v) else ""
            header.append(s or f"col_{idx}")

    body = raw.iloc[header_i + 1:].copy()
    body.columns = header
    body = body.dropna(axis=1, how="all").dropna(axis=0, how="all").reset_index(drop=True)
    # NOTE: column dtypes (text / number / date + date format) are decided by
    # the model from sample rows in column_types.infer_and_apply_types, called
    # during indexing. _recover_header only finds the header and structures the
    # table; it deliberately does not guess types here.
    return body


def _clean_columns(df: "pd.DataFrame") -> "pd.DataFrame":
    df = df.copy()
    cleaned = []
    seen: dict[str, int] = {}
    for c in df.columns:
        name = str(c)
        name = name.replace("'", "").replace('"', "").replace("`", "")  # quotes break codegen
        name = re.sub(r"\s+", " ", name).strip()
        if name in seen:                       # make duplicates unique (BANK GROUPS, BANK GROUPS.1)
            seen[name] += 1
            name = f"{name}.{seen[name]}"
        else:
            seen[name] = 0
        cleaned.append(name)
    df.columns = cleaned
    return df


def _numberish(v) -> bool:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return True
    try:
        float(str(v).replace(",", "."))
        return True
    except (TypeError, ValueError):
        return False


def _header_like_row(row) -> bool:
    """A plausible header: 2+ filled cells, mostly non-numeric text."""
    vals = [v for v in row if _cell_filled(v)]
    if len(vals) < 2:
        return False
    texty = sum(1 for v in vals if not _numberish(v))
    return texty / len(vals) >= 0.6


def _split_regions(raw: "pd.DataFrame") -> list:
    """Split a raw sheet into vertical regions separated by runs of >= 2
    fully-blank rows -- the common 'several tables stacked in one sheet'
    layout. Single blank rows are kept (they are often visual spacing inside
    one table). Returns [raw] unchanged when there is nothing to split."""
    if raw is None or raw.empty or not cfg.MULTI_TABLE_SHEETS:
        return [raw]
    blank = [not any(_cell_filled(v) for v in raw.iloc[i]) for i in range(len(raw))]
    regions, start, blanks = [], None, 0
    for i, b in enumerate(blank):
        if b:
            blanks += 1
            if blanks >= 2 and start is not None:
                regions.append((start, i - blanks + 1))
                start = None
        else:
            if start is None:
                start = i
            blanks = 0
    if start is not None:
        regions.append((start, len(raw)))
    regions = [(a, b) for a, b in regions if b - a >= 2]
    if len(regions) <= 1:
        return [raw]
    # Evidence guard: every region AFTER the first must open with its own
    # header-like row (mostly text). Blank gaps inside one table -- section
    # spacing, subtotal gaps -- continue with DATA rows, so any non-header
    # continuation cancels the whole split: one sheet stays one table.
    for a, b in regions[1:]:
        if not _header_like_row(raw.iloc[a]):
            return [raw]
    return [raw.iloc[a:b].reset_index(drop=True) for a, b in regions]


def _merged_ranges_from_zip(path: str) -> dict:
    """Read merged-cell ranges straight from the .xlsx XML: {sheet_title:
    [(r0, c0, r1, c1), ...]} with 0-based inclusive coordinates. Milliseconds
    even on very large workbooks -- this replaced a full openpyxl re-load that
    cost ~30s on the 77k-row loan schedule while only the merge RANGES were
    needed (the top-left VALUES are already in the pandas grid)."""
    import zipfile
    import xml.etree.ElementTree as ET

    def _ref_to_rc(ref: str):
        m = re.fullmatch(r"([A-Z]+)(\d+)", ref)
        if not m:
            return None
        col = 0
        for ch in m.group(1):
            col = col * 26 + (ord(ch) - 64)
        return int(m.group(2)) - 1, col - 1

    out: dict = {}
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
          "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
    with zipfile.ZipFile(path) as z:
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        rid_to_target = {
            rel.get("Id"): rel.get("Target")
            for rel in rels
            if (rel.get("Type") or "").endswith("/worksheet")
        }
        for sh in wb.find("m:sheets", ns) or []:
            title = sh.get("name")
            rid = sh.get("{http://schemas.openxmlformats.org/officeDocument"
                         "/2006/relationships}id")
            target = rid_to_target.get(rid)
            if not target:
                continue
            target = target.lstrip("/")
            if not target.startswith("xl/"):
                target = "xl/" + target
            try:
                data = z.read(target)
            except KeyError:
                continue
            # Do NOT parse the whole worksheet XML (sheetData for a 109k-row
            # sheet costs ~9s in ElementTree); scan the bytes for the small
            # <mergeCells> section directly. Sheets without merges cost one
            # substring check.
            a = data.find(b"<mergeCells")
            if a == -1:
                continue
            b = data.find(b"</mergeCells>", a)
            seg = data[a:(b + 13) if b != -1 else min(len(data), a + 65536)]
            ranges = []
            for m in re.finditer(rb'ref="([A-Z]+\d+):([A-Z]+\d+)"', seg):
                p0 = _ref_to_rc(m.group(1).decode())
                p1 = _ref_to_rc(m.group(2).decode())
                if p0 and p1:
                    ranges.append((p0[0], p0[1], p1[0], p1[1]))
            if ranges:
                out[title] = ranges
    return out


def _fill_merged_headers(path: str, raw_sheets: dict) -> None:
    """Forward-fill the top-left value of every merged cell across its span
    (header zone only) so multi-row merged headers flatten exactly instead of
    by inference. The ranges come from the workbook XML (fast); the values
    come from the already-loaded pandas grid, so the workbook is never opened
    a second time. Best-effort: any failure leaves the pandas-only behaviour
    untouched."""
    try:
        merged = _merged_ranges_from_zip(path)
    except Exception as exc:
        cfg.dbg(f"ingestion._fill_merged_headers ranges [{path}]", exc)
        return
    try:
        for title, ranges in merged.items():
            raw = raw_sheets.get(title)
            if raw is None or raw.empty:
                continue
            for r0, c0, r1, c1 in ranges:
                if r0 >= min(len(raw), _MAX_HEADER_SCAN):
                    continue  # merges deep in the data zone are not headers
                if r0 >= len(raw) or c0 >= raw.shape[1]:
                    continue
                val = raw.iat[r0, c0]
                if not _cell_filled(val):
                    continue
                for r in range(r0, min(r1 + 1, len(raw))):
                    for c in range(c0, min(c1 + 1, raw.shape[1])):
                        if not _cell_filled(raw.iat[r, c]):
                            try:
                                raw.iat[r, c] = val
                            except (TypeError, ValueError):
                                # pandas-3 typed column (e.g. str) rejecting a
                                # cross-dtype fill: relax to object and retry
                                col = raw.columns[c]
                                raw[col] = raw[col].astype(object)
                                raw.iat[r, c] = val
    except Exception as exc:
        cfg.dbg("ingestion._fill_merged_headers fill", exc)


_EU_NUM_RE = re.compile(r"^-?\d{1,3}(\.\d{3})+(,\d+)?$|^-?\d+,\d+$")

# --------------------------------------------------------------------------- #
# US / EU disambiguation                                                       #
# --------------------------------------------------------------------------- #
# The two number styles collide on one shape: a lone 3-digit comma group.
# '19,999' is nineteen THOUSAND in US style but nineteen-point-999 in EU style,
# and '250,000' likewise. The old code let _EU_NUM_RE ('^-?\d+,\d+$') match that
# shape, so US thousands columns were claimed by the EU pass and divided by 1000
# ('480,500' -> 480.50), while columns that also held a true millions value
# ('19,999,999', two comma groups) matched NEITHER pass and were left as unparsed
# strings. Resolve it once, per column, from the values only one style can produce.
#
#   US-STRONG: a comma used as a thousands separator in a way EU never writes --
#              two or more 3-digit groups ('19,999,999') or a group plus a dot
#              decimal ('1,234.50'); also a plain dot decimal ('1234.56').
#   EU-STRONG: dot thousands ('1.234.567' / '1.234,56') or a comma decimal with
#              1-2 digits ('12,5' / '0,75') -- US never puts 1-2 digits after a
#              comma.
#   AMBIGUOUS: exactly one 3-digit comma group and nothing else ('19,999').
_US_STRONG_RE = re.compile(
    r"^-?\(?\$?\s?\d{1,3}(,\d{3}){2,}(\.\d+)?\)?$"   # 2+ groups: 19,999,999
    r"|^-?\(?\$?\s?\d{1,3}(,\d{3})+\.\d+\)?$"         # group + decimal: 1,234.50
    r"|^-?\d+\.\d+$")                                 # plain decimal: 1234.56
_EU_STRONG_RE = re.compile(
    r"^-?\d{1,3}(\.\d{3})+(,\d+)?$"                   # dot thousands: 1.234.567(,89)
    r"|^-?\d+,\d{1,2}$")                              # comma + 1-2 dp: 12,5 / 0,75
_AMBIG_NUM_RE = re.compile(r"^-?\d{1,3},\d{3}$")      # lone 3-digit group: 19,999
# a lone dotted 3-digit group ('1.234') is three-way ambiguous -- US decimal,
# EU thousands, OR an internal code -- so it is NOT strong US evidence; the
# codes-stay-text contract owns it (see _apply_eu_numbers' comma-required guard).
_AMBIG_DOT_RE = re.compile(r"^-?\d{1,3}\.\d{3}$")


def _column_number_locale(vals) -> "str | None":
    """Decide whether a string column is US-formatted, EU-formatted, or neither,
    from the whole column's evidence. A single ambiguous '19,999'-style token is
    read as US thousands (3-decimal-place money is vanishingly rare, and these
    columns are overwhelmingly thousands); a lone dotted '1.234' is left alone as
    a possible code; a column that shows both styles returns None so both passes
    leave it untouched."""
    us = eu = ambig_comma = ambig_dot = 0
    for v in vals:
        if _AMBIG_DOT_RE.fullmatch(v):
            ambig_dot += 1
        elif _US_STRONG_RE.fullmatch(v):
            us += 1
        elif _EU_STRONG_RE.fullmatch(v):
            eu += 1
        elif _AMBIG_NUM_RE.fullmatch(v):
            ambig_comma += 1
    if us and eu:
        return None                       # contradictory evidence -- don't guess
    if us:
        return "us"
    if eu:
        return "eu"
    if ambig_comma and not ambig_dot:
        return "us"                       # lone 3-digit comma groups -> thousands
    return None                           # dotted codes / no clear style


def _apply_eu_numbers(df: "pd.DataFrame") -> "pd.DataFrame":
    """Convert columns stored in European locale format ('1.234,56') to real
    numbers. Conservative: >= 80% of non-null values must FULLY match the EU
    pattern, and the conversion must not lose values, otherwise the column is
    left untouched."""
    if not cfg.EU_NUMBER_PARSE or df is None or df.empty:
        return df
    for col in df.columns:
        s = df[col]
        if getattr(s.dtype, "kind", "O") in "iufcMmb" or hasattr(s, "columns"):
            continue
        vals = s.dropna().astype(str).str.strip()
        if len(vals) < 2:
            continue
        # only claim columns the whole-column vote reads as EU; a US thousands
        # column ('19,999', '480,500') is left for _coerce_us_numbers instead of
        # being divided by 1000 here
        if _column_number_locale(vals) != "eu":
            continue
        rate = float(vals.map(lambda v: bool(_EU_NUM_RE.fullmatch(v))).mean())
        if rate < 0.8:
            continue
        # comma evidence required: a column of dotted CODES ('1.234') matches
        # the thousands pattern but is not money -- without a single decimal
        # comma anywhere, leave it alone (red-team find)
        if not vals.str.contains(",", regex=False).any():
            continue
        conv = pd.to_numeric(
            vals.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
            errors="coerce",
        )
        if conv.notna().sum() < len(vals) * 0.95:
            continue
        out = pd.to_numeric(
            s.astype(str).str.strip()
             .str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
            errors="coerce",
        )
        df[col] = out
    return df


_US_NUM_RE = re.compile(r"^-?\(?\$?\s?\d{1,3}(,\d{3})*(\.\d+)?\)?$"
                        r"|^-?\d+(\.\d+)?$")


def _coerce_us_numbers(df: "pd.DataFrame") -> "pd.DataFrame":
    """Convert string columns that are consistently US/accounting-formatted
    numbers ('105,247,755.84', '(1,234.50)') to real numbers.

    Needed for PDF-extracted tables, where EVERY cell arrives as a string;
    Excel numeric columns are already numeric dtype and pass through
    untouched. Conservative, mirroring _apply_eu_numbers: >= 80% of non-null
    values must fully match, conversion must keep >= 95% of them, and any
    column with EU-style values is left for the EU pass."""
    if df is None or df.empty:
        return df
    for col in df.columns:
        s = df[col]
        if getattr(s.dtype, "kind", "O") in "iufcMmb" or hasattr(s, "columns"):
            continue
        vals = s.dropna().astype(str).str.strip()
        vals = vals[vals != ""]
        if len(vals) < 2:
            continue
        # claim the column iff the whole-column vote reads it as US; this is the
        # same decision _apply_eu_numbers used to skip, so exactly one pass owns
        # any column and a lone '19,999,999' no longer falls through both
        if _column_number_locale(vals) != "us":
            continue
        rate = float(vals.map(lambda v: bool(_US_NUM_RE.fullmatch(v))).mean())
        if rate < 0.8:
            continue
        # comma or decimal evidence required: a column of bare integer CODES
        # ('1024', '2048') stays text -- exact-match filters must keep working
        if not (vals.str.contains(",", regex=False).any()
                or vals.str.contains(".", regex=False).any()):
            continue

        def _conv(v):
            t = str(v).strip().replace("$", "").replace(" ", "")
            neg = t.startswith("(") and t.endswith(")")
            t = t.strip("()").replace(",", "")
            try:
                x = float(t)
            except ValueError:
                return None
            return -x if neg else x

        conv = vals.map(_conv)
        if conv.notna().sum() < len(vals) * 0.95:
            continue
        df[col] = s.map(lambda v: _conv(v)
                        if isinstance(v, str) and v.strip() else v)
    return df


def _strip_text_cells(df: "pd.DataFrame") -> "pd.DataFrame":
    """Strip leading/trailing whitespace from string CELLS once, at load
    time. A stored 'Karanikolaos,Panagiotis ' (trailing space) breaks any
    downstream exact comparison the model writes; stripping at the source
    removes the whole failure class instead of defending against it in four
    places. Only str instances are touched -- dates/numbers kept as objects
    pass through unchanged."""
    for c in df.columns:
        s = df[c]
        if hasattr(s, "columns") or getattr(s.dtype, "kind", "O") != "O":
            continue
        try:
            df[c] = s.map(lambda v: v.strip() if isinstance(v, str) else v)
        except (TypeError, ValueError, AttributeError) as exc:
            cfg.dbg(f"ingestion._strip_text_cells [{c}]", exc)
    return df


def _skip_sheet(name: str) -> bool:
    """True for sheets that are derived views / helpers, not primary data."""
    n = str(name).strip().lower()
    if not n:
        return True
    if n in getattr(cfg, "SKIP_SHEET_NAMES", []):
        return True
    if any(n.startswith(p) for p in getattr(cfg, "SKIP_SHEET_PREFIXES", [])):
        return True
    if any(s in n for s in getattr(cfg, "SKIP_SHEET_SUBSTRINGS", [])):
        return True
    return False


def _is_degenerate_table(df) -> bool:
    """A recovered table with almost no distinct column names is a pivot dump
    (e.g. columns 'TRADING COMPANIES', 'TRADING COMPANIES.1', ...), not data.
    Runs on the RECOVERED header, where such sheets collapse to 1-3 bases."""
    try:
        cols = [str(c) for c in df.columns]
    except Exception:
        return False
    width = len(cols)
    if width == 0:
        return True
    bases, placeholders = set(), 0
    for c in cols:
        b = re.sub(r"\.\d+$", "", c).strip()
        if (not b) or re.fullmatch(r"col_\d+", b) or b.lower().startswith("unnamed"):
            placeholders += 1
            continue
        bases.add(b.lower())
    if placeholders >= 0.6 * width:
        return True
    if (width >= getattr(cfg, "DEGENERATE_MIN_WIDTH", 8)
            and len(bases) <= getattr(cfg, "DEGENERATE_MAX_DISTINCT", 3)):
        return True
    return False


def load_excel(path: str) -> Tuple[list[dict], dict]:
    """Excel -> {'file:sheet': DataFrame}; with debugging and engine handling."""
    name = os.path.basename(path)
    print(f"\n📄 Loading Excel: {name}")
    
    # High row ceiling (None = no limit): the loan schedules have 78k-109k rows
    # and an "as at <date>" balance needs all of them. Junk sheets are skipped
    # up front so this stays fast despite the larger cap.
    MAX_ROWS = getattr(cfg, "EXCEL_MAX_ROWS", 250000)

    # Engine: prefer 'calamine' (Rust reader; ~4x faster on large sheets and
    # cell-identical to openpyxl on the real Kronospan files). Fall back to
    # openpyxl/xlrd when python-calamine is not installed.
    engine = None
    try:
        import python_calamine  # noqa: F401
        engine = 'calamine'
    except ImportError:
        if path.lower().endswith(('.xlsx', '.xlsm')):
            engine = 'openpyxl'
        elif path.lower().endswith('.xls'):
            engine = 'xlrd'

    # Only read the sheets that hold primary data.
    try:
        with pd.ExcelFile(path, engine=engine) as _xl:
            all_names = list(_xl.sheet_names)
    except Exception as e:
        print(f"  ❌ Error reading file: {e}")
        return [], {}
    keep = [s for s in all_names if not _skip_sheet(s)] or all_names
    dropped = [s for s in all_names if s not in keep]
    if dropped:
        print(f"  Skipping non-data sheets: {dropped}")

    try:
        raw_sheets = pd.read_excel(
            path, sheet_name=keep, header=None, engine=engine, nrows=MAX_ROWS)
        if isinstance(raw_sheets, pd.DataFrame):   # single sheet -> wrap
            raw_sheets = {keep[0]: raw_sheets}
        print(f"  Sheets read: {list(raw_sheets.keys())}")
    except Exception as e:
        print(f"  ❌ Error reading file: {e}")
        return [], {}
    
    if path.lower().endswith((".xlsx", ".xlsm")):
        _fill_merged_headers(path, raw_sheets)
    
    dataframes: dict[str, pd.DataFrame] = {}
    
    for sheet, raw in raw_sheets.items():
        print(f"\n  Sheet: {sheet}")
        print(f"    Raw shape: {raw.shape}")
        
        regions = _split_regions(raw)
        print(f"    Regions found: {len(regions)}")
        
        for ri, region in enumerate(regions):
            print(f"      Region {ri}: shape {region.shape}")
            
            df = _recover_header(region)
            if df is None:
                print(f"        ❌ recover_header returned None")
                continue
            if df.empty:
                print(f"        ❌ DataFrame empty after recover_header")
                continue
            
            print(f"        After recover_header: {df.shape}")
            
            df = _apply_eu_numbers(_strip_text_cells(_clean_columns(df)))
            print(f"        After cleaning: {df.shape}")
            
            if df.empty:
                print(f"        ❌ DataFrame empty after cleaning")
                continue

            if _is_degenerate_table(df):
                print(f"        ⤳ skipped: no clear data columns (pivot/derived)")
                continue

            key = f"{name}:{sheet}" if ri == 0 else f"{name}:{sheet}#{ri + 1}"
            dataframes[key] = df
            print(f"        ✓ Added: {key}")
    
    print(f"\n  Total dataframes: {len(dataframes)}\n")
    return [], dataframes

def ingest_file(path: str) -> Tuple[list[dict], dict]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf(path)
    if ext in (".xlsx", ".xls"):
        return load_excel(path)
    raise ValueError(f"Unsupported file type: {ext}")


# --------------------------------------------------------------------------- #
# Chunking (PDF text only)                                                    #
# --------------------------------------------------------------------------- #
def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Paragraph-aware character chunking with overlap.

    Keeps paragraphs together where possible; hard-splits paragraphs that
    exceed ``size``; then stitches a tail of the previous chunk onto each
    successive chunk so context is not lost across boundaries.
    """
    text = (text or "").strip()
    if not text:
        return []

    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    base: list[str] = []
    cur = ""
    for p in paras:
        if len(cur) + len(p) + 1 <= size:
            cur = (cur + "\n" + p).strip()
        else:
            if cur:
                base.append(cur)
            if len(p) <= size:
                cur = p
            else:
                step = max(1, size - overlap)
                for i in range(0, len(p), step):
                    base.append(p[i : i + size])
                cur = ""
    if cur:
        base.append(cur)

    if overlap <= 0 or len(base) <= 1:
        return base

    stitched = [base[0]]
    for i in range(1, len(base)):
        tail = base[i - 1][-overlap:]
        stitched.append((tail + "\n" + base[i]).strip())
    return stitched


def build_chunks(records: list[dict], size: int, overlap: int) -> list[dict]:
    chunks: list[dict] = []
    for r in records:
        for piece in chunk_text(r["text"], size, overlap):
            chunks.append(
                {
                    "text": piece,
                    "source": r["source"],
                    "location": r["location"],
                    "chunk_id": len(chunks),
                }
            )
    return chunks