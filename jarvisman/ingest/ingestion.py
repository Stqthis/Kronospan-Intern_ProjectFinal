

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
    import fitz  # PyMuPDF; imported lazily to keep import time low

    name = os.path.basename(path)
    records: list[dict] = []
    doc = fitz.open(path)
    try:
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                records.append({"text": text, "source": name, "location": f"p.{i + 1}"})
    finally:
        doc.close()
    return records, {}


_MAX_HEADER_SCAN = 15  # how many top rows to consider when locating the header


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


def _fill_merged_headers(path: str, raw_sheets: dict) -> None:
    """openpyxl metadata pass: forward-fill the top-left value of every merged
    cell across its span (header zone only) so multi-row merged headers
    flatten exactly instead of by inference. Best-effort: any failure leaves
    the pandas-only behaviour untouched."""
    try:
        import openpyxl
        wb = openpyxl.load_workbook(path, data_only=True, read_only=False)
    except Exception as exc:
        # feature-isolation: the whole merged-header pass is optional; any
        # failure (missing dep, bad/locked file) leaves the pandas-only
        # behaviour untouched. Surface it rather than hide it.
        cfg.dbg(f"ingestion._fill_merged_headers load [{path}]", exc)
        return
    try:
        for ws in wb.worksheets:
            raw = raw_sheets.get(ws.title)
            if raw is None or raw.empty:
                continue
            for rng in list(getattr(ws, "merged_cells", []).ranges or []):
                r0, c0 = rng.min_row - 1, rng.min_col - 1
                if r0 >= min(len(raw), _MAX_HEADER_SCAN):
                    continue  # merges deep in the data zone are not headers
                try:
                    val = ws.cell(rng.min_row, rng.min_col).value
                except (IndexError, ValueError, AttributeError) as exc:
                    cfg.dbg("ingestion merged-cell read", exc)
                    continue
                if val is None:
                    continue
                for r in range(r0, min(rng.max_row, len(raw))):
                    for c in range(c0, min(rng.max_col, raw.shape[1])):
                        if not _cell_filled(raw.iat[r, c]):
                            raw.iat[r, c] = val
    except Exception as exc:
        # isolation around the fill loop: a malformed merge map must not abort
        # ingestion; the sheet still loads via the pandas path.
        cfg.dbg("ingestion._fill_merged_headers fill", exc)
    finally:
        try:
            wb.close()
        except Exception:
            pass  # cleanup: a close() failure is never actionable


_EU_NUM_RE = re.compile(r"^-?\d{1,3}(\.\d{3})+(,\d+)?$|^-?\d+,\d+$")


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

    # Determine engine based on file extension
    if path.lower().endswith('.xlsx') or path.lower().endswith('.xlsm'):
        engine = 'openpyxl'
    elif path.lower().endswith('.xls'):
        engine = 'xlrd'
    else:
        engine = None  # Let pandas figure it out

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