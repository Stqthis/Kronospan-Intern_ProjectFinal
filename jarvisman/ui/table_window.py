"""Interactive table dialog: a real QTableWidget for results too wide or too
long to read inline in the chat. Unlike the chat's QTextBrowser (a text
document, which lays tables out to viewport width and wraps cell text), this
gives native horizontal + vertical scrolling, click-to-sort, and columns the
user can drag to resize. Presentation only -- the values are exactly the
strings the engine produced."""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


def _condense_rows(headers: list, rows: list) -> tuple[list, list, str]:
    """Shrink a repetitive table for display. Two information-preserving moves:
    lift columns that hold one value in every row into a caption, and fold
    exact-duplicate rows into a single row with a 'count'. Returns
    (headers, rows, caption). Small or already-unique tables are unchanged with
    an empty caption. Never lifts away every column, and only collapses when
    duplication is heavy, so genuinely-varied tables are left alone."""
    n = len(rows)
    if n <= 20 or not headers:
        return headers, rows, ""
    ncol = len(headers)
    # 1) constant columns -> caption
    const_idx = []
    for c in range(ncol):
        vals = {(r[c] if c < len(r) else "") for r in rows}
        if len(vals) <= 1:
            const_idx.append(c)
    caption_bits = []
    keep_idx = list(range(ncol))
    if const_idx and len(const_idx) < ncol:
        for c in const_idx:
            v = rows[0][c] if c < len(rows[0]) else ""
            caption_bits.append(f"{headers[c]} = {v if v != '' else '(blank)'}")
        keep_idx = [c for c in range(ncol) if c not in const_idx]
        headers = [headers[c] for c in keep_idx]
        rows = [[r[c] if c < len(r) else "" for c in keep_idx] for r in rows]
    # 2) collapse exact-duplicate rows -> add a count
    counts: dict = {}
    order: list = []
    for r in rows:
        key = tuple(r)
        if key not in counts:
            counts[key] = 0
            order.append(key)
        counts[key] += 1
    if len(order) <= n * 0.6 and len(order) < n:
        headers = list(headers) + ["count"]
        rows = [list(k) + [f"{counts[k]:,}"] for k in
                sorted(order, key=lambda k: counts[k], reverse=True)]
        caption_bits.append(
            f"collapsed {n:,} rows \u2192 {len(order):,} unique")
    return headers, rows, ("  \u2022  ".join(caption_bits) if caption_bits
                           else "")


def _is_number(s: str) -> bool:
    probe = (s or "").replace(",", "").replace("%", "").replace("\u20ac", "").strip()
    if not probe:
        return False
    try:
        float(probe)
        return True
    except ValueError:
        return False


class _NumericItem(QTableWidgetItem):
    """Sorts numerically when the cell holds a formatted number, so '1,200'
    sorts after '999' instead of lexicographically before it."""

    def __lt__(self, other: QTableWidgetItem) -> bool:  # type: ignore[override]
        a, b = self.text(), other.text()
        if _is_number(a) and _is_number(b):
            clean = lambda s: float(s.replace(",", "").replace("%", "")
                                     .replace("\u20ac", "").strip())
            try:
                return clean(a) < clean(b)
            except ValueError:
                pass
        return a.casefold() < b.casefold()


class TableWindow(QDialog):
    def __init__(self, title: str, headers: list, rows: list,
                 theme: Optional[dict] = None, parent=None,
                 total_rows: Optional[int] = None) -> None:
        super().__init__(parent)
        self.theme = theme or {}
        headers = [str(h) for h in (headers or [])]
        rows = [list(r) for r in (rows or [])]
        # Collapse repetition at the point of display, so it works no matter
        # which engine path produced the table (and even if the query-time
        # condensing didn't run): lift columns that are identical in every row
        # into a caption, and fold exact-duplicate rows into one with a count.
        raw_total = total_rows or len(rows)
        headers, rows, self._condense_note = _condense_rows(headers, rows)
        self._headers = headers
        self._rows = rows
        # TRUE row count from the engine. The HTML is capped
        # (QUERY_MAX_TABLE_ROWS), so the caption must not present the capped
        # number as the whole answer.
        self._total = raw_total
        self._title = (title or "Result").strip()
        self.setWindowTitle(("Table \u2014 " + self._title)[:90])
        # QDialog defaults to close-only; ask for the min/max hints so the
        # window can be maximized/fullscreened like any normal window.
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint)
        self.resize(980, 620)
        self._build()

    # ------------------------------------------------------------------ #
    @classmethod
    def from_table(cls, title: str, headers: list, rows: list,
                   theme: Optional[dict] = None, parent=None,
                   total_rows: Optional[int] = None) -> "TableWindow":
        return cls(title, headers, rows, theme=theme, parent=parent,
                   total_rows=total_rows)

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        top = QHBoxLayout()
        if self._total > len(self._rows):
            cap_txt = (f"showing {len(self._rows):,} of {self._total:,} rows "
                       f"\u00d7 {len(self._headers)} columns \u2014 truncated")
        else:
            cap_txt = f"{len(self._rows):,} rows \u00d7 {len(self._headers)} columns"
        if getattr(self, "_condense_note", ""):
            cap_txt += f"    ({self._condense_note})"
        cap = QLabel(cap_txt)
        cap.setStyleSheet(f"color:{self.theme.get('muted', '#888')};")
        top.addWidget(cap)
        top.addStretch(1)
        copy_btn = QPushButton("Copy all")
        copy_btn.clicked.connect(self._copy_all)
        top.addWidget(copy_btn)
        root.addLayout(top)

        t = QTableWidget(len(self._rows), len(self._headers), self)
        t.setHorizontalHeaderLabels(self._headers)
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.setAlternatingRowColors(True)
        # THE point of this window: never squeeze columns to fit the viewport;
        # size each to its content and let the user scroll sideways instead.
        t.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive)
        t.horizontalHeader().setStretchLastSection(False)
        t.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        t.setHorizontalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel)

        for r, row in enumerate(self._rows):
            for c in range(len(self._headers)):
                val = str(row[c]) if c < len(row) else ""
                item = _NumericItem(val)
                if _is_number(val):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                          | Qt.AlignmentFlag.AlignVCenter)
                t.setItem(r, c, item)

        t.resizeColumnsToContents()
        # keep any single runaway column from dominating the viewport
        for c in range(t.columnCount()):
            if t.columnWidth(c) > 380:
                t.setColumnWidth(c, 380)
        t.setSortingEnabled(True)
        self._style_table(t)
        root.addWidget(t)
        self.table = t

        QShortcut(QKeySequence.StandardKey.Copy, self, self._copy_selection)

    # ------------------------------------------------------------------ #
    def _style_table(self, t: QTableWidget) -> None:
        if not self.theme:
            return
        from jarvisman.ui.theme_manager import ThemeManager
        t.setStyleSheet(ThemeManager.table_stylesheet(self.theme))

    # ------------------------------------------------------------------ #
    def _tsv(self, rows: list) -> str:
        out = ["\t".join(self._headers)]
        out += ["\t".join(r) for r in rows]
        return "\n".join(out)

    def _copy_all(self) -> None:
        QGuiApplication.clipboard().setText(
            self._tsv([[str(v) for v in r] for r in self._rows]))

    def _copy_selection(self) -> None:
        sel = self.table.selectedRanges()
        if not sel:
            return self._copy_all()
        rows = []
        for rng in sel:
            for r in range(rng.topRow(), rng.bottomRow() + 1):
                rows.append([self.table.item(r, c).text()
                             if self.table.item(r, c) else ""
                             for c in range(self.table.columnCount())])
        QGuiApplication.clipboard().setText(self._tsv(rows))