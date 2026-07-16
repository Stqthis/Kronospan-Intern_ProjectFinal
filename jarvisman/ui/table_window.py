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
                 theme: Optional[dict] = None, parent=None) -> None:
        super().__init__(parent)
        self.theme = theme or {}
        self._headers = [str(h) for h in (headers or [])]
        self._rows = [list(r) for r in (rows or [])]
        self._title = (title or "Result").strip()
        self.setWindowTitle(("Table \u2014 " + self._title)[:90])
        self.resize(980, 620)
        self._build()

    # ------------------------------------------------------------------ #
    @classmethod
    def from_table(cls, title: str, headers: list, rows: list,
                   theme: Optional[dict] = None, parent=None) -> "TableWindow":
        return cls(title, headers, rows, theme=theme, parent=parent)

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        top = QHBoxLayout()
        cap = QLabel(f"{len(self._rows)} rows \u00d7 {len(self._headers)} columns")
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
        T = self.theme
        if not T:
            return
        t.setStyleSheet(
            f"QTableWidget {{ background:{T.get('panel', '#fff')}; "
            f"alternate-background-color:{T.get('panel2', '#f4f4f4')}; "
            f"color:{T.get('text', '#222')}; "
            f"gridline-color:{T.get('border', '#ddd')}; }}"
            f"QHeaderView::section {{ background:{T.get('panel2', '#eee')}; "
            f"color:{T.get('text', '#222')}; padding:6px; "
            f"border:0px; border-bottom:1px solid {T.get('border', '#ddd')}; "
            f"font-weight:600; }}")

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