"""Interactive chart dialog: a live matplotlib canvas with zoom/pan/save plus
controls to re-shape any answer into a chart. Drawing is delegated to
``charting`` (pure, Qt-free)."""
from __future__ import annotations

from typing import Optional

import pandas as pd
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure

from jarvisman import config as cfg
from jarvisman.runtime import numfmt
from jarvisman.ui import charting


class ChartWindow(QDialog):
    def __init__(self, title: str, df: pd.DataFrame,
                 theme: Optional[dict] = None, parent=None) -> None:
        super().__init__(parent)
        self.df = df if df is not None else pd.DataFrame()
        self.theme = theme or charting.DEFAULT_THEME
        self._title = title or "Chart"
        self.setWindowTitle(("Chart \u2014 " + self._title)[:90])
        self.resize(860, 580)
        self._annot = None
        self._build()
        self._replot()

    # ------------------------------------------------------------------ #
    @classmethod
    def from_table(cls, title: str, headers: list, rows: list,
                   theme: Optional[dict] = None, parent=None) -> "ChartWindow":
        """Build from a parsed HTML table (headers, rows)."""
        df = charting.frame_from_table(headers, rows)
        return cls(title, df, theme=theme, parent=parent)

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        cats, nums = charting.infer_roles(self.df)
        cols = [str(c) for c in self.df.columns]
        spec = charting.default_spec(self.df, top_n=cfg.CHART_TOP_N,
                                     value_labels=cfg.CHART_VALUE_LABELS)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        controls = QHBoxLayout()
        controls.setSpacing(8)

        self.type_combo = QComboBox()
        self.type_combo.addItems(list(charting.CHART_TYPES))
        self.type_combo.setCurrentText(spec.chart_type)

        self.x_combo = QComboBox()
        self.x_combo.addItems(cols or ["(none)"])
        if spec.x:
            self.x_combo.setCurrentText(spec.x)

        self.y_combo = QComboBox()
        self.y_combo.addItems(nums or cols or ["(none)"])
        if spec.y:
            self.y_combo.setCurrentText(spec.y[0])

        self.hue_combo = QComboBox()
        self.hue_combo.addItems(["(none)"] + cats)

        self.topn = QSpinBox()
        self.topn.setRange(0, 1000)
        self.topn.setValue(cfg.CHART_TOP_N)
        self.topn.setToolTip("Show the top N categories; 0 = all")

        self.sort_desc = QCheckBox("Sort \u2193")
        self.sort_desc.setChecked(True)

        for lbl, w in (("Type", self.type_combo), ("Axis", self.x_combo),
                       ("Measure", self.y_combo), ("Breakdown", self.hue_combo),
                       ("Top", self.topn)):
            cap = QLabel(lbl)
            cap.setStyleSheet(f"color:{self.theme['muted']}; font-size:11px;")
            controls.addWidget(cap)
            controls.addWidget(w)
        controls.addWidget(self.sort_desc)
        controls.addStretch(1)
        root.addLayout(controls)

        self.fig = Figure(figsize=(6.5, 4.2))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        root.addWidget(self.toolbar)
        root.addWidget(self.canvas, stretch=1)

        # re-plot on any change
        for combo in (self.type_combo, self.x_combo, self.y_combo,
                      self.hue_combo):
            combo.currentTextChanged.connect(self._replot)
        self.topn.valueChanged.connect(self._replot)
        self.sort_desc.toggled.connect(self._replot)

        # hover tooltip over bars / points (best-effort)
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)

        self._style_controls()

    def _style_controls(self) -> None:
        T = self.theme
        self.setStyleSheet(f"""
            QDialog {{ background: {T['bg']}; }}
            QLabel {{ color: {T['muted']}; }}
            QComboBox, QSpinBox {{ background: {T['panel']}; color: {T['text']};
                border: 1px solid {T['grid']}; border-radius: 6px;
                padding: 4px 8px; }}
            QComboBox QAbstractItemView {{ background: {T['panel']};
                color: {T['text']}; selection-background-color: {T['accent']}; }}
            QCheckBox {{ color: {T['text']}; }}
            QToolBar {{ background: {T['panel']}; border: none; }}
        """)

    # ------------------------------------------------------------------ #
    def _spec(self) -> charting.ChartSpec:
        hue = self.hue_combo.currentText()
        y = self.y_combo.currentText()
        x = self.x_combo.currentText()
        return charting.ChartSpec(
            chart_type=self.type_combo.currentText(),
            x=None if x in ("", "(none)") else x,
            y=[y] if y and y != "(none)" else [],
            hue=None if hue == "(none)" else hue,
            top_n=self.topn.value(),
            sort_desc=self.sort_desc.isChecked(),
            value_labels=cfg.CHART_VALUE_LABELS,
        )

    def _replot(self) -> None:
        self._annot = None
        charting.render(self.fig, self.df, self._spec(), self.theme,
                        title=self._title)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ #
    def _on_hover(self, event) -> None:
        """Show a value tooltip over the bar/point under the cursor. Fully
        guarded -- a hover hiccup must never disturb the chart."""
        try:
            ax = event.inaxes
            if ax is None:
                self._hide_annot()
                return
            target = None
            for patch in ax.patches:                      # bars
                contains, _ = patch.contains(event)
                if contains:
                    h, w = patch.get_height(), patch.get_width()
                    val = h if abs(h) >= abs(w) else w
                    target = (event.xdata, event.ydata, val)
                    break
            if target is None:
                self._hide_annot()
                return
            x, y, val = target
            if self._annot is None:
                self._annot = ax.annotate(
                    "", xy=(0, 0), xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.4",
                              fc=self.theme["panel"], ec=self.theme["accent"]),
                    color=self.theme["text"], fontsize=9)
            self._annot.xy = (x, y)
            self._annot.set_text(numfmt.fmt(val))
            self._annot.set_visible(True)
            self.canvas.draw_idle()
        except Exception:
            self._hide_annot()

    def _hide_annot(self) -> None:
        try:
            if self._annot is not None and self._annot.get_visible():
                self._annot.set_visible(False)
                self.canvas.draw_idle()
        except Exception:
            self._annot = None
