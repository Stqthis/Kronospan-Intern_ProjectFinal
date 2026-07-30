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
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QKeySequence, QShortcut
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Wedge

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
        # A QDialog defaults to a close button only -- no maximize, so the
        # window could not be fullscreened. Asking for the min/max hints
        # turns it into a normal resizable window with the full titlebar
        # controls (and enables double-click-titlebar to maximize).
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint)
        self.resize(860, 580)
        self._annot = None
        self._anim = None
        self._anim_token = 0
        self._shown_once = False
        self._build()
        self._replot()

    def showEvent(self, ev):                              # noqa: N802
        """Replay the entrance animation on first show. Windows built
        before .show() (e.g. ChartWindow.from_table) used to animate while
        still hidden, so the user only ever saw the finished frame."""
        super().showEvent(ev)
        if not self._shown_once:
            self._shown_once = True
            if getattr(cfg, "ANIMATIONS", True):
                QTimer.singleShot(50, self._replot)

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
        # Default to showing EVERY category (0 = all). The user can still cap
        # to the top-N by typing a number into the "Top" spinbox below.
        spec = charting.default_spec(self.df, top_n=0,
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
        self.topn.setRange(0, 100000)
        self.topn.setValue(0)   # 0 = show all rows by default
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

        # Fullscreen toggle: fills the whole screen so a chart with many
        # categories is readable. Also bound to F11; Esc exits fullscreen.
        self.fs_btn = QPushButton("\u26f6  Fullscreen")
        self.fs_btn.setToolTip("Toggle fullscreen (F11)")
        self.fs_btn.clicked.connect(self._toggle_fullscreen)
        controls.addWidget(self.fs_btn)
        root.addLayout(controls)

        QShortcut(QKeySequence(Qt.Key.Key_F11), self,
                  activated=self._toggle_fullscreen)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self,
                  activated=self._exit_fullscreen)

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
    def _toggle_fullscreen(self) -> None:
        """Flip between fullscreen and normal. The matplotlib canvas has
        stretch=1, so it grows to fill the extra space and redraws crisply."""
        if self.isFullScreen():
            self._exit_fullscreen()
        else:
            self.showFullScreen()
            if hasattr(self, "fs_btn"):
                self.fs_btn.setText("\u26f6  Exit fullscreen")

    def _exit_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
            if hasattr(self, "fs_btn"):
                self.fs_btn.setText("\u26f6  Fullscreen")

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
        self._stop_anim()
        charting.render(self.fig, self.df, self._spec(), self.theme,
                        title=self._title)
        if getattr(cfg, "ANIMATIONS", True):
            self._animate_entrance()
        else:
            self.canvas.draw_idle()

    def _stop_anim(self) -> None:
        try:
            if self._anim is not None:
                if hasattr(self._anim, "event_source"):
                    self._anim.event_source.stop()
                else:
                    self._anim.stop()
        except Exception:
            pass
        self._anim = None
        self._anim_token += 1   # invalidate any pending frames

    def _animate_entrance(self) -> None:
        """Grow/sweep/draw the freshly-rendered chart in. Fully guarded: any
        hiccup falls back to a static draw so a chart never fails to show."""
        try:
            ax = self.fig.axes[0] if self.fig.axes else None
            if ax is None:
                self.canvas.draw_idle()
                return
            horizontal = self._spec().chart_type == "barh"
            rects = [(r, r.get_height(), r.get_width())
                     for r in ax.patches if isinstance(r, Rectangle)]
            wedges = [(w, w.theta1, w.theta2)
                      for w in ax.patches if isinstance(w, Wedge)]
            lines = [(ln, list(ln.get_xdata()), list(ln.get_ydata()))
                     for ln in ax.lines]
            colls = [(c, c.get_alpha() if c.get_alpha() is not None else 1.0)
                     for c in ax.collections]
            texts = [(t, t.get_alpha() if t.get_alpha() is not None else 1.0)
                     for t in ax.texts]
            if not (rects or wedges or lines or colls):
                self.canvas.draw_idle()
                return

            def _ease(t):
                return t * t * (3.0 - 2.0 * t)          # smoothstep

            def _set(t):
                for r, h, w in rects:
                    if horizontal:
                        r.set_width(w * t)
                    else:
                        r.set_height(h * t)
                for wd, a1, a2 in wedges:
                    wd.set_theta2(a1 + (a2 - a1) * t)
                for ln, xs, ys in lines:
                    n = max(1, int(len(xs) * t)) if xs else 0
                    ln.set_data(xs[:n], ys[:n])
                for c, a in colls:
                    c.set_alpha(a * t)
                for tx, a in texts:
                    tx.set_alpha(a * t)

            frames = max(2, int(getattr(cfg, "CHART_ANIM_FRAMES", 26)))
            interval = int(getattr(cfg, "CHART_ANIM_INTERVAL_MS", 22))
            _set(0.0)
            self.canvas.draw_idle()
            token = self._anim_token
            timer = QTimer(self)
            self._anim = timer
            state = {"i": 0}

            def _tick():
                if self._anim_token != token:      # superseded by a re-plot
                    timer.stop()
                    return
                state["i"] += 1
                t = (1.0 if state["i"] >= frames
                     else _ease(state["i"] / frames))
                _set(t)
                self.canvas.draw_idle()
                if state["i"] >= frames:
                    timer.stop()

            timer.timeout.connect(_tick)
            timer.start(interval)
        except Exception:
            try:
                self.canvas.draw_idle()
            except Exception:
                pass

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