"""Dashboard / Reports / System pages for the redesigned shell.

Every page is presentation-only: it reads the data the app has already
loaded (``window.agent.dataframes`` and the PDF chunk store) and renders
KPI cards, animated charts and deterministic report tables. Nothing here
mutates the index or talks to the network.

Animations (all disabled by RAG_ANIMATIONS=0):
  * KPI values count up from 0 to their real value.
  * Charts grow in — bars rise, horizontal bars extend, donuts sweep.
  * Clicking any dashboard chart pops it up enlarged and replays the
    animation (use the popup for reading exact figures).
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Callable, Optional

from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from jarvisman import config as cfg


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _theme(window) -> dict:
    return getattr(window, "current_theme", {}) or {}


def _tables(window) -> dict:
    return getattr(window.agent, "dataframes", {}) or {}


def _later(msec: int, owner, fn):
    """A single-shot timer OWNED by `owner`.

    The static QTimer.singleShot belongs to nobody and fires even after its
    target widget is destroyed -- which aborts the process, because the
    RuntimeError surfaces inside the Qt event loop where nothing can catch
    it. A timer parented to the widget is deleted with it, so a rebuild that
    tears down cards mid-animation is simply a no-op.
    """
    t = QTimer(owner)
    t.setSingleShot(True)
    t.timeout.connect(fn)
    t.start(max(0, int(msec)))
    return t


def _bridge(window):
    """RemoteBridge when the app runs as a thin client of the Docker
    backend (RAG_BACKEND_URL), else None."""
    return getattr(window, "bridge", None)


def _docs(window) -> list:
    br = _bridge(window)
    if br is not None:
        return [(d.get("file", "?"), int(d.get("pages") or 0))
                for d in (br.status() or {}).get("docs") or []]
    try:
        return window._indexed_documents()
    except Exception:
        return []


def _chunks(window) -> int:
    br = _bridge(window)
    if br is not None:
        return int((br.status() or {}).get("chunks") or 0)
    return int(getattr(window.store, "count", 0) or 0)


def _tables_meta(window) -> list:
    """[(name, rows, cols)] from the local frames or the remote status."""
    br = _bridge(window)
    if br is not None:
        return [(t.get("name", "?"), int(t.get("rows") or 0),
                 int(t.get("cols") or 0))
                for t in (br.status() or {}).get("tables") or []]
    return [(str(k), df.shape[0], df.shape[1])
            for k, df in _tables(window).items()]


def _remote_report_df(br, rid: int):
    import pandas as pd
    payload = br.run_report(rid) or {}
    return pd.DataFrame(payload.get("rows") or [],
                        columns=payload.get("columns") or [])


def _animations_on() -> bool:
    return bool(getattr(cfg, "ANIMATIONS", True))


def _ease(t: float) -> float:
    """Smoothstep easing — gentle start, gentle landing."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _fmt_int(n) -> str:
    """Integer in the active locale style (see runtime.numfmt)."""
    from jarvisman.runtime import numfmt
    try:
        return numfmt.fmt(int(n))
    except Exception:
        return str(n)


def _human(n) -> str:
    """2223000000 -> '2.22B'; 12345 -> '12,345' -- abbreviated for KPI cards,
    but using the SAME decimal/grouping convention as chat, tables and charts.
    The dashboard used to hardcode US separators while numfmt resolved to EU
    from the locale, so one screen showed both."""
    from jarvisman.runtime import numfmt
    try:
        n = float(n)
    except Exception:
        return str(n)
    eu = numfmt.style() == "eu"
    def _dec(s):
        return s.replace(".", ",") if eu else s
    a = abs(n)
    if a >= 1e9:
        return _dec(f"{n / 1e9:.2f}") + "B"
    if a >= 1e6:
        return _dec(f"{n / 1e6:.1f}") + "M"
    if a >= 1e5:
        return _dec(f"{n / 1e3:.0f}") + "K"
    return numfmt.fmt(n)


_AMOUNT_HINTS = ("eur", "amount", "balance", "value", "total", "outstanding",
                 "sum", "funds", "capital", "deposit")
_CAT_HINTS = ("country", "currency", "bank", "company", "lender", "borrower",
              "name", "type", "category", "counterparty", "entity")


def _numeric_cols(df) -> list:
    return [c for c in df.columns
            if str(df[c].dtype).startswith(("int", "float"))]


def _best_amount_col(df) -> Optional[str]:
    nums = _numeric_cols(df)
    if not nums:
        return None
    for c in nums:
        if any(h in str(c).lower() for h in _AMOUNT_HINTS):
            return c
    try:
        return max(nums, key=lambda c: abs(float(df[c].sum())))
    except Exception:
        return nums[0]


def _cat_cols(df, max_unique: int = 60) -> list:
    out = []
    for c in df.columns:
        if str(df[c].dtype) in ("object", "category", "string"):
            try:
                u = df[c].nunique(dropna=True)
            except Exception:
                continue
            if 2 <= u <= max_unique:
                out.append((c, u))
    out.sort(key=lambda cu: (not any(h in str(cu[0]).lower()
                                     for h in _CAT_HINTS), cu[1]))
    return [c for c, _u in out]


def _clear_layout(lay) -> None:
    while lay.count():
        item = lay.takeAt(0)
        w = item.widget()
        if w is not None:
            w.deleteLater()
        child = item.layout()
        if child is not None:
            _clear_layout(child)


def _page_header(title: str, subtitle: str,
                 refresh: Optional[Callable] = None) -> QHBoxLayout:
    head = QHBoxLayout()
    box = QVBoxLayout()
    box.setSpacing(2)
    t = QLabel(title)
    t.setObjectName("pagetitle")
    s = QLabel(subtitle)
    s.setObjectName("pagesub")
    s.setWordWrap(True)
    box.addWidget(t)
    box.addWidget(s)
    head.addLayout(box)
    head.addStretch(1)
    if refresh is not None:
        btn = QPushButton("⟳  Refresh")
        btn.setObjectName("linkbtn")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(refresh)
        head.addWidget(btn, alignment=Qt.AlignmentFlag.AlignTop)
    return head


def _fade_in(widget, duration: int = 220, parent=None) -> None:
    """Fade a widget in; clears the effect afterwards (matplotlib canvases
    dislike a permanently-installed opacity effect)."""
    if not _animations_on():
        return
    try:
        eff = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(eff)
        anim = QPropertyAnimation(eff, b"opacity", parent or widget)
        anim.setDuration(duration)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.finished.connect(lambda w=widget: w.setGraphicsEffect(None))
        widget._fade_anim = anim          # keep a ref so it is not GC'd
        anim.start()
    except Exception:
        try:
            widget.setGraphicsEffect(None)
        except Exception:
            pass


class KpiCard(QFrame):
    """Stat card: NAME (caps, muted) / big value / muted sub line.
    When ``raw`` (a number) and ``fmt`` are given, the value counts up
    from zero to its real figure."""

    FRAMES = 28
    INTERVAL = 16          # ms -> ~450 ms count-up

    def __init__(self, name: str, value: str, sub: str = "",
                 badge: str = "", badge_color: str = "#3d8bfd",
                 raw=None, fmt: Optional[Callable] = None) -> None:
        super().__init__()
        self.setObjectName("kpi")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 13, 16, 13)
        lay.setSpacing(4)
        top = QHBoxLayout()
        nm = QLabel(name.upper())
        nm.setObjectName("kpiname")
        top.addWidget(nm)
        top.addStretch(1)
        if badge:
            b = QLabel(badge)
            b.setObjectName("srcbadge")
            b.setStyleSheet(f"background:{badge_color};")
            top.addWidget(b)
        lay.addLayout(top)
        self._value_lbl = QLabel(value)
        self._value_lbl.setObjectName("kpivalue")
        lay.addWidget(self._value_lbl)
        if sub:
            sb = QLabel(sub)
            sb.setObjectName("kpisub")
            sb.setWordWrap(True)
            lay.addWidget(sb)
        lay.addStretch(1)
        self.setMinimumHeight(96)

        self._final_text = value
        if raw is not None and _animations_on():
            try:
                self._count_up(float(raw), fmt or _human)
            except Exception:
                self._value_lbl.setText(value)
        _fade_in(self, 260, self)

    def _count_up(self, target: float, fmt: Callable) -> None:
        state = {"i": 0}
        timer = QTimer(self)
        self._count_timer = timer

        def _tick():
            state["i"] += 1
            t = _ease(state["i"] / self.FRAMES)
            if state["i"] >= self.FRAMES:
                self._value_lbl.setText(self._final_text)
                timer.stop()
            else:
                self._value_lbl.setText(fmt(target * t))
        self._value_lbl.setText(fmt(0))
        timer.timeout.connect(_tick)
        timer.start(self.INTERVAL)


class ChartCard(QFrame):
    """Card with a title, subtitle and an embedded matplotlib figure.

    ``play(spec)`` draws the chart with an entrance animation (bars grow,
    donuts sweep). Clicking the card calls ``on_click`` (the dashboard uses
    this to pop the chart up enlarged)."""

    FRAMES = 26
    INTERVAL = 20          # ms -> ~520 ms sweep

    def __init__(self, theme: dict, title: str, subtitle: str = "",
                 on_click: Optional[Callable] = None) -> None:
        super().__init__()
        self.setObjectName("card")
        self._theme = theme
        self._on_click = on_click
        self._timer = None
        self._spec = None
        if on_click is not None:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setToolTip("Click to enlarge")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 12)
        lay.setSpacing(6)
        trow = QHBoxLayout()
        t = QLabel(title)
        t.setStyleSheet("font-size:14px; font-weight:700;")
        trow.addWidget(t)
        trow.addStretch(1)
        if on_click is not None:
            hint = QLabel("⤢")
            hint.setStyleSheet(
                f"color:{theme.get('muted', '#8f9ab0')}; font-size:13px;")
            trow.addWidget(hint)
        lay.addLayout(trow)
        if subtitle:
            s = QLabel(subtitle)
            s.setObjectName("kpisub")
            lay.addWidget(s)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg as FigureCanvas)
        self.fig = Figure(figsize=(4.4, 2.9), dpi=100)
        self.fig.patch.set_facecolor(theme.get("panel", "#111624"))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background: transparent;")
        self.canvas.setMinimumHeight(230)
        lay.addWidget(self.canvas, stretch=1)
        self.setMinimumHeight(300)

    def mousePressEvent(self, ev):                        # noqa: N802
        if self._on_click is not None and self._spec is not None:
            try:
                self._on_click(self._spec)
            except Exception:
                pass
        super().mousePressEvent(ev)

    # -- public API ---------------------------------------------------------- #
    def play(self, spec) -> None:
        """spec = (kind, title, sub, labels, values, color)."""
        self._spec = spec
        kind, _t, _s, labels, values, color = spec
        self._stop()
        if not labels or not values:
            self.draw_empty()
            return
        if not labels or not values:
            self.draw_empty()
            return
        n = min(len(labels), len(values))
        labels, values = list(labels)[:n], list(values)[:n]
        if not _animations_on():
            self._draw_frame(kind, labels, values, color, 1.0)
            return
        state = {"i": 0}
        timer = QTimer(self)
        self._timer = timer

        def _tick():
            state["i"] += 1
            t = _ease(state["i"] / self.FRAMES)
            done = state["i"] >= self.FRAMES
            try:
                self._draw_frame(kind, labels, values, color,
                                 1.0 if done else t)
            except Exception:
                timer.stop()
                self.draw_error()
                return
            if done:
                timer.stop()
        try:
            self._draw_frame(kind, labels, values, color, 0.04)
        except Exception:
            self.draw_error()
            return
        timer.timeout.connect(_tick)
        timer.start(self.INTERVAL)

    def _stop(self) -> None:
        if self._timer is not None:
            try:
                self._timer.stop()
            except Exception:
                pass
            self._timer = None

    # -- frame renderers ------------------------------------------------------ #
    def _draw_frame(self, kind: str, labels, values, color, t: float) -> None:
        if kind == "donut":
            self._frame_donut(labels, values, t)
        elif kind == "hbar":
            self._frame_hbar(labels, values, color, t)
        else:
            self._frame_bar(labels, values, color, t)
        self.canvas.draw_idle()

    def _style_axes(self, ax, horizontal: bool = False) -> None:
        T = self._theme
        ax.set_facecolor(T.get("panel", "#111624"))
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors=T.get("muted", "#8f9ab0"), labelsize=8)
        if horizontal:
            ax.xaxis.grid(True, color=T.get("grid", "#222b40"), linewidth=0.7)
        else:
            ax.yaxis.grid(True, color=T.get("grid", "#222b40"), linewidth=0.7)
        ax.set_axisbelow(True)

    def _frame_donut(self, labels, values, t: float) -> None:
        T = self._theme
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(T.get("panel", "#111624"))
        cyc = T.get("cycle", ["#3d8bfd"])
        colors = [cyc[i % len(cyc)] for i in range(len(values))]
        total = sum(values) or 1.0
        if t >= 1.0:
            shown, cols = list(values), list(colors)
        else:
            # sweep: scale the real slices by t and fill the rest with an
            # invisible wedge so the visible part grows clockwise
            shown = [v * t for v in values] + [total * (1.0 - t)]
            cols = colors + [T.get("panel", "#111624")]
        wedges, _ = ax.pie(
            shown, colors=cols, startangle=90, counterclock=False,
            wedgeprops=dict(width=0.42,
                            edgecolor=T.get("panel", "#111624"),
                            linewidth=1.5))
        if t >= 1.0:
            ax.legend(wedges[:len(labels)], labels, loc="center left",
                      bbox_to_anchor=(0.98, 0.5), frameon=False, fontsize=8,
                      labelcolor=T.get("text", "#e8ecf4"))
        self.fig.subplots_adjust(left=0.0, right=0.68, top=0.98, bottom=0.02)

    def _frame_bar(self, labels, values, color, t: float) -> None:
        T = self._theme
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        self._style_axes(ax)
        c = color or T.get("accent", "#3d8bfd")
        ax.bar(range(len(values)), [v * t for v in values],
               color=c, width=0.62)
        top = max(values) if values else 1.0
        ax.set_ylim(0, top * 1.06 if top > 0 else 1.0)
        ax.set_xticks(range(len(labels)))
        short = [str(l)[:12] + ("…" if len(str(l)) > 12 else "")
                 for l in labels]
        ax.set_xticklabels(short, rotation=30, ha="right",
                           color=T.get("muted", "#8f9ab0"))
        self.fig.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.30)

    def _frame_hbar(self, labels, values, color, t: float) -> None:
        T = self._theme
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(T.get("panel", "#111624"))
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors=T.get("muted", "#8f9ab0"), labelsize=8)
        ax.xaxis.grid(True, color=T.get("grid", "#222b40"), linewidth=0.7)
        ax.set_axisbelow(True)
        c = color or T.get("accent", "#3d8bfd")
        y = range(len(values))
        ax.barh(y, [v * t for v in values], color=c, height=0.55)
        top = max(values) if values else 1.0
        ax.set_xlim(0, top * 1.06 if top > 0 else 1.0)
        ax.set_yticks(list(y))
        short = [str(l)[:22] + ("…" if len(str(l)) > 22 else "")
                 for l in labels]
        ax.set_yticklabels(short, color=T.get("text", "#e8ecf4"))
        ax.invert_yaxis()
        self.fig.subplots_adjust(left=0.34, right=0.97, top=0.96, bottom=0.10)

    def draw_empty(self, message: str = "No data loaded yet") -> None:
        T = self._theme
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(T.get("panel", "#111624"))
        ax.axis("off")
        ax.text(0.5, 0.5, message, ha="center", va="center",
                color=T.get("muted", "#8f9ab0"), fontsize=10)
        self.canvas.draw_idle()

    def draw_error(self, message: str = "Chart data was not usable") -> None:
        """A visible failure state. A silently blank card is indistinguishable
        from real zero data, which is the worse outcome on a finance
        dashboard."""
        T = self._theme
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(T.get("panel", "#111624"))
        ax.axis("off")
        ax.text(0.5, 0.56, "\u26a0", ha="center", va="center",
                color=T.get("warn", "#fbbf24"), fontsize=20)
        ax.text(0.5, 0.34, message, ha="center", va="center",
                color=T.get("muted", "#8f9ab0"), fontsize=9, wrap=True)
        self.canvas.draw_idle()


class ChartPopup(QDialog):
    """Enlarged pop-up of a dashboard chart; replays the entrance animation
    each time it is shown."""

    def __init__(self, theme: dict, spec, parent=None) -> None:
        super().__init__(parent)
        self._spec = spec
        kind, title, sub, _labels, _values, _color = spec
        self.setWindowTitle(title[:90])
        # QDialog defaults to close-only; ask for the min/max hints so the
        # popup can be maximized/fullscreened.
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint)
        self.resize(940, 620)
        T = theme
        self.setStyleSheet(
            f"QDialog {{ background: {T.get('bg', '#0b0f16')}; }}"
            f"QLabel {{ color: {T.get('text', '#e8ecf4')}; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 14)
        lay.setSpacing(6)
        t = QLabel(title)
        t.setStyleSheet("font-size:19px; font-weight:800;")
        lay.addWidget(t)
        if sub:
            s = QLabel(sub)
            s.setStyleSheet(
                f"color:{T.get('muted', '#8f9ab0')}; font-size:12px;")
            lay.addWidget(s)
        self.card = ChartCard(theme, "", "")
        self.card.setObjectName("card")
        self.card.canvas.setMinimumHeight(440)
        lay.addWidget(self.card, stretch=1)
        hint = QLabel("Charts animate in — close and reopen to replay.")
        hint.setStyleSheet(
            f"color:{T.get('muted', '#8f9ab0')}; font-size:10px;")
        lay.addWidget(hint)

    def showEvent(self, ev):                              # noqa: N802
        super().showEvent(ev)
        _later(60, self, lambda: self.card.play(self._spec))


# --------------------------------------------------------------------------- #
# Dashboard                                                                    #
# --------------------------------------------------------------------------- #
class DashboardPage(QWidget):
    """Live KPIs across the loaded tables and indexed documents."""

    def __init__(self, window) -> None:
        super().__init__()
        self.window = window
        self._popups = []
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        self.lay = QVBoxLayout(body)
        self.lay.setContentsMargins(26, 20, 26, 20)
        self.lay.setSpacing(14)
        scroll.setWidget(body)
        outer.addWidget(scroll)

        self.lay.addLayout(_page_header(
            "Dashboard",
            "Live KPIs across your loaded tables, documents and index.",
            self.refresh))
        self.kpi_row = QHBoxLayout()
        self.kpi_row.setSpacing(12)
        self.lay.addLayout(self.kpi_row)
        self.chart_grid = QGridLayout()
        self.chart_grid.setSpacing(12)
        self.lay.addLayout(self.chart_grid)
        self.lay.addStretch(1)
        self.refresh()

    # ------------------------------------------------------------------ #
    def refresh(self) -> None:
        _clear_layout(self.kpi_row)
        _clear_layout(self.chart_grid)
        T = _theme(self.window)
        tables = _tables(self.window)
        docs = _docs(self.window)
        meta = _tables_meta(self.window)
        chunks = _chunks(self.window)
        total_rows = sum(r for _n, r, _c in meta)
        total_pages = sum(p for _f, p in docs)
        cycle = T.get("cycle", ["#3d8bfd"])

        # ---- KPI cards (count up) -------------------------------------- #
        cards = [KpiCard("Tables loaded", _fmt_int(len(meta)),
                         f"{_fmt_int(total_rows)} total rows", "DATA",
                         cycle[0], raw=len(meta), fmt=_fmt_int)]
        money = self._headline_amount(tables)
        if money is not None:
            tname, col, total = money
            cards.append(KpiCard(f"Total {col}"[:26], _human(total),
                                 f"{tname} · sum of {col}", "KPI", cycle[1],
                                 raw=total, fmt=_human))
        cards.append(KpiCard("PDF sources", _fmt_int(len(docs)),
                             f"{_fmt_int(total_pages)} indexed pages", "PDF",
                             "#2f6fd8", raw=len(docs), fmt=_fmt_int))
        cards.append(KpiCard("Text chunks", _fmt_int(chunks),
                             "full-text searchable", "RAG", cycle[3],
                             raw=chunks, fmt=_fmt_int))
        model = (getattr(self.window.agent, "chat_model", "")
                 or cfg.DEFAULT_CHAT_MODEL)
        cards.append(KpiCard("Model", model.split(":")[0],
                             model, "LLM", cycle[4]))
        for c in cards:
            self.kpi_row.addWidget(c, 1)

        # ---- charts (animate in, click to enlarge) ---------------------- #
        br = _bridge(self.window)
        if br is not None:
            cyc = T.get("cycle", ["#3d8bfd"])
            specs = [(d.get("kind", "bar"), d.get("title", ""),
                      d.get("sub", ""), d.get("labels") or [],
                      d.get("values") or [], cyc[i % len(cyc)])
                     for i, d in enumerate(br.charts())]
        else:
            specs = self._chart_specs(tables, docs)
        if not specs:
            card = ChartCard(T, "Overview", "load data to see charts")
            card.draw_empty("Add documents on the System page, "
                            "then build the index.")
            self.chart_grid.addWidget(card, 0, 0)
            return
        for i, spec in enumerate(specs):
            _kind, title, sub, _labels, _values, _color = spec
            card = ChartCard(T, title, sub, on_click=self._popup)
            self.chart_grid.addWidget(card, i // 2, i % 2)
            _fade_in(card, 260, card)
            # stagger the entrances left-to-right for a lively dashboard
            delay = 120 * i if _animations_on() else 0
            _later(delay, card, lambda c=card, s=spec: c.play(s))

    def _popup(self, spec) -> None:
        try:
            dlg = ChartPopup(_theme(self.window), spec, parent=self)
            self._popups = [d for d in self._popups if d.isVisible()]
            self._popups.append(dlg)
            dlg.show()
            dlg.raise_()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def _headline_amount(self, tables):
        """(table, column, total) for the most 'money-like' numeric column."""
        best = None
        for name, df in tables.items():
            col = _best_amount_col(df)
            if not col:
                continue
            try:
                total = float(df[col].sum())
            except Exception:
                continue
            if best is None or abs(total) > abs(best[2]):
                best = (str(name), str(col), total)
        return best

    def _chart_specs(self, tables, docs) -> list:
        """Up to four (kind, title, sub, labels, values, color) chart specs
        derived from whatever data is loaded."""
        T = _theme(self.window)
        cycle = T.get("cycle", ["#3d8bfd"])
        specs = []
        pairs = []
        for name, df in tables.items():
            num = _best_amount_col(df)
            if not num:
                continue
            for cat in _cat_cols(df)[:2]:
                pairs.append((str(name), df, str(cat), str(num)))
        seen_cats = set()
        for tname, df, cat, num in pairs:
            key = cat.lower()
            if key in seen_cats or len(specs) >= 3:
                continue
            seen_cats.add(key)
            try:
                g = (df.groupby(cat)[num].sum()
                       .sort_values(ascending=False))
            except Exception:
                continue
            top = g.head(8 if len(specs) == 0 else 10)
            labels = [str(i) for i in top.index]
            values = [float(v) for v in top.values]
            if not values or not any(values):
                continue
            kind = "donut" if len(specs) == 0 else (
                "hbar" if len(specs) == 1 else "bar")
            specs.append((kind, f"{num} by {cat}",
                          f"{tname} · top {len(labels)}",
                          labels, values, cycle[len(specs) % len(cycle)]))
        if tables and len(specs) < 4:
            items = sorted(((str(k), df.shape[0]) for k, df in tables.items()),
                           key=lambda kv: -kv[1])[:8]
            specs.append(("donut" if not specs else "bar",
                          "Rows by table", "loaded tables",
                          [k for k, _v in items], [v for _k, v in items],
                          cycle[1]))
        if docs and len(specs) < 4:
            top = sorted(docs, key=lambda fp: -fp[1])[:10]
            specs.append(("hbar", "Document knowledge base",
                          "indexed pages per PDF",
                          [f for f, _p in top], [p for _f, p in top],
                          "#2f6fd8"))
        return specs[:4]


# --------------------------------------------------------------------------- #
# Reports                                                                      #
# --------------------------------------------------------------------------- #
class _ReportCard(QFrame):
    """A clickable rail item that WRAPS its title/subtitle onto multiple
    lines instead of growing wider, so a long report name makes the card
    taller (vertical scroll, which is fine) rather than forcing the rail to
    scroll horizontally (which QPushButton's single-line text did)."""

    def __init__(self, title: str, sub: str, theme: dict, on_click) -> None:
        super().__init__()
        self.setObjectName("reportcard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setProperty("selected", False)
        self._on_click = on_click

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 9, 12, 9)
        lay.setSpacing(2)

        self.title_lbl = QLabel(title)
        self.title_lbl.setWordWrap(True)
        self.title_lbl.setStyleSheet(
            f"font-weight:700; font-size:13px; color:{theme.get('text', '#fff')};")
        lay.addWidget(self.title_lbl)

        self.sub_lbl = QLabel(sub)
        self.sub_lbl.setWordWrap(True)
        self.sub_lbl.setStyleSheet(
            f"font-size:11px; color:{theme.get('muted', '#888')};")
        lay.addWidget(self.sub_lbl)

        T = theme
        self.setStyleSheet(f"""
            QFrame#reportcard {{ background: {T.get('panel2', '#1c1c22')};
                border: 1px solid {T.get('border', '#333')};
                border-radius: 10px; }}
            QFrame#reportcard:hover {{ border: 1px solid {T.get('accent', '#3d8bfd')}; }}
            QFrame#reportcard[selected="true"] {{
                background: {T.get('accent_dim', '#22314f')};
                border: 1px solid {T.get('accent', '#3d8bfd')}; }}
        """)

    def set_selected(self, value: bool) -> None:
        self.setProperty("selected", value)
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event) -> None:
        if self._on_click:
            self._on_click()
        super().mousePressEvent(event)


class ReportsPage(QWidget):
    """Pre-built deterministic reports over the loaded tables: same pandas
    operation, identical results every run — no LLM round-trip."""

    def __init__(self, window) -> None:
        super().__init__()
        self.window = window
        self._buttons = []
        self._current = None            # (title, sub, df)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(26, 20, 26, 20)
        outer.setSpacing(12)
        outer.addLayout(_page_header(
            "Reports",
            "Pre-built reports over your data. Deterministic — no AI "
            "round-trip, identical results every run.",
            self.refresh))

        split = QHBoxLayout()
        split.setSpacing(14)
        outer.addLayout(split, stretch=1)

        rail_scroll = QScrollArea()
        rail_scroll.setWidgetResizable(True)
        rail_scroll.setFrameShape(QFrame.Shape.NoFrame)
        rail_scroll.setFixedWidth(280)
        # Vertical scroll only -- report titles/subtitles wrap inside each
        # _ReportCard instead of overflowing, so horizontal scroll should
        # never be needed; disabling it outright also hides the stray
        # unstyled scrollbar that showed up when a card's text overflowed.
        rail_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        rail_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        rail_body = QWidget()
        self.rail = QVBoxLayout(rail_body)
        self.rail.setContentsMargins(0, 0, 6, 0)
        self.rail.setSpacing(8)
        rail_scroll.setWidget(rail_body)
        split.addWidget(rail_scroll)

        right = QVBoxLayout()
        right.setSpacing(8)
        head = QHBoxLayout()
        tbox = QVBoxLayout()
        tbox.setSpacing(2)
        self.rep_title = QLabel("Select a report")
        self.rep_title.setStyleSheet("font-size:17px; font-weight:800;")
        self.rep_meta = QLabel("")
        self.rep_meta.setObjectName("kpisub")
        tbox.addWidget(self.rep_title)
        tbox.addWidget(self.rep_meta)
        head.addLayout(tbox)
        head.addStretch(1)
        self.export_csv = QPushButton("Export CSV")
        self.export_csv.setEnabled(False)   # nothing to export until a report loads
        self.export_csv.clicked.connect(lambda: self._export("csv"))
        self.export_xlsx = QPushButton("Export Excel")
        self.export_xlsx.setEnabled(False)
        self.export_xlsx.clicked.connect(lambda: self._export("xlsx"))
        head.addWidget(self.export_csv)
        head.addWidget(self.export_xlsx)
        right.addLayout(head)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search rows …")
        self.search.textChanged.connect(self._apply_filter)
        right.addWidget(self.search)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        right.addWidget(self.table, stretch=1)
        split.addLayout(right, stretch=1)
        self.refresh()

    # ------------------------------------------------------------------ #
    def refresh(self) -> None:
        _clear_layout(self.rail)
        self._buttons = []
        specs = self._report_specs()
        if not specs:
            empty = QLabel("Load data on the System page to unlock the "
                           "pre-built reports.")
            empty.setObjectName("kpisub")
            empty.setWordWrap(True)
            self.rail.addWidget(empty)
            self.rail.addStretch(1)
            return
        theme = _theme(self.window)
        for title, sub, fn in specs:
            card = _ReportCard(title, sub, theme, None)
            card._on_click = (
                lambda t=title, s=sub, f=fn, c=card: self._run(t, s, f, c))
            self._buttons.append(card)
            self.rail.addWidget(card)
        self.rail.addStretch(1)

    def _report_specs(self) -> list:
        br = _bridge(self.window)
        if br is not None:
            # deterministic reports run on the backend; each spec fetches
            # the finished table as JSON
            return [(r.get("title", f"Report {r.get('id')}"),
                     r.get("sub", ""),
                     (lambda rid=r.get("id", -1): _remote_report_df(br, rid)))
                    for r in br.reports()]
        tables = _tables(self.window)
        specs = []
        if tables:
            def _inventory():
                import pandas as pd
                rows = [(self.window._source_code(n, i), str(n),
                         df.shape[0], df.shape[1])
                        for i, (n, df) in enumerate(tables.items())]
                return pd.DataFrame(rows, columns=["CODE", "TABLE",
                                                   "ROWS", "COLUMNS"])
            specs.append(("Tables inventory",
                          "every loaded table with its size", _inventory))
        for name, df in tables.items():
            n = str(name)
            specs.append((f"Overview — {n}"[:48],
                          f"first 200 rows of {df.shape[0]:,}",
                          lambda d=df: d.head(200).copy()))
            num = _best_amount_col(df)
            if num:
                for cat in _cat_cols(df)[:2]:
                    def _agg(d=df, c=cat, m=num):
                        g = (d.groupby(c)[m].agg(["sum", "count"])
                              .sort_values("sum", ascending=False)
                              .reset_index())
                        g.columns = [str(c), f"total {m}", "rows"]
                        return g
                    specs.append((f"{num} by {cat}"[:48],
                                  f"{n} · grouped totals", _agg))
            if _numeric_cols(df):
                def _summary(d=df):
                    s = d.describe().transpose().reset_index()
                    s = s.rename(columns={"index": "column"})
                    return s.round(2)
                specs.append((f"Numeric summary — {n}"[:48],
                              "count / mean / min / max per column",
                              _summary))
        docs = _docs(self.window)
        if docs:
            def _docs_report():
                import pandas as pd
                return pd.DataFrame(docs, columns=["FILE", "PAGES"])
            specs.append(("Indexed documents",
                          "PDF knowledge base contents", _docs_report))
        return specs

    # ------------------------------------------------------------------ #
    def _run(self, title: str, sub: str, fn, card) -> None:
        for b in self._buttons:
            b.set_selected(b is card)
        try:
            df = fn()
        except Exception as exc:
            self.rep_title.setText(title)
            self.rep_meta.setText(f"report failed: {type(exc).__name__}")
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self._current = None
            self.export_csv.setEnabled(False)
            self.export_xlsx.setEnabled(False)
            return
        self._current = (title, sub, df)
        self.export_csv.setEnabled(True)
        self.export_xlsx.setEnabled(True)
        self.rep_title.setText(title)
        stamp = datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")
        self.rep_meta.setText(
            f"{len(df):,} rows · generated {stamp}")
        self.search.blockSignals(True)
        self.search.clear()
        self.search.blockSignals(False)
        self._fill_table(df)
        _fade_in(self.table, 240, self)

    def _fill_table(self, df) -> None:
        self.table.setSortingEnabled(False)
        self.table.setRowCount(min(len(df), 1000))
        self.table.setColumnCount(df.shape[1])
        self.table.setHorizontalHeaderLabels(
            [str(c).upper() for c in df.columns])
        for r in range(min(len(df), 1000)):
            for c in range(df.shape[1]):
                v = df.iat[r, c]
                if isinstance(v, float):
                    text = f"{v:,.2f}"
                else:
                    text = "" if v is None else str(v)
                item = QTableWidgetItem(text)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight
                        | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(r, c, item)
        self.table.resizeColumnsToContents()
        self.table.setSortingEnabled(True)

    def _apply_filter(self, text: str) -> None:
        needle = (text or "").strip().lower()
        for r in range(self.table.rowCount()):
            if not needle:
                self.table.setRowHidden(r, False)
                continue
            hit = False
            for c in range(self.table.columnCount()):
                it = self.table.item(r, c)
                if it is not None and needle in it.text().lower():
                    hit = True
                    break
            self.table.setRowHidden(r, not hit)

    def _export(self, kind: str) -> None:
        if not self._current:
            self._notify("No report loaded to export yet.", error=True)
            return
        title, _sub, df = self._current
        stem = "".join(ch if ch.isalnum() else "_" for ch in title)[:40]
        if kind == "csv":
            path, _ = QFileDialog.getSaveFileName(
                self, "Export CSV", f"{stem}.csv", "CSV (*.csv)")
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Excel", f"{stem}.xlsx", "Excel (*.xlsx)")
        if not path:
            return   # user cancelled the dialog -- not an error
        try:
            if kind == "csv":
                df.to_csv(path, index=False)
            else:
                # Explicit engine: pandas auto-detects by extension, but being
                # explicit surfaces a clear ImportError if openpyxl is ever
                # missing instead of a confusing engine-lookup failure.
                df.to_excel(path, index=False, engine="openpyxl")
        except ImportError:
            self._notify(
                "Excel export needs the 'openpyxl' package "
                "(pip install openpyxl).", error=True)
            return
        except PermissionError:
            self._notify(
                f"Could not write to {path} -- file may be open elsewhere "
                "or the location isn't writable.", error=True)
            return
        except Exception as exc:
            self._notify(f"Export failed: {type(exc).__name__}: {exc}",
                         error=True)
            return
        self._notify(f"Exported to {path}")

    def _notify(self, message: str, error: bool = False) -> None:
        """Surface export feedback on the status bar (falls back to a
        message box if there's no status bar to write to), so failures are
        never silent."""
        if self.window is not None:
            try:
                self.window.status.showMessage(message, 6000 if error else 4000)
                return
            except Exception:
                pass
        if error:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Export", message)


# --------------------------------------------------------------------------- #
# System                                                                       #
# --------------------------------------------------------------------------- #
class SystemPage(QWidget):
    """Health, data inventory and ingestion — plus the model / document
    management controls (the app's settings)."""

    def __init__(self, window) -> None:
        super().__init__()
        self.window = window
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(26, 20, 26, 20)
        lay.setSpacing(14)
        scroll.setWidget(body)
        outer.addWidget(scroll)

        lay.addLayout(_page_header(
            "System",
            "Health, data inventory, ingestion status and settings.",
            self.refresh))

        self.health_row = QHBoxLayout()
        self.health_row.setSpacing(10)
        lay.addLayout(self.health_row)

        # accuracy scoreboard + data warnings (trust surface)
        self.score_row = QHBoxLayout()
        self.score_row.setSpacing(10)
        lay.addLayout(self.score_row)

        self.stats_row = QHBoxLayout()
        self.stats_row.setSpacing(12)
        lay.addLayout(self.stats_row)

        cols = QHBoxLayout()
        cols.setSpacing(14)
        lay.addLayout(cols, stretch=1)

        left = QVBoxLayout()
        left.setSpacing(8)
        tlab = QLabel("Tables")
        tlab.setStyleSheet("font-size:15px; font-weight:800;")
        left.addWidget(tlab)
        self.tables_view = self._make_table(
            ["CODE", "TABLE", "ROWS", "COLUMNS"])
        left.addWidget(self.tables_view)
        dlab = QLabel("Indexed documents")
        dlab.setStyleSheet("font-size:15px; font-weight:800;")
        left.addWidget(dlab)
        self.docs_view = self._make_table(["FILE", "TYPE", "PAGES / CHUNKS"])
        left.addWidget(self.docs_view)
        rlab = QLabel("House rules — business policies the model applies "
                      "to every answer")
        rlab.setStyleSheet("font-size:15px; font-weight:800;")
        left.addWidget(rlab)
        self.rules_edit = QPlainTextEdit()
        self.rules_edit.setMinimumHeight(140)
        self.rules_edit.setStyleSheet(
            "font-family:'Consolas','Courier New',monospace; font-size:12px;")
        left.addWidget(self.rules_edit)
        rrow = QHBoxLayout()
        self.rules_info = QLabel("")
        self.rules_info.setObjectName("kpisub")
        self.rules_save_btn = QPushButton("Save rules")
        self.rules_save_btn.setObjectName("primary")
        self.rules_save_btn.clicked.connect(self._save_rules)
        rrow.addWidget(self.rules_info, stretch=1)
        rrow.addWidget(self.rules_save_btn)
        left.addLayout(rrow)
        self._load_rules()
        cols.addLayout(left, stretch=3)

        right = QVBoxLayout()
        right.setSpacing(12)
        right.addWidget(window._build_models_card())
        right.addWidget(window._build_documents_card(), stretch=1)
        holder = QWidget()
        holder.setLayout(right)
        holder.setFixedWidth(300)
        cols.addWidget(holder)

        self.refresh()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _make_table(headers: list) -> QTableWidget:
        t = QTableWidget()
        t.setColumnCount(len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.setAlternatingRowColors(True)
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setStretchLastSection(True)
        t.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive)
        t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        t.setMinimumHeight(170)
        t.setSizePolicy(QSizePolicy.Policy.Expanding,
                        QSizePolicy.Policy.Expanding)
        return t

    def _chip(self, label: str, ok: Optional[bool]) -> QLabel:
        T = _theme(self.window)
        if ok is None:
            obj, txt, color = "healthok", "…", T.get("muted", "#8f9ab0")
        elif ok:
            obj, txt, color = "healthok", "OK", T.get("ok", "#34d399")
        else:
            obj, txt, color = "healthbad", "DOWN", T.get("err", "#f87171")
        chip = QLabel(f"● {label}   —   {txt}")
        chip.setObjectName(obj)
        chip.setStyleSheet(f"color:{color};")
        return chip

    def refresh(self) -> None:
        w = self.window
        _clear_layout(self.health_row)
        _clear_layout(self.score_row)
        _clear_layout(self.stats_row)
        T = _theme(w)
        tables = _tables(w)
        docs = _docs(w)
        meta = _tables_meta(w)
        chunks = _chunks(w)
        total_rows = sum(r for _n, r, _c in meta)
        total_pages = sum(p for _f, p in docs)
        alive = getattr(w, "_ollama_alive", None)

        self.health_row.addWidget(self._chip("Ollama host", alive))
        self.health_row.addWidget(self._chip(
            "Model loaded", None if alive is None else bool(alive)))
        self.health_row.addWidget(self._chip(
            "Index store", bool(meta) or chunks > 0))
        self.health_row.addWidget(self._chip(
            "Data directory", os.path.isdir(cfg.INDEX_DIR)
            or os.path.isdir(os.path.dirname(cfg.INDEX_DIR))))
        self.health_row.addStretch(1)

        # -- accuracy scoreboard + data warnings -------------------------- #
        br = _bridge(w)
        try:
            if br is not None:
                sb = br.scoreboard()
                warns = (br.status() or {}).get("warnings") or []
            else:
                from jarvisman.runtime import scoreboard as _sb
                sb = _sb.read()
                warns = []
        except Exception:
            sb, warns = {}, []
        dc = sb.get("data_checks") or {}
        if dc:
            ok = int(dc.get("passed", 0)) == int(dc.get("total", -1))
            self.score_row.addWidget(self._chip(
                f"Data checks {dc.get('passed', 0)}/{dc.get('total', 0)} "
                f"(run {str(dc.get('ts', ''))[:16]})", ok))
        ac = sb.get("answer_checks") or {}
        if ac:
            ok = int(ac.get("FAIL", 0)) == 0 and int(ac.get("ERROR", 0)) == 0
            self.score_row.addWidget(self._chip(
                f"Answer checks  PASS {ac.get('PASS', 0)} · "
                f"FAIL {ac.get('FAIL', 0)} · INFO {ac.get('INFO', 0)} "
                f"(run {str(ac.get('ts', ''))[:16]})", ok))
        if not dc and not ac:
            note = QLabel("No verification runs recorded yet — run "
                          "`python -m eval.run_use_cases` and "
                          "`python test_queries.py` to fill this scoreboard.")
            note.setObjectName("kpisub")
            self.score_row.addWidget(note)
        for wmsg in warns[-3:]:
            chip = QLabel("⚠ " + str(wmsg.get("message", ""))[:110])
            chip.setObjectName("healthbad")
            T2 = _theme(w)
            chip.setStyleSheet(f"color:{T2.get('warn', '#fbbf24')};")
            chip.setWordWrap(True)
            self.score_row.addWidget(chip, stretch=1)
        self.score_row.addStretch(1)

        model = getattr(w.agent, "chat_model", "") or cfg.DEFAULT_CHAT_MODEL
        cycle = T.get("cycle", ["#3d8bfd"])
        db_mb = self._index_size_mb()
        for i, (name, value, sub, raw, fmt) in enumerate([
            ("Model", model.split(":")[0], model, None, None),
            ("Tables", _fmt_int(len(meta)), "loaded for querying",
             len(meta), _fmt_int),
            ("Total rows", _fmt_int(total_rows), "across all tables",
             total_rows, _fmt_int),
            ("PDF sources", _fmt_int(len(docs)),
             f"{_fmt_int(total_pages)} pages", len(docs), _fmt_int),
            ("Text chunks", _fmt_int(chunks), "in the vector index",
             chunks, _fmt_int),
            ("Index size", f"{db_mb:.1f} MB", cfg.INDEX_DIR,
             db_mb, lambda x: f"{x:.1f} MB"),
        ]):
            self.stats_row.addWidget(
                KpiCard(name, value, sub, badge_color=cycle[i % len(cycle)],
                        raw=raw, fmt=fmt), 1)

        self.tables_view.setRowCount(len(meta))
        for i, (name, nrows, ncols) in enumerate(meta):
            code = w._source_code(name, i)
            for c, text in enumerate([code, str(name),
                                      _fmt_int(nrows),
                                      _fmt_int(ncols)]):
                item = QTableWidgetItem(text)
                if c in (2, 3):
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight
                        | Qt.AlignmentFlag.AlignVCenter)
                self.tables_view.setItem(i, c, item)
        self.tables_view.resizeColumnsToContents()

        self.docs_view.setRowCount(len(docs))
        for i, (fname, pages) in enumerate(docs):
            ext = os.path.splitext(fname)[1].lstrip(".").upper() or "PDF"
            for c, text in enumerate([fname, ext, _fmt_int(pages)]):
                item = QTableWidgetItem(text)
                if c == 2:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight
                        | Qt.AlignmentFlag.AlignVCenter)
                self.docs_view.setItem(i, c, item)
        self.docs_view.resizeColumnsToContents()

    # -- house rules editor ---------------------------------------------- #
    def _load_rules(self) -> None:
        br = _bridge(self.window)
        try:
            if br is not None:
                payload = br.rules_get()
                text, path = payload.get("text", ""), payload.get("path", "")
            else:
                import os as _os
                path = getattr(cfg, "HOUSE_RULES_PATH", "") or ""
                text = ""
                if path and _os.path.exists(path):
                    text = open(path, encoding="utf-8").read()
                else:
                    from jarvisman.runtime.house_rules import \
                        _DEFAULT_RULES_PATH
                    if _os.path.exists(_DEFAULT_RULES_PATH):
                        text = open(_DEFAULT_RULES_PATH,
                                    encoding="utf-8").read()
            self.rules_edit.setPlainText(text)
            self.rules_info.setText(
                f"Saved rules apply to the NEXT question — no restart, no "
                f"re-index. File: {path or 'rules.txt'}")
        except Exception as exc:
            self.rules_info.setText(f"Could not load rules: "
                                    f"{type(exc).__name__}")

    def _save_rules(self) -> None:
        text = self.rules_edit.toPlainText()
        br = _bridge(self.window)
        try:
            if br is not None:
                ok = br.rules_save(text)
            else:
                import os as _os
                path = getattr(cfg, "HOUSE_RULES_PATH", "") or "rules.txt"
                _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(text)
                ok = True
        except Exception:
            ok = False
        self.rules_info.setText(
            "Rules saved — active for the next question." if ok
            else "Save FAILED — check permissions / backend.")

    @staticmethod
    def _index_size_mb() -> float:
        total = 0
        try:
            for root, _dirs, files in os.walk(cfg.INDEX_DIR):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
        except Exception:
            return 0.0
        return total / (1024 * 1024)