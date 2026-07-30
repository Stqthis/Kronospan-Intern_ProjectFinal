"""Chart rendering helpers. Pure functions over a matplotlib Figure -- no Qt,
no pyplot -- so the caller picks the backend and the logic stays testable."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from jarvisman.runtime import numfmt

CHART_TYPES = ("bar", "barh", "line", "area", "pie", "scatter")

# Warm fallback aligned with the app's default (light) theme; callers such as
# the chart window pass their own active theme, which takes precedence.
DEFAULT_THEME = {
    "bg": "#f5f8fc", "panel": "#ffffff", "text": "#16273c",
    "muted": "#5b6b81", "grid": "#d5e0ee", "accent": "#0f5aa8",
    "cycle": ["#0f5aa8", "#2e7d4f", "#b7791f", "#c0563f", "#7a5cae",
              "#2f8f9e", "#c1567f", "#5b6b81", "#3f7fd1", "#8f7a3f"],
}


# --------------------------------------------------------------------------- #
# Data preparation                                                           #
# --------------------------------------------------------------------------- #
def _to_number(cell: str):
    """Parse a display-formatted cell back to a float, or None."""
    if cell is None:
        return None
    s = str(cell).strip()
    if not s:
        return None
    try:
        return float(numfmt.canon(s))
    except (TypeError, ValueError):
        try:
            return float(s)
        except (TypeError, ValueError):
            return None


def frame_from_table(headers: list[str], rows: list[list[str]]) -> pd.DataFrame:
    """Rebuild a DataFrame from (headers, rows); a column becomes numeric only
    when every non-empty cell parses as a number."""
    headers = [str(h) for h in (headers or [])]
    width = len(headers)
    norm_rows = [(r + [""] * width)[:width] for r in (rows or [])]
    df = pd.DataFrame(norm_rows, columns=headers if width else None)
    for col in df.columns:
        parsed = df[col].map(_to_number)
        non_empty = df[col].map(lambda v: str(v).strip() != "")
        if non_empty.any() and parsed[non_empty].notna().all():
            df[col] = parsed
    return df


def infer_roles(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (categorical_columns, numeric_columns)."""
    numeric, categorical = [], []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(str(c))
        else:
            categorical.append(str(c))
    return categorical, numeric


# --------------------------------------------------------------------------- #
# Chart specification                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class ChartSpec:
    chart_type: str = "bar"
    x: Optional[str] = None            # category / label axis
    y: list = field(default_factory=list)  # measure column(s)
    hue: Optional[str] = None          # optional second category -> grouped/stacked
    top_n: int = 20
    sort_desc: bool = True
    value_labels: bool = True


def default_spec(df: pd.DataFrame, top_n: int = 20,
                 value_labels: bool = True) -> ChartSpec:
    """First categorical as axis, first numeric as measure, bar by default."""
    cats, nums = infer_roles(df)
    x = cats[0] if cats else None
    y = nums[:1] if nums else []
    ctype = "bar" if x is not None else "line"
    return ChartSpec(chart_type=ctype, x=x, y=y, top_n=top_n,
                     value_labels=value_labels)


# --------------------------------------------------------------------------- #
# Rendering                                                                   #
# --------------------------------------------------------------------------- #
def apply_theme(fig, ax, theme: dict) -> None:
    fig.patch.set_facecolor(theme["bg"])
    ax.set_facecolor(theme["bg"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(theme["grid"])
    ax.tick_params(colors=theme["muted"], labelsize=9)
    ax.yaxis.label.set_color(theme["muted"])
    ax.xaxis.label.set_color(theme["muted"])
    ax.title.set_color(theme["text"])
    ax.grid(True, axis="both", color=theme["grid"], linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)


def _aggregate(df: pd.DataFrame, x: str, y: str, hue: Optional[str]):
    """Group duplicate x (and hue) and sum the measure."""
    cols = [c for c in (x, hue) if c]
    if not cols:
        return df[[y]].copy()
    g = df.groupby(cols, dropna=False, sort=False)[y].sum().reset_index()
    return g


def _top_n(frame: pd.DataFrame, x: str, y: str, n: int, desc: bool):
    """Keep the top-n categories by measure; fold the rest into 'Other'."""
    s = frame.groupby(x, dropna=False, sort=False)[y].sum()
    s = s.sort_values(ascending=not desc)
    if n and len(s) > n:
        keep = set(s.index[:n])
        frame = frame.copy()
        frame[x] = frame[x].where(frame[x].isin(keep), other="Other")
        frame = frame.groupby(x, dropna=False, sort=False)[y].sum().reset_index()
    order = frame.groupby(x, dropna=False, sort=False)[y].sum() \
                 .sort_values(ascending=not desc).index.tolist()
    frame[x] = pd.Categorical(frame[x], categories=order, ordered=True)
    return frame.sort_values(x)


def _label_bars(ax, bars, theme, horizontal=False):
    for b in bars:
        v = b.get_width() if horizontal else b.get_height()
        if v is None or v != v:
            continue
        txt = numfmt.fmt(v)
        if horizontal:
            ax.annotate(txt, (b.get_width(), b.get_y() + b.get_height() / 2),
                        xytext=(4, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=8, color=theme["muted"])
        else:
            ax.annotate(txt, (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        va="bottom", ha="center", fontsize=8, color=theme["muted"])


def render(fig, df: pd.DataFrame, spec: ChartSpec,
           theme: Optional[dict] = None, title: str = "") -> None:
    """Draw spec over df onto fig (cleared first). Never raises -- on bad input
    it draws a short message instead."""
    theme = theme or DEFAULT_THEME
    fig.clear()
    ax = fig.add_subplot(111)
    apply_theme(fig, ax, theme)
    try:
        _render_inner(fig, ax, df, spec, theme, title)
    except Exception as exc:  # presentation must never crash the window
        ax.clear()
        apply_theme(fig, ax, theme)
        ax.text(0.5, 0.5, f"Could not draw this chart:\n{type(exc).__name__}",
                ha="center", va="center", color=theme["muted"],
                transform=ax.transAxes, fontsize=10)
    try:
        fig.tight_layout()
    except Exception:
        pass


def _render_inner(fig, ax, df, spec, theme, title):
    cats, nums = infer_roles(df)
    cycle = theme["cycle"]
    ct = spec.chart_type if spec.chart_type in CHART_TYPES else "bar"
    ys = [c for c in (spec.y or nums[:1]) if c in df.columns]
    if not ys and nums:
        ys = nums[:1]

    # ---- pie -------------------------------------------------------------- #
    if ct == "pie" and spec.x and ys:
        frame = _aggregate(df, spec.x, ys[0], None)
        frame = _top_n(frame, spec.x, ys[0], spec.top_n, spec.sort_desc)
        vals = frame[ys[0]].clip(lower=0)
        wedges, _txt, _aut = ax.pie(
            vals, labels=[str(v) for v in frame[spec.x]],
            autopct=lambda p: f"{p:.0f}%", colors=cycle,
            textprops={"color": theme["text"], "fontsize": 9})
        ax.set_title(title or f"{ys[0]} by {spec.x}")
        ax.axis("equal")
        return

    # ---- scatter ---------------------------------------------------------- #
    if ct == "scatter" and len(ys) >= 1 and spec.x and spec.x in nums:
        ax.scatter(df[spec.x], df[ys[0]], color=cycle[0], alpha=0.8, s=28)
        ax.set_xlabel(spec.x); ax.set_ylabel(ys[0])
        ax.set_title(title or f"{ys[0]} vs {spec.x}")
        return

    # ---- bar / barh / line / area with a category axis -------------------- #
    if spec.x and spec.x in df.columns:
        if spec.hue and spec.hue in df.columns and len(ys) == 1:
            # grouped bars: pivot category x hue
            frame = _aggregate(df, spec.x, ys[0], spec.hue)
            pivot = frame.pivot_table(index=spec.x, columns=spec.hue,
                                      values=ys[0], aggfunc="sum", fill_value=0)
            order = pivot.sum(axis=1).sort_values(
                ascending=not spec.sort_desc).index[: spec.top_n or None]
            pivot = pivot.loc[order]
            _plot_frame(ax, pivot, ct, cycle, theme, spec, stacked=(ct == "area"))
            ax.set_title(title or f"{ys[0]} by {spec.x} / {spec.hue}")
            return
        # one or more measures over the category
        frame = df.copy()
        if frame[spec.x].duplicated().any():
            frame = frame.groupby(spec.x, dropna=False, sort=False)[ys].sum() \
                         .reset_index()
        # order by the first measure
        frame = frame.sort_values(ys[0], ascending=not spec.sort_desc)
        if spec.top_n and len(frame) > spec.top_n:
            frame = frame.head(spec.top_n)
        frame = frame.set_index(spec.x)[ys]
        _plot_frame(ax, frame, ct, cycle, theme, spec, stacked=(ct == "area"))
        ax.set_title(title or (f"{ys[0]} by {spec.x}" if len(ys) == 1
                               else f"{', '.join(ys)} by {spec.x}"))
        return

    # ---- no category: line/bar over the row index ------------------------- #
    frame = df[ys] if ys else df.select_dtypes("number")
    _plot_frame(ax, frame, "line" if ct in ("line", "area") else "bar",
                cycle, theme, spec, stacked=(ct == "area"))
    ax.set_title(title or "Values")


def _plot_frame(ax, frame, ct, cycle, theme, spec, stacked=False):
    """Plot a (possibly multi-column) frame as bars/lines/area."""
    cols = list(frame.columns)
    colors = [cycle[i % len(cycle)] for i in range(len(cols))]
    if ct == "barh":
        frame.plot.barh(ax=ax, color=colors, legend=len(cols) > 1, width=0.8)
        if spec.value_labels and len(cols) == 1:
            _label_bars(ax, ax.patches, theme, horizontal=True)
    elif ct in ("line", "area"):
        frame.plot(ax=ax, color=colors, legend=len(cols) > 1, linewidth=2,
                   marker="o", markersize=3)
        if ct == "area":
            for i, c in enumerate(cols):
                ax.fill_between(range(len(frame)), frame[c].values,
                                color=colors[i], alpha=0.15)
        ax.tick_params(axis="x", rotation=30)
    else:  # bar (vertical)
        frame.plot.bar(ax=ax, color=colors, legend=len(cols) > 1, width=0.8,
                       stacked=stacked)
        ax.tick_params(axis="x", rotation=30)
        if spec.value_labels and len(cols) == 1:
            _label_bars(ax, ax.patches, theme, horizontal=False)
    leg = ax.get_legend()
    if leg is not None:
        leg.get_frame().set_facecolor(theme["panel"])
        leg.get_frame().set_edgecolor(theme["grid"])
        for t in leg.get_texts():
            t.set_color(theme["text"])
