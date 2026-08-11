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
    "ramp": ["#79aee0", "#0b3f78"],
}


# --------------------------------------------------------------------------- #
# Theme-aligned colours                                                       #
# --------------------------------------------------------------------------- #
def _hex_to_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(round(c)))) for c in rgb)


def series_colors(theme: dict, n: int) -> list:
    """n colours sampled across the theme's sequential ramp (light -> deep), so
    a single-series bar chart or a pie reads as one coherent theme hue (blue for
    Dark/Light) with each bar/slice a distinct shade -- instead of the multi-hue
    ``cycle``, which is reserved for charts with several series to tell apart."""
    lo, hi = (theme.get("ramp") or DEFAULT_THEME["ramp"])[:2]
    a, b = _hex_to_rgb(lo), _hex_to_rgb(hi)
    if n <= 1:
        return [hi]
    return [_rgb_to_hex(tuple(a[k] + (b[k] - a[k]) * (i / (n - 1))
                              for k in range(3)))
            for i in range(n)]


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


def _looks_like_id(name: str) -> bool:
    n = str(name).strip().lower()
    return (n in ("id", "code", "index", "no", "no.", "ref", "key")
            or n.endswith("_id") or n.endswith(" id") or n.endswith("code")
            or n.endswith(" no") or n.endswith("_no"))


def infer_roles(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (categorical_columns, numeric_columns)."""
    numeric, categorical = [], []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(str(c))
        else:
            categorical.append(str(c))
    return categorical, numeric


def _is_timelike(df: pd.DataFrame, col: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        return True
    n = str(col).lower()
    return any(k in n for k in ("date", "month", "year", "quarter", "period"))


def _best_x(df: pd.DataFrame, cats: list, nums: list) -> Optional[str]:
    """Pick the label axis a human would: prefer a time column; otherwise the
    categorical with a sensible number of distinct values (2..40), skipping
    ID-like and all-unique columns. Falls back to the first categorical."""
    n = len(df)
    # a time column wins (line chart over time reads naturally)
    for c in list(cats) + list(nums):
        if _is_timelike(df, c) and df[c].nunique(dropna=False) > 1:
            return str(c)
    scored = []
    for c in cats:
        if _looks_like_id(c):
            continue
        u = df[c].nunique(dropna=False)
        if u <= 1 or u == n:          # constant or unique-per-row -> useless axis
            continue
        # closeness to an ideal ~12 distinct categories
        scored.append((abs(u - 12), str(c)))
    if scored:
        return sorted(scored)[0][1]
    return cats[0] if cats else None


def _best_y(df: pd.DataFrame, nums: list, x: Optional[str]) -> list:
    """Pick the measure: a numeric column that actually varies and isn't an
    ID/year, preferring the one with the widest spread of values."""
    best, best_score = None, -1.0
    for c in nums:
        if c == x or _looks_like_id(c) or _is_timelike(df, c):
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty or s.nunique() <= 1:
            continue
        score = float(s.abs().sum())      # bias toward real magnitude measures
        if score > best_score:
            best, best_score = str(c), score
    if best:
        return [best]
    # fall back to any numeric that isn't the axis
    rest = [c for c in nums if c != x]
    return rest[:1]


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
    """Auto-pick a chart a non-expert would consider correct: a sensible label
    axis, a real measure, and a chart type that fits the shape of the data."""
    cats, nums = infer_roles(df)
    x = _best_x(df, cats, nums)
    y = _best_y(df, nums, x)
    # choose a chart type from the data shape
    if x is not None and _is_timelike(df, x):
        ctype = "line"                       # a measure over time
    elif x is None:
        ctype = "line"                       # nothing categorical to group by
    else:
        distinct = df[x].nunique(dropna=False)
        if len(y) == 1 and 3 <= distinct <= 4:
            ctype = "pie"                    # a few parts of a whole
        elif distinct > 12:
            ctype = "barh"                   # many labels -> horizontal, readable
        else:
            ctype = "bar"
    return ChartSpec(chart_type=ctype, x=x, y=y, top_n=top_n,
                     value_labels=value_labels)


def describe_spec(spec: ChartSpec) -> str:
    """A plain-language, non-technical sentence describing what the chart shows,
    so the user never has to reason about 'x' and 'y'."""
    kind = {"bar": "bar chart", "barh": "horizontal bar chart", "line": "line chart",
            "area": "area chart", "pie": "pie chart",
            "scatter": "scatter plot"}.get(spec.chart_type, "chart")
    measure = spec.y[0] if spec.y else "value"
    if spec.chart_type == "scatter":
        return f"Showing {measure} against {spec.x} as a {kind}."
    if spec.x is None:
        return f"Showing {measure} as a {kind}."
    lead = "trend of" if spec.chart_type == "line" else "total"
    by = f" for each {spec.x}"
    if spec.hue:
        by += f", split by {spec.hue}"
    return f"Showing the {lead} {measure}{by} as a {kind}."


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
    # Value labels only help when they fit. With many entities the bars get
    # narrow and adjacent numbers collide, so above a threshold we drop the
    # labels entirely (the axis still shows magnitude) rather than print an
    # unreadable pile of overlapping text. Vertical bars crowd much sooner than
    # horizontal ones, where labels stack down the side.
    real = [b for b in bars if (b.get_width() if horizontal
                                else b.get_height()) not in (None,)]
    limit = 30 if horizontal else 10
    if len(real) > limit:
        return
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
    theme = {**DEFAULT_THEME, **(theme or {})}
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
            autopct=lambda p: f"{p:.0f}%",
            colors=series_colors(theme, len(vals)),
            textprops={"color": theme["text"], "fontsize": 9})
        ax.set_title(title or f"{ys[0]} by {spec.x}")
        ax.axis("equal")
        return

    # ---- scatter ---------------------------------------------------------- #
    if ct == "scatter" and len(ys) >= 1 and spec.x and spec.x in nums:
        ax.scatter(df[spec.x], df[ys[0]], color=theme.get("accent", cycle[0]),
                   alpha=0.8, s=28)
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
    """Plot a (possibly multi-column) frame as bars/lines/area.

    One series -> the theme ramp (each bar a distinct shade of the theme hue;
    line/area in the accent), so the chart reads as one coherent colour. Several
    series -> the multi-hue ``cycle`` so they stay tellable apart in the legend.
    """
    cols = list(frame.columns)
    single = len(cols) == 1
    accent = theme.get("accent", cycle[0])
    colors = [accent] if single else [cycle[i % len(cycle)]
                                      for i in range(len(cols))]
    if ct == "barh":
        frame.plot.barh(ax=ax, color=colors, legend=len(cols) > 1, width=0.8)
        if single:
            _shade_bars(ax, theme)
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
        if single:
            _shade_bars(ax, theme)
        ax.tick_params(axis="x", rotation=30)
        if spec.value_labels and len(cols) == 1:
            _label_bars(ax, ax.patches, theme, horizontal=False)
    leg = ax.get_legend()
    if leg is not None:
        leg.get_frame().set_facecolor(theme["panel"])
        leg.get_frame().set_edgecolor(theme["grid"])
        for t in leg.get_texts():
            t.set_color(theme["text"])


def _shade_bars(ax, theme) -> None:
    """Recolour a single-series bar plot's bars across the theme ramp, so each
    bar is a distinct on-theme shade. Done on the patches after pandas plots,
    because pandas colours per-column (one colour for a single series)."""
    from matplotlib.patches import Rectangle
    bars = [p for p in ax.patches if isinstance(p, Rectangle)]
    shades = series_colors(theme, len(bars))
    for patch, col in zip(bars, shades):
        patch.set_facecolor(col)