"""Theme management: four named palettes (Dark / Light / Emerald / Purple).

Every palette carries the same keys, so any component that reads the theme
dict works under all of them. ``get_theme`` keeps its old boolean signature
for back-compat; new code should use ``ThemeManager.get(name)``.

Palette contract -- every key here must exist in EVERY palette, because the
stylesheet names each surface explicitly rather than letting Qt fall back to
the desktop palette (a dark desktop was painting black table rows under the
Light theme's near-black text, making figures unreadable until selected).

  row_hover  row tint under the cursor; distinct from panel and panel2
  sel_bg     selected-row background
  sel_text   text on a selected row -- AA contrast against sel_bg

Contrast: body text, muted text and status colours all clear WCAG AA (4.5:1)
against the surfaces they are painted on, in all four palettes. Selection
pairs clear 6:1. Figures on a projector or a factory-floor monitor stay
legible.
"""


class ThemeManager:
    """Named UI palettes."""

    DARK_THEME = {
        "name": "Dark",
        "bg": "#0b0f16",
        "panel": "#111624",
        "panel2": "#161d2e",
        "border": "#222b40",
        "text": "#e8ecf4",
        "muted": "#8f9ab0",
        "accent": "#3d8bfd",
        "accent_dim": "#16294a",
        "accent_text": "#ffffff",
        "accent_hover": "#2f6fd8",
        "bot": "#111624",
        "ok": "#34d399",
        "err": "#f87171",
        "warn": "#fbbf24",
        "code_bg": "#0a0e18",
        "grid": "#222b40",
        "row_hover": "#1c2438",
        "sel_bg": "#1e4a86",
        "sel_text": "#ffffff",
        "cycle": ["#3d8bfd", "#22c55e", "#eab308", "#a855f7", "#ec4899",
                  "#14b8a6", "#f97316", "#64748b", "#60a5fa", "#84cc16"],
    }

    LIGHT_THEME = {
        "name": "Light",
        "bg": "#f5f8fc",
        "panel": "#ffffff",
        "panel2": "#e9f0f8",
        "border": "#d5e0ee",
        "text": "#16273c",
        "muted": "#5b6b81",
        "accent": "#0f5aa8",
        "accent_dim": "#d8e6f5",
        "accent_text": "#ffffff",
        "accent_hover": "#0c4886",
        "bot": "#ffffff",
        # darkened from #2e7d4f / #b7791f: the originals sat at 4.39 and 3.17
        # against panel2, under the 4.5 AA floor for body text.
        "ok": "#1f6b40",
        "err": "#c0392b",
        "warn": "#8a5a00",
        "code_bg": "#eef3fa",
        "grid": "#d5e0ee",
        "row_hover": "#dce8f6",
        "sel_bg": "#0f5aa8",
        "sel_text": "#ffffff",
        "cycle": ["#0f5aa8", "#2e7d4f", "#b7791f", "#c0563f", "#7a5cae",
                  "#2f8f9e", "#c1567f", "#5b6b81", "#3f7fd1", "#8f7a3f"],
    }

    EMERALD_THEME = {
        "name": "Emerald",
        "bg": "#07130f",
        "panel": "#0c1d17",
        "panel2": "#122720",
        "border": "#1d3a2f",
        "text": "#e6f4ee",
        "muted": "#8fb0a3",
        "accent": "#10b981",
        "accent_dim": "#0c3d2e",
        "accent_text": "#ffffff",
        "accent_hover": "#0d9668",
        "bot": "#0c1d17",
        "ok": "#34d399",
        "err": "#f87171",
        "warn": "#fbbf24",
        "code_bg": "#06100c",
        "grid": "#1d3a2f",
        "row_hover": "#17332a",
        # white on the raw accent was 2.54:1 -- unreadable. Deepened.
        "sel_bg": "#0a5f46",
        "sel_text": "#ffffff",
        "cycle": ["#10b981", "#34d399", "#a3e635", "#2dd4bf", "#4ade80",
                  "#facc15", "#38bdf8", "#86efac", "#5eead4", "#bef264"],
    }

    PURPLE_THEME = {
        "name": "Purple",
        "bg": "#0e0a1a",
        "panel": "#161027",
        "panel2": "#1d1633",
        "border": "#2c2347",
        "text": "#ece8f7",
        "muted": "#9c93b8",
        "accent": "#a78bfa",
        "accent_dim": "#2b1f4d",
        "accent_text": "#ffffff",
        "accent_hover": "#8b5cf6",
        "bot": "#161027",
        "ok": "#34d399",
        "err": "#f87171",
        "warn": "#fbbf24",
        "code_bg": "#0b0816",
        "grid": "#2c2347",
        "row_hover": "#251c40",
        "sel_bg": "#4c2f9e",
        "sel_text": "#ffffff",
        "cycle": ["#a78bfa", "#ec4899", "#38bdf8", "#f472b6", "#c084fc",
                  "#e879f9", "#818cf8", "#fb7185", "#d8b4fe", "#f0abfc"],
    }

    THEMES = {
        "Dark": DARK_THEME,
        "Light": LIGHT_THEME,
        "Emerald": EMERALD_THEME,
        "Purple": PURPLE_THEME,
    }

    @classmethod
    def get(cls, name: str) -> dict:
        """Palette by name; unknown names fall back to Dark."""
        return cls.THEMES.get(name, cls.DARK_THEME)

    @classmethod
    def names(cls) -> list:
        return list(cls.THEMES)

    @classmethod
    def next_name(cls, name: str) -> str:
        """The theme after ``name``, cycling (used by the quick-toggle)."""
        order = cls.names()
        try:
            return order[(order.index(name) + 1) % len(order)]
        except ValueError:
            return order[0]

    @staticmethod
    def get_theme(is_dark: bool = True) -> dict:
        """Back-compat boolean accessor."""
        return (ThemeManager.DARK_THEME if is_dark
                else ThemeManager.LIGHT_THEME)

    @staticmethod
    def get_chat_stylesheet(theme: dict) -> str:
        """Get stylesheet for chat display."""
        return f"""
            QTextBrowser {{
                background-color: {theme['bg']};
                color: {theme['text']};
                border: 1px solid {theme['border']};
                padding: 10px;
                font-family: 'Segoe UI', Arial, sans-serif;
            }}
        """

    @staticmethod
    def get_button_stylesheet(theme: dict) -> str:
        """Get stylesheet for buttons."""
        return f"""
            QPushButton {{
                background-color: {theme['accent']};
                color: {theme['accent_text']};
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {theme['accent_hover']};
            }}
            QPushButton:pressed {{
                background-color: {theme['accent_hover']};
            }}
        """

    @staticmethod
    def table_stylesheet(theme: dict) -> str:
        """Stylesheet for a standalone QTableWidget (TableWindow).

        Named explicitly rather than inherited: a table that only sets `color`
        picks its background up from the desktop palette, which is how dark
        text ended up on black rows under the Light theme.
        """
        T = theme or ThemeManager.DARK_THEME
        g = T.get
        return f"""
            QTableWidget {{ background: {g('panel', '#fff')};
                alternate-background-color: {g('panel2', '#f4f4f4')};
                color: {g('text', '#222')};
                gridline-color: {g('border', '#ddd')};
                border: 1px solid {g('border', '#ddd')};
                border-radius: 8px; outline: none;
                selection-background-color: {g('sel_bg', g('accent', '#3d8bfd'))};
                selection-color: {g('sel_text', '#ffffff')}; }}
            QTableWidget::item {{ background: transparent;
                color: {g('text', '#222')}; padding: 5px 8px; border: none; }}
            QTableWidget::item:hover {{
                background: {g('row_hover', g('panel2', '#f4f4f4'))};
                color: {g('text', '#222')}; }}
            QTableWidget::item:selected,
            QTableWidget::item:selected:!active {{
                background: {g('sel_bg', g('accent', '#3d8bfd'))};
                color: {g('sel_text', '#ffffff')}; }}
            QHeaderView {{ background: {g('panel2', '#eee')}; border: none; }}
            QHeaderView::section {{ background: {g('panel2', '#eee')};
                color: {g('text', '#222')}; padding: 7px 8px; border: none;
                border-right: 1px solid {g('border', '#ddd')};
                border-bottom: 1px solid {g('border', '#ddd')};
                font-weight: 600; }}
            QHeaderView::section:hover {{
                background: {g('accent_dim', g('panel2', '#eee'))}; }}
            QTableCornerButton::section {{ background: {g('panel2', '#eee')};
                border: none;
                border-bottom: 1px solid {g('border', '#ddd')}; }}
        """