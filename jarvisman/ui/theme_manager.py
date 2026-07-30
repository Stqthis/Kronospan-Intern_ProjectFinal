"""Theme management: four named palettes (Dark / Light / Emerald / Purple).

Every palette carries the same keys, so any component that reads the theme
dict works under all of them. ``get_theme`` keeps its old boolean signature
for back-compat; new code should use ``ThemeManager.get(name)``.
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
        "ok": "#2e7d4f",
        "err": "#c0392b",
        "warn": "#b7791f",
        "code_bg": "#eef3fa",
        "grid": "#d5e0ee",
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
        "accent": "#8b5cf6",
        "accent_dim": "#2b1f4d",
        "accent_text": "#ffffff",
        "accent_hover": "#7745e0",
        "bot": "#161027",
        "ok": "#34d399",
        "err": "#f87171",
        "warn": "#fbbf24",
        "code_bg": "#0b0816",
        "grid": "#2c2347",
        "cycle": ["#8b5cf6", "#ec4899", "#a78bfa", "#f472b6", "#c084fc",
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
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {theme['accent_hover']};
            }}
            QPushButton:pressed {{
                opacity: 0.8;
            }}
        """
