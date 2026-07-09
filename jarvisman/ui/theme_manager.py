"""Theme management for light/dark mode."""


class ThemeManager:
    """Manage light and dark themes."""
    
    DARK_THEME = {
        "bg": "#262624",
        "panel": "#30302e",
        "panel2": "#3a3a37",
        "border": "#454440",
        "text": "#f5f4ee",
        "muted": "#b0aea3",
        "accent": "#d97757",
        "accent_dim": "#5c3d30",
        "accent_text": "#ffffff",
        "accent_hover": "#c15f3c",
        "bot": "#30302e",
        "ok": "#7fb069",
        "err": "#e0776a",
        "warn": "#d9a441",
        "code_bg": "#1c1c1a",
        "grid": "#454440",
        "cycle": ["#d97757", "#7fb069", "#d9a441", "#6b9bd1", "#b58bc4",
                  "#5fb0a8", "#d98ca8", "#c9a96a", "#8fbf7f", "#e0a06b"],
    }

    LIGHT_THEME = {
        "bg": "#faf9f5",
        "panel": "#ffffff",
        "panel2": "#f0eee7",
        "border": "#e3e0d6",
        "text": "#2b2a27",
        "muted": "#6f6b60",
        "accent": "#c65d3b",
        "accent_dim": "#f0d9cf",
        "accent_text": "#ffffff",
        "accent_hover": "#a84a2e",
        "bot": "#ffffff",
        "ok": "#3d8b52",
        "err": "#c0392b",
        "warn": "#b7791f",
        "code_bg": "#f4f2ec",
        "grid": "#e3e0d6",
        "cycle": ["#c65d3b", "#3d8b52", "#b7791f", "#3a6ea5", "#8a5cae",
                  "#2f9488", "#c1567f", "#a8863f", "#5a9e64", "#d4834f"],
    }
    
    @staticmethod
    def get_theme(is_dark: bool = True) -> dict:
        """Get theme based on preference."""
        return ThemeManager.DARK_THEME if is_dark else ThemeManager.LIGHT_THEME
    
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
