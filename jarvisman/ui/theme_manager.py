"""Theme management for light/dark mode."""


class ThemeManager:
    """Manage light and dark themes."""
    
    DARK_THEME = {
        "bg": "#0f1826",
        "panel": "#172336",
        "panel2": "#1e2d44",
        "border": "#2c3d59",
        "text": "#e9f0fa",
        "muted": "#94a5bd",
        "accent": "#4d94e8",
        "accent_dim": "#1d3f63",
        "accent_text": "#ffffff",
        "accent_hover": "#2f7bd4",
        "bot": "#172336",
        "ok": "#6fbf8a",
        "err": "#e0776a",
        "warn": "#d9a441",
        "code_bg": "#0a1220",
        "grid": "#2c3d59",
        "cycle": ["#4d94e8", "#6fbf8a", "#d9a441", "#e0776a", "#b58bc4",
                  "#5fb0a8", "#d98ca8", "#c9a96a", "#8fbf7f", "#7fa8d9"],
    }

    LIGHT_THEME = {
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
