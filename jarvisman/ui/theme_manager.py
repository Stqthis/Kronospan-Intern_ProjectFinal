"""Theme management for light/dark mode."""


class ThemeManager:
    """Manage light and dark themes."""
    
    DARK_THEME = {
        "bg": "#0f1419",
        "panel": "#171c24",
        "panel2": "#1f262f",
        "border": "#2a323d",
        "text": "#e6edf3",
        "muted": "#8b97a5",
        "accent": "#4c9aff",
        "accent_dim": "#2d5a88",
        "bot": "#1f262f",
        "ok": "#3fb950",
        "err": "#f85149",
        "warn": "#d29922",
        "code_bg": "#11151a",
        "grid": "#2a323d",
    }
    
    LIGHT_THEME = {
        "bg": "#ffffff",
        "panel": "#f6f8fa",
        "panel2": "#eaeef2",
        "border": "#d0d7de",
        "text": "#24292f",
        "muted": "#57606a",
        "accent": "#0969da",
        "accent_dim": "#54aeff",
        "bot": "#f6f8fa",
        "ok": "#1a7f37",
        "err": "#da3633",
        "warn": "#9e6a03",
        "code_bg": "#f6f8fa",
        "grid": "#d0d7de",
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
                background-color: {theme['accent_dim']};
            }}
            QPushButton:pressed {{
                opacity: 0.8;
            }}
        """
