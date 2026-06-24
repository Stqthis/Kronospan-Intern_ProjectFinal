"""Chart selector dropdown widget - allows users to select and visualize chart types."""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QComboBox, QPushButton, QVBoxLayout, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
import pandas as pd


class ChartTypeSelector(QWidget):
    """
    A dropdown selector widget for choosing chart types and visualizing data.
    Emits signal when user selects a chart type to visualize.
    """
    
    chart_selected = pyqtSignal(str)
    visualization_requested = pyqtSignal(str, pd.DataFrame)

    def __init__(self, parent=None, theme: dict = None):
        super().__init__(parent)
        self.theme = theme or {
            "bg": "#171c24", "panel": "#1f262f", "border": "#2a323d",
            "text": "#e6edf3", "accent": "#4c9aff", "muted": "#8b97a5"
        }
        self.current_df = None
        self.available_charts = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the selector UI."""
        layout = QHBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        label = QLabel("📊 Visualize as:")
        label.setStyleSheet(f"color: {self.theme['text']}; font-weight: bold;")

        self.chart_combo = QComboBox()
        self.chart_combo.setStyleSheet(f"""
            QComboBox {{
                background: {self.theme['panel']};
                color: {self.theme['text']};
                border: 1px solid {self.theme['border']};
                border-radius: 6px;
                padding: 6px 10px;
                min-width: 120px;
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QComboBox QAbstractItemView {{
                background: {self.theme['panel']};
                color: {self.theme['text']};
                border: 1px solid {self.theme['border']};
                selection-background-color: {self.theme['accent']};
            }}
        """)
        self.chart_combo.currentTextChanged.connect(self._on_chart_selected)

        self.visualize_btn = QPushButton("Generate Chart")
        self.visualize_btn.setStyleSheet(f"""
            QPushButton {{
                background: {self.theme['accent']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #5b9cff;
            }}
            QPushButton:pressed {{
                background: #3d7acc;
            }}
            QPushButton:disabled {{
                background: {self.theme['muted']};
                color: {self.theme['border']};
            }}
        """)
        self.visualize_btn.clicked.connect(self._on_visualize_clicked)
        self.visualize_btn.setEnabled(False)

        layout.addWidget(label)
        layout.addWidget(self.chart_combo, 1)
        layout.addWidget(self.visualize_btn)
        layout.addStretch()

        self.setLayout(layout)
        self.setStyleSheet(f"background: {self.theme['bg']}; border: none;")

    def set_available_charts(self, chart_types: list[str]) -> None:
        """Set the available chart types to display in dropdown."""
        self.available_charts = chart_types or []
        
        self.chart_combo.clear()
        
        if not chart_types:
            self.chart_combo.addItem("No charts available")
            self.visualize_btn.setEnabled(False)
            return

        chart_labels = {
            'bar': '📊 Bar Chart',
            'barh': '📊 Horizontal Bar',
            'pie': '🥧 Pie Chart',
            'donut': '🍩 Donut Chart',
            'line': '📈 Line Chart',
            'area': '📈 Area Chart',
            'scatter': '• Scatter Plot',
        }

        self.chart_combo.blockSignals(True)
        for chart_type in chart_types:
            label = chart_labels.get(chart_type, chart_type.capitalize())
            self.chart_combo.addItem(label, chart_type)
        self.chart_combo.blockSignals(False)
        
        self.visualize_btn.setEnabled(True)

    def set_dataframe(self, df: pd.DataFrame) -> None:
        """Store reference to the dataframe for charting."""
        self.current_df = df

    def _on_chart_selected(self, text: str) -> None:
        """Handle chart type selection."""
        self.chart_selected.emit(text)

    def _on_visualize_clicked(self) -> None:
        """Handle visualize button click."""
        if self.current_df is None:
            return
        
        chart_type = self.chart_combo.currentData()
        if chart_type:
            self.visualization_requested.emit(chart_type, self.current_df)

    def get_selected_chart_type(self) -> str:
        """Get the currently selected chart type."""
        return self.chart_combo.currentData() or ""

    def set_theme(self, theme: dict) -> None:
        """Update the theme colors."""
        self.theme = theme
        self._setup_ui()

    def setVisible(self, visible: bool) -> None:
        """Override visibility with proper styling."""
        super().setVisible(visible)
        if visible and self.current_df is None:
            self.visualize_btn.setEnabled(False)
        elif visible:
            self.visualize_btn.setEnabled(True)


class DataSummaryPanel(QFrame):
    """
    Display a summary of detected data in the response.
    Shows column names, data types, and row count.
    """

    def __init__(self, parent=None, theme: dict = None):
        super().__init__(parent)
        self.theme = theme or {
            "bg": "#171c24", "panel": "#1f262f", "border": "#2a323d",
            "text": "#e6edf3", "accent": "#4c9aff", "muted": "#8b97a5"
        }
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setLineWidth(1)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the panel UI."""
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        self.info_label = QLabel("Data detected: Ready for visualization")
        self.info_label.setStyleSheet(f"""
            color: {self.theme['muted']};
            font-size: 11px;
        """)
        layout.addWidget(self.info_label)

        self.setLayout(layout)
        self.setStyleSheet(f"""
            QFrame {{
                background: {self.theme['panel']};
                border: 1px solid {self.theme['border']};
                border-radius: 6px;
            }}
        """)

    def update_summary(self, df) -> None:
        """Update summary based on dataframe."""
        if df is None or df.empty:
            self.info_label.setText("No data available")
            return

        num_rows = len(df)
        num_cols = len(df.columns)
        numeric_cols = len(df.select_dtypes(include=['number']).columns)
        
        summary = (
            f"📋 {num_rows} rows × {num_cols} columns "
            f"({numeric_cols} numeric)"
        )
        self.info_label.setText(summary)

    def set_theme(self, theme: dict) -> None:
        """Update the theme colors."""
        self.theme = theme
        self._setup_ui()
