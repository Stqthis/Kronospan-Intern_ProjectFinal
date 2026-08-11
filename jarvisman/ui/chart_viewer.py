"""Chart viewer widget for displaying interactive Plotly charts."""
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl
import plotly.graph_objects as go
import tempfile
import os


class ChartViewer(QWidget):
    """Display interactive Plotly charts in PyQt6."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.browser = QWebEngineView()
        layout = QVBoxLayout()
        layout.addWidget(self.browser)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.temp_file = None
    
    def display_chart(self, fig: go.Figure) -> None:
        """Display a Plotly figure."""
        try:
            # Create temporary HTML file
            if self.temp_file:
                try:
                    os.unlink(self.temp_file)
                except:
                    pass
            
            # Write HTML to temp file
            temp_dir = tempfile.gettempdir()
            self.temp_file = os.path.join(temp_dir, 'jarvis_chart.html')
            
            fig.write_html(self.temp_file)
            
            # Load in browser
            self.browser.load(QUrl.fromLocalFile(self.temp_file))
            
        except Exception as e:
            print(f"Error displaying chart: {e}")
    
    def clear(self) -> None:
        """Clear the chart."""
        self.browser.setHtml("")