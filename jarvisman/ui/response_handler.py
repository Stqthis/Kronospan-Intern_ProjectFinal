"""Enhanced response handler with dropdown chart selector for numerical data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import re
import pandas as pd


@dataclass
class DataExtraction:
    """Container for extracted numerical data from responses."""
    has_data: bool
    data_type: str  # 'table', 'list', 'sequence'
    df: Optional[pd.DataFrame] = None
    title: str = ""
    suggested_charts: list[str] = None

    def __post_init__(self):
        if self.suggested_charts is None:
            self.suggested_charts = []


class ResponseDataExtractor:
    """Extracts numerical data from AI responses for charting."""

    CHART_TYPES = ["bar", "barh", "line", "area", "pie", "scatter", "donut"]

    @staticmethod
    def detect_numerical_data(response_text: str) -> DataExtraction:
        """
        Detect if response contains numerical data suitable for charting.
        Returns DataExtraction with extracted data if found.
        """
        # Try to find markdown tables
        df = ResponseDataExtractor._extract_markdown_table(response_text)
        if df is not None and not df.empty:
            suggested = ResponseDataExtractor._suggest_charts(df)
            return DataExtraction(
                has_data=True,
                data_type='table',
                df=df,
                title="Response Data",
                suggested_charts=suggested
            )

        # Try to find CSV-like data
        df = ResponseDataExtractor._extract_csv_data(response_text)
        if df is not None and not df.empty:
            suggested = ResponseDataExtractor._suggest_charts(df)
            return DataExtraction(
                has_data=True,
                data_type='table',
                df=df,
                title="Response Data",
                suggested_charts=suggested
            )

        # Try to find numbered lists with values
        df = ResponseDataExtractor._extract_list_data(response_text)
        if df is not None and not df.empty:
            suggested = ResponseDataExtractor._suggest_charts(df)
            return DataExtraction(
                has_data=True,
                data_type='list',
                df=df,
                title="Response Data",
                suggested_charts=suggested
            )

        return DataExtraction(has_data=False, data_type='none')

    @staticmethod
    def _extract_markdown_table(text: str) -> Optional[pd.DataFrame]:
        """Extract data from markdown tables."""
        try:
            lines = text.split('\n')
            table_start = -1
            
            for i, line in enumerate(lines):
                if '|' in line and any(c.isalnum() for c in line):
                    table_start = i
                    break
            
            if table_start == -1:
                return None

            headers = []
            rows = []
            
            for i in range(table_start, len(lines)):
                line = lines[i].strip()
                if not line or '|' not in line:
                    break
                
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                
                if all(c.replace('-', '').replace(':', '').replace(' ', '') == '' 
                    for c in cells):
                    continue
                
                if not headers:
                    headers = cells
                else:
                    rows.append(cells)
            
            if headers and rows:
                df = pd.DataFrame(rows, columns=headers)
                
                # Convert numeric columns to actual numbers
                for col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        pass
                
                return df
        
        except Exception:
            pass
        
        return None

    @staticmethod
    def _extract_csv_data(text: str) -> Optional[pd.DataFrame]:
        """Extract comma or tab-separated data."""
        try:
            from io import StringIO
            
            lines = text.split('\n')
            csv_lines = []
            
            for line in lines:
                line = line.strip()
                if line and (',' in line or '\t' in line):
                    csv_lines.append(line)
            
            if len(csv_lines) < 2:
                return None
            
            csv_text = '\n'.join(csv_lines[:20])
            df = pd.read_csv(StringIO(csv_text), sep='[,\t]', engine='python')
            
            if not df.empty and len(df) > 0:
                return df
        
        except Exception:
            pass
        
        return None

    @staticmethod
    def _extract_list_data(text: str) -> Optional[pd.DataFrame]:
        """Extract numbered or bulleted list data with values."""
        try:
            lines = text.split('\n')
            data = []
            
            for line in lines:
                match = re.search(
                    r'[\d\-\*][\.\)]?\s*(.+?)[:\-]\s*(\d+(?:\.\d+)?)', line)
                if match:
                    label = match.group(1).strip()
                    value = float(match.group(2))
                    data.append({'Item': label, 'Value': value})
            
            if len(data) >= 2:
                return pd.DataFrame(data)
        
        except Exception:
            pass
        
        return None

    @staticmethod
    @staticmethod
    def _suggest_charts(df: pd.DataFrame) -> list[str]:
        """Always return all 7 chart types for maximum flexibility."""
        # Show all charts regardless of data structure
        # Users can choose what works best for their data
        return ['bar', 'barh', 'line', 'area', 'pie', 'donut', 'scatter']
    @staticmethod
    def get_all_chart_types() -> list[str]:
        """Return all supported chart types."""
        return ResponseDataExtractor.CHART_TYPES
