"""Interactive chart generation with hover tooltips using Plotly."""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Optional


class InteractiveCharts:
    """Generate interactive charts with hover information."""
    
    COLORS = [
        "#4c9aff", "#3fb950", "#d29922", "#f85149", "#a371f7",
        "#39c5cf", "#db61a2", "#e3b341", "#7ee787", "#ffa657"
    ]
    
    @staticmethod
    def create_bar_chart(df: pd.DataFrame, title: str = "") -> go.Figure:
        """Create interactive bar chart."""
        cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not num_cols:
            return None
        
        x_col = cat_cols[0] if cat_cols else None
        y_col = num_cols[0]
        
        if x_col:
            fig = px.bar(df, x=x_col, y=y_col, 
                        title=title or f"{y_col} by {x_col}",
                        hover_data={x_col: True, y_col: ':.2f'},
                        color_discrete_sequence=[InteractiveCharts.COLORS[0]])
        else:
            fig = px.bar(x=df.index, y=df[y_col],
                        title=title or y_col,
                        hover_data={y_col: ':.2f'},
                        color_discrete_sequence=[InteractiveCharts.COLORS[0]])
        
        fig.update_layout(
            plot_bgcolor="#0f1419",
            paper_bgcolor="#0f1419",
            font=dict(color="#e6edf3", size=12),
            title=dict(font=dict(color="#e6edf3")),
            xaxis=dict(gridcolor="#2a323d"),
            yaxis=dict(gridcolor="#2a323d"),
            hovermode='x unified'
        )
        return fig
    
    @staticmethod
    def create_line_chart(df: pd.DataFrame, title: str = "") -> go.Figure:
        """Create interactive line chart."""
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not num_cols:
            return None
        
        fig = go.Figure()
        
        for i, col in enumerate(num_cols[:5]):  # Max 5 lines
            fig.add_trace(go.Scatter(
                y=df[col],
                name=col,
                mode='lines+markers',
                line=dict(color=InteractiveCharts.COLORS[i % len(InteractiveCharts.COLORS)], width=2),
                marker=dict(size=6),
                hovertemplate=f"<b>{col}</b><br>Value: %{{y:.2f}}<extra></extra>"
            ))
        
        fig.update_layout(
            title=title or "Line Chart",
            plot_bgcolor="#0f1419",
            paper_bgcolor="#0f1419",
            font=dict(color="#e6edf3", size=12),
            title_font=dict(color="#e6edf3"),
            xaxis=dict(gridcolor="#2a323d"),
            yaxis=dict(gridcolor="#2a323d"),
            hovermode='x unified'
        )
        return fig
    
    @staticmethod
    def create_area_chart(df: pd.DataFrame, title: str = "") -> go.Figure:
        """Create interactive area chart."""
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not num_cols:
            return None
        
        fig = go.Figure()
        
        for i, col in enumerate(num_cols[:5]):  # Max 5 areas
            fig.add_trace(go.Scatter(
                y=df[col],
                name=col,
                mode='lines',
                line=dict(color=InteractiveCharts.COLORS[i % len(InteractiveCharts.COLORS)], width=2),
                stackgroup='one',
                fillcolor=InteractiveCharts.COLORS[i % len(InteractiveCharts.COLORS)],
                hovertemplate=f"<b>{col}</b><br>Value: %{{y:.2f}}<extra></extra>"
            ))
        
        fig.update_layout(
            title=title or "Area Chart",
            plot_bgcolor="#0f1419",
            paper_bgcolor="#0f1419",
            font=dict(color="#e6edf3", size=12),
            title_font=dict(color="#e6edf3"),
            xaxis=dict(gridcolor="#2a323d"),
            yaxis=dict(gridcolor="#2a323d"),
            hovermode='x unified'
        )
        return fig
    
    @staticmethod
    def create_pie_chart(df: pd.DataFrame, title: str = "") -> go.Figure:
        """Create interactive pie chart."""
        cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not num_cols:
            return None
        
        labels = df[cat_cols[0]].tolist() if cat_cols else [str(i) for i in range(len(df))]
        values = df[num_cols[0]].tolist()
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=InteractiveCharts.COLORS),
            hovertemplate="<b>%{label}</b><br>Value: %{value}<br>Percentage: %{percent}<extra></extra>"
        )])
        
        fig.update_layout(
            title=title or "Pie Chart",
            plot_bgcolor="#0f1419",
            paper_bgcolor="#0f1419",
            font=dict(color="#e6edf3", size=12),
            title_font=dict(color="#e6edf3")
        )
        return fig
    
    @staticmethod
    def create_scatter_chart(df: pd.DataFrame, title: str = "") -> go.Figure:
        """Create interactive scatter chart."""
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(num_cols) < 2:
            return None
        
        fig = px.scatter(df, x=num_cols[0], y=num_cols[1],
                        title=title or f"{num_cols[1]} vs {num_cols[0]}",
                        hover_data={num_cols[0]: ':.2f', num_cols[1]: ':.2f'},
                        color_discrete_sequence=[InteractiveCharts.COLORS[0]])
        
        fig.update_layout(
            plot_bgcolor="#0f1419",
            paper_bgcolor="#0f1419",
            font=dict(color="#e6edf3", size=12),
            title_font=dict(color="#e6edf3"),
            xaxis=dict(gridcolor="#2a323d"),
            yaxis=dict(gridcolor="#2a323d"),
            hovermode='closest'
        )
        return fig

    @staticmethod
    def get_chart(chart_type: str, df: pd.DataFrame, title: str = ""):
        """Get the appropriate chart type with better error handling."""
        chart_type = chart_type.lower().strip()
        
        print(f"DEBUG: get_chart called with type={chart_type}, df.shape={df.shape}")
        
        try:
            if chart_type == 'bar':
                result = InteractiveCharts.create_bar_chart(df, title)
            elif chart_type == 'barh':
                result = InteractiveCharts.create_barh_chart(df, title)
            elif chart_type == 'line':
                result = InteractiveCharts.create_line_chart(df, title)
            elif chart_type == 'area':
                result = InteractiveCharts.create_area_chart(df, title)
            elif chart_type == 'pie':
                result = InteractiveCharts.create_pie_chart(df, title)
            elif chart_type == 'donut':
                result = InteractiveCharts.create_donut_chart(df, title)
            elif chart_type == 'scatter':
                result = InteractiveCharts.create_scatter_chart(df, title)
            else:
                print(f"DEBUG: Unknown chart type: {chart_type}, defaulting to bar")
                result = InteractiveCharts.create_bar_chart(df, title)
            
            if result is None:
                print(f"DEBUG: Chart creation returned None")
                return None
            
            print(f"DEBUG: Chart created successfully")
            return result
            
        except Exception as e:
            print(f"DEBUG: Error in get_chart: {e}")
            import traceback
            print(traceback.format_exc())
            return None