import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Optional

class VisualizationManager:
    @staticmethod
    def auto_visualize(df: pd.DataFrame, question: str) -> Optional[go.Figure]:
        """Automatically create appropriate visualization based on data"""
        if df.empty:
            return None
        
        # Simple heuristics for choosing visualization type
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Time series plot
        if datetime_cols and numeric_cols:
            return px.line(df, x=datetime_cols[0], y=numeric_cols[0],
                          title=f"Time Series: {numeric_cols[0]} over {datetime_cols[0]}")
        
        # Bar chart for categorical vs numeric
        elif categorical_cols and numeric_cols and len(df) <= 50:
            return px.bar(df, x=categorical_cols[0], y=numeric_cols[0],
                         title=f"{numeric_cols[0]} by {categorical_cols[0]}")
        
        # Scatter plot for two numeric columns
        elif len(numeric_cols) >= 2:
            return px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                            title=f"{numeric_cols[1]} vs {numeric_cols[0]}")
        
        # Histogram for single numeric column
        elif len(numeric_cols) == 1:
            return px.histogram(df, x=numeric_cols[0],
                              title=f"Distribution of {numeric_cols[0]}")
        
        return None
