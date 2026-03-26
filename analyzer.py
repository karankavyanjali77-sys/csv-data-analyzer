"""
All data-processing and chart-building logic, separated from the Streamlit UI.
Functions are pure (no st.* calls) so they are independently testable.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional


# ─── OVERVIEW ─────────────────────────────────────────────────────────────────

def compute_overview(df: pd.DataFrame) -> dict:
    """Return basic shape / health metrics for the dataset."""
    return {
        "rows":       df.shape[0],
        "cols":       df.shape[1],
        "missing":    int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "memory_kb":  round(df.memory_usage(deep=True).sum() / 1024, 1),
    }


def compute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted DataFrame of columns with missing values."""
    missing_count = df.isna().sum()
    missing_pct   = (df.isna().mean() * 100).round(2)
    result = pd.DataFrame({
        "Column":        missing_count.index,
        "Missing Count": missing_count.values,
        "Missing %":     missing_pct.values,
    })
    result = result[result["Missing Count"] > 0].sort_values(
        "Missing Count", ascending=False
    ).reset_index(drop=True)
    return result


def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    IQR-based outlier detection for all numeric columns.
    Returns a DataFrame summarising outlier counts per column.
    """
    num_cols = get_numeric_cols(df)
    records = []
    for col in num_cols:
        series = df[col].dropna()
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr     = q3 - q1
        lower   = q1 - 1.5 * iqr
        upper   = q3 + 1.5 * iqr
        n_out   = int(((series < lower) | (series > upper)).sum())
        if n_out > 0:
            records.append({
                "Column":         col,
                "Outlier Count":  n_out,
                "Outlier %":      round(n_out / len(series) * 100, 2),
                "Lower Fence":    round(lower, 4),
                "Upper Fence":    round(upper, 4),
            })
    return pd.DataFrame(records).sort_values("Outlier Count", ascending=False) \
           if records else pd.DataFrame()


def compute_data_quality_score(df: pd.DataFrame) -> int:
    """
    Score from 0–100 based on:
      - Missing values  (40 pts)
      - Duplicates      (30 pts)
      - Column coverage (30 pts — penalise columns that are >50% missing)
    """
    total_cells    = df.shape[0] * df.shape[1]
    missing_ratio  = df.isna().sum().sum() / max(total_cells, 1)
    dup_ratio      = df.duplicated().sum() / max(len(df), 1)
    bad_cols_ratio = (df.isna().mean() > 0.5).sum() / max(df.shape[1], 1)

    score = (
        40 * (1 - missing_ratio)
        + 30 * (1 - dup_ratio)
        + 30 * (1 - bad_cols_ratio)
    )
    return max(0, min(100, int(round(score))))


# ─── COLUMN HELPERS ───────────────────────────────────────────────────────────

def get_numeric_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include="number").columns.tolist()


def get_categorical_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


# ─── CHARTS ───────────────────────────────────────────────────────────────────

_PALETTE = "#6c63ff"   # brand accent

def build_histogram(df: pd.DataFrame, col: str):
    fig = px.histogram(
        df, x=col,
        title=f"Distribution of {col}",
        color_discrete_sequence=[_PALETTE],
        marginal="box",
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="#1a1a2e",
        title_font_size=14,
    )
    return fig


def build_line(df: pd.DataFrame, col: str):
    fig = px.line(
        df, y=col,
        title=f"{col} over index",
        color_discrete_sequence=[_PALETTE],
    )
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white", font_color="#1a1a2e",
    )
    return fig


def build_box(df: pd.DataFrame, col: str):
    fig = px.box(
        df, y=col,
        title=f"Box Plot — {col}",
        color_discrete_sequence=[_PALETTE],
        points="outliers",
    )
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white", font_color="#1a1a2e",
    )
    return fig


def build_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    color_col: Optional[str] = None,
):
    fig = px.scatter(
        df, x=x, y=y, color=color_col,
        title=f"{y} vs {x}",
        opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Vivid,
        trendline="ols" if color_col is None else None,
    )
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white", font_color="#1a1a2e",
    )
    return fig


def build_bar_categorical(df: pd.DataFrame, col: str, top_n: int = 10):
    counts = df[col].value_counts().head(top_n).reset_index()
    counts.columns = [col, "Count"]
    fig = px.bar(
        counts, x=col, y="Count",
        title=f"Top {top_n} values in '{col}'",
        color="Count",
        color_continuous_scale="Purples",
    )
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white", font_color="#1a1a2e",
        coloraxis_showscale=False,
    )
    return fig


def build_multi_line(df: pd.DataFrame, cols: List[str]):
    fig = px.line(
        df, y=cols,
        title="Multi-column comparison",
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white", font_color="#1a1a2e",
    )
    return fig


def build_heatmap(df: pd.DataFrame, num_cols: List[str]):
    corr = df[num_cols].corr().round(2)
    fig  = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
    )
    fig.update_layout(
        paper_bgcolor="white", font_color="#1a1a2e", title_font_size=14,
    )
    return fig
