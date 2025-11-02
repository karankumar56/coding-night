"""Streamlit Data Analysis and Visualization Dashboard."""

from __future__ import annotations

from io import BytesIO, StringIO
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DEFAULT_THEME_NAME = "Professional Blue"


THEME_PRESETS: Dict[str, Dict[str, Any]] = {
    "Professional Blue": {
        "style": "whitegrid",
        "palette": ["#2563eb", "#3b82f6", "#60a5fa", "#93c5fd", "#dbeafe"],
        "primary_color": "#2563eb",
        "secondary_color": "#1e40af",
        "accent_color": "#60a5fa",
        "background": "linear-gradient(135deg, #f8fafc 0%, #f1f5f9 50%, #e2e8f0 100%)",
        "sidebar_bg": "linear-gradient(180deg, #1e3a8a 0%, #1e40af 100%)",
        "sidebar_text": "#ffffff",
        "text_color": "#0f172a",
        "card_bg": "#ffffff",
        "border_color": "#e2e8f0",
        "heatmap_cmap": "Blues",
        "success_color": "#10b981",
        "warning_color": "#f59e0b",
        "error_color": "#ef4444",
    },
    "Modern Dark": {
        "style": "darkgrid",
        "palette": ["#6366f1", "#818cf8", "#a78bfa", "#c4b5fd", "#ddd6fe"],
        "primary_color": "#6366f1",
        "secondary_color": "#4f46e5",
        "accent_color": "#818cf8",
        "background": "linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%)",
        "sidebar_bg": "linear-gradient(180deg, #1e1b4b 0%, #312e81 100%)",
        "sidebar_text": "#f8fafc",
        "text_color": "#f1f5f9",
        "card_bg": "#1e293b",
        "border_color": "#334155",
        "heatmap_cmap": "viridis",
        "success_color": "#22c55e",
        "warning_color": "#fbbf24",
        "error_color": "#f87171",
    },
    "Corporate Gray": {
        "style": "whitegrid",
        "palette": ["#475569", "#64748b", "#94a3b8", "#cbd5e1", "#e2e8f0"],
        "primary_color": "#475569",
        "secondary_color": "#334155",
        "accent_color": "#64748b",
        "background": "linear-gradient(135deg, #ffffff 0%, #f8fafc 50%, #f1f5f9 100%)",
        "sidebar_bg": "linear-gradient(180deg, #1e293b 0%, #334155 100%)",
        "sidebar_text": "#ffffff",
        "text_color": "#0f172a",
        "card_bg": "#ffffff",
        "border_color": "#e2e8f0",
        "heatmap_cmap": "gray",
        "success_color": "#059669",
        "warning_color": "#d97706",
        "error_color": "#dc2626",
    },
    "Ocean Professional": {
        "style": "white",
        "palette": ["#0ea5e9", "#38bdf8", "#7dd3fc", "#bae6fd", "#e0f2fe"],
        "primary_color": "#0ea5e9",
        "secondary_color": "#0284c7",
        "accent_color": "#38bdf8",
        "background": "linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 50%, #bae6fd 100%)",
        "sidebar_bg": "linear-gradient(180deg, #075985 0%, #0c4a6e 100%)",
        "sidebar_text": "#ffffff",
        "text_color": "#0c4a6e",
        "card_bg": "#ffffff",
        "border_color": "#bae6fd",
        "heatmap_cmap": "coolwarm",
        "success_color": "#06b6d4",
        "warning_color": "#f59e0b",
        "error_color": "#ef4444",
    },
    "Emerald Professional": {
        "style": "whitegrid",
        "palette": ["#059669", "#10b981", "#34d399", "#6ee7b7", "#a7f3d0"],
        "primary_color": "#059669",
        "secondary_color": "#047857",
        "accent_color": "#10b981",
        "background": "linear-gradient(135deg, #ecfdf5 0%, #d1fae5 50%, #a7f3d0 100%)",
        "sidebar_bg": "linear-gradient(180deg, #064e3b 0%, #065f46 100%)",
        "sidebar_text": "#ffffff",
        "text_color": "#064e3b",
        "card_bg": "#ffffff",
        "border_color": "#a7f3d0",
        "heatmap_cmap": "Greens",
        "success_color": "#10b981",
        "warning_color": "#f59e0b",
        "error_color": "#ef4444",
    },
}


def init_session_state() -> None:
    """Ensure required keys exist in Streamlit session state."""

    defaults = {
        "dataset": None,
        "dataset_name": "",
        "cleaning_actions": [],
        "theme_choice": DEFAULT_THEME_NAME,
        "theme_config": THEME_PRESETS[DEFAULT_THEME_NAME],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def update_dataset(
    df: Optional[pd.DataFrame], *, name: str = "", reset_history: bool = True
) -> None:
    """Persist dataset metadata in session state."""

    st.session_state.dataset = df
    if name:
        st.session_state.dataset_name = name
    if reset_history:
        st.session_state.cleaning_actions = []


def apply_theme(theme_name: str) -> None:
    """Apply visual theme to the Streamlit app and plotting library."""

    theme = THEME_PRESETS.get(theme_name, THEME_PRESETS[DEFAULT_THEME_NAME])
    st.session_state.theme_choice = theme_name
    st.session_state.theme_config = theme

    # Set seaborn theme - handle both list and string palettes
    palette_val = theme["palette"]
    sns.set_theme(style=theme["style"], palette=palette_val)

    # Get theme colors with defaults
    card_bg = theme.get('card_bg', '#ffffff')
    border_color = theme.get('border_color', '#e2e8f0')
    sidebar_text = theme.get('sidebar_text', '#ffffff')
    secondary_color = theme.get('secondary_color', theme['primary_color'])
    accent_color = theme.get('accent_color', theme['primary_color'])
    success_color = theme.get('success_color', '#10b981')
    warning_color = theme.get('warning_color', '#f59e0b')
    error_color = theme.get('error_color', '#ef4444')

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@400;500;600;700;800&display=swap');
    
    * {{
        font-family: 'Inter', 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        box-sizing: border-box;
    }}
    
    /* Main App Container - Complete Webpage Look */
    .stApp {{
        background: {theme['background']};
        color: {theme['text_color']};
        min-height: 100vh;
    }}
    
    /* Main Content Area */
    .main [data-testid="stAppViewContainer"] {{
        background: transparent;
        padding: 2rem 3rem;
        max-width: 1400px;
        margin: 0 auto;
    }}
    
    /* Professional Header Styling */
    h1 {{
        background: linear-gradient(135deg, {theme['primary_color']} 0%, {secondary_color} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'Poppins', sans-serif;
        font-weight: 800;
        font-size: 2.75rem;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
        line-height: 1.2;
    }}
    
    h2 {{
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 1.875rem;
        margin-top: 2.5rem;
        margin-bottom: 1.25rem;
        color: {theme['text_color']};
        position: relative;
        padding-bottom: 0.75rem;
        padding-left: 1rem;
        border-left: 4px solid {theme['primary_color']};
    }}
    
    h3 {{
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 1.5rem;
        color: {theme['text_color']};
        margin-top: 2rem;
        margin-bottom: 1rem;
    }}
    
    /* Professional Sidebar */
    [data-testid="stSidebar"] {{
        background: {theme['sidebar_bg']} !important;
        backdrop-filter: blur(20px);
        box-shadow: 4px 0 30px rgba(0,0,0,0.15);
        border-right: 1px solid {border_color};
    }}
    
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown p {{
        color: {sidebar_text} !important;
    }}
    
    [data-testid="stSidebar"] .stMarkdown h1 {{
        background: linear-gradient(135deg, {sidebar_text}, {sidebar_text}dd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 1.5rem;
        margin-bottom: 1.5rem;
    }}
    
    [data-testid="stSidebar"] .stMarkdown h3 {{
        color: {sidebar_text} !important;
        font-weight: 600;
        opacity: 0.95;
    }}
    
    [data-testid="stSidebar"] [data-baseweb="radio"] label {{
        color: {sidebar_text} !important;
        font-weight: 500;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        transition: all 0.3s ease;
    }}
    
    [data-testid="stSidebar"] [data-baseweb="radio"] label:hover {{
        background: rgba(255,255,255,0.1);
    }}
    
    /* Ensure sidebar selectbox is visible */
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSelectbox > div > div {{
        color: {sidebar_text} !important;
    }}
    
    [data-testid="stSidebar"] .stSelectbox > div > div {{
        background: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
    }}
    
    /* Professional Metrics Cards */
    [data-testid="stMetricValue"] {{
        font-family: 'Poppins', sans-serif;
        font-size: 2.25rem !important;
        font-weight: 700 !important;
        color: {theme['primary_color']} !important;
        line-height: 1.2;
    }}
    
    [data-testid="stMetricLabel"] {{
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        color: {theme['text_color']}99 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }}
    
    [data-testid="stMetricDelta"] {{
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }}
    
    /* Professional Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, {theme['primary_color']} 0%, {secondary_color} 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 0.625rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 14px 0 rgba(0,0,0,0.15);
        text-transform: none;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px 0 rgba(0,0,0,0.2);
        background: linear-gradient(135deg, {secondary_color} 0%, {theme['primary_color']} 100%);
    }}
    
    .stButton > button:active {{
        transform: translateY(0);
    }}
    
    /* Professional File Uploader */
    [data-testid="stFileUploader"] {{
        border-radius: 16px;
        border: 2px dashed {theme['primary_color']};
        opacity: 0.6;
        padding: 3rem 2rem;
        transition: all 0.3s ease;
        background: {card_bg};
        backdrop-filter: blur(10px);
    }}
    
    [data-testid="stFileUploader"]:hover {{
        border-color: {theme['primary_color']};
        opacity: 1;
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
    }}
    
    /* Enhanced Selectbox and Input */
    .stSelectbox > div > div {{
        background: {card_bg} !important;
        border-radius: 12px;
        border: 1.5px solid {border_color} !important;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }}
    
    .stSelectbox > div > div:hover {{
        border-color: {theme['primary_color']} !important;
        box-shadow: 0 4px 12px {theme['primary_color']}20;
    }}
    
    /* Professional Expander */
    [data-testid="stExpander"] {{
        background: {card_bg} !important;
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid {border_color} !important;
        margin: 1.5rem 0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }}
    
    /* Professional Card Container */
    .modern-card {{
        background: {card_bg};
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        border: 1px solid {border_color};
        transition: all 0.3s ease;
    }}
    
    .modern-card:hover {{
        box-shadow: 0 15px 50px rgba(0,0,0,0.12);
        transform: translateY(-2px);
    }}
    
    /* Professional Messages */
    .stSuccess {{
        border-radius: 12px;
        border-left: 4px solid {success_color};
        background: rgba(16, 185, 129, 0.1);
        padding: 1.25rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }}
    
    .stError {{
        border-radius: 12px;
        border-left: 4px solid {error_color};
        background: rgba(239, 68, 68, 0.1);
        padding: 1.25rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }}
    
    .stInfo {{
        border-radius: 12px;
        border-left: 4px solid {theme['primary_color']};
        background: rgba(37, 99, 235, 0.1);
        padding: 1.25rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }}
    
    .stWarning {{
        border-radius: 12px;
        border-left: 4px solid {warning_color};
        background: rgba(245, 158, 11, 0.1);
        padding: 1.25rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }}
    
    /* Professional Radio Buttons */
    [data-testid="stRadio"] {{
        padding: 1.25rem;
        background: {card_bg}90;
        border-radius: 12px;
        margin: 0.75rem 0;
        border: 1px solid {border_color};
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }}
    
    [data-testid="stRadio"] label {{
        font-weight: 500;
    }}
    
    /* Professional Slider */
    .stSlider {{
        margin: 1.5rem 0;
    }}
    
    .stSlider > div > div {{
        background: {border_color};
        border-radius: 10px;
        height: 10px;
    }}
    
    .stSlider > div > div > div {{
        background: linear-gradient(90deg, {theme['primary_color']}, {accent_color});
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}
    
    /* Professional DataFrame */
    .stDataFrame {{
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 4px 24px rgba(0,0,0,0.08);
        border: 1px solid {border_color};
        background: {card_bg};
    }}
    
    /* Professional Checkbox */
    [data-testid="stCheckbox"] {{
        padding: 0.75rem;
    }}
    
    [data-testid="stCheckbox"] label {{
        font-weight: 500;
    }}
    
    /* Smooth Animations */
    @keyframes fadeInUp {{
        from {{
            opacity: 0;
            transform: translateY(20px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    .main-content {{
        animation: fadeInUp 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    
    /* Professional Scrollbar */
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {theme['background']};
        border-radius: 10px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(180deg, {theme['primary_color']}, {accent_color});
        border-radius: 10px;
        border: 2px solid {theme['background']};
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(180deg, {secondary_color}, {theme['primary_color']});
    }}
    
    /* Text Styling */
    body, h1, h2, h3, h4, h5, h6, label, p, span {{
        color: {theme['text_color']} !important;
    }}
    
    p {{
        line-height: 1.7;
        font-size: 1rem;
    }}
    
    /* Metric Styling */
    .stMetric, .stMetricLabel, .stMetricDelta {{
        color: {theme['text_color']} !important;
    }}
    
    /* Professional Divider */
    hr {{
        border: none;
        border-top: 2px solid {border_color};
        margin: 2rem 0;
        opacity: 0.3;
    }}
    
    /* Section Spacing */
    .element-container {{
        margin-bottom: 1.5rem;
    }}
    
    /* Professional Background Pattern */
    .stApp::before {{
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(circle at 15% 20%, {theme['primary_color']}06 0%, transparent 50%),
            radial-gradient(circle at 85% 80%, {accent_color}04 0%, transparent 50%),
            radial-gradient(circle at 50% 50%, {secondary_color}03 0%, transparent 70%);
        pointer-events: none;
        z-index: -1;
    }}
    
    /* Table Styling */
    table {{
        border-radius: 12px;
        overflow: hidden;
    }}
    
    /* Download Button */
    [data-testid="baseButton-secondary"] {{
        background: {card_bg} !important;
        border: 1.5px solid {theme['primary_color']} !important;
        color: {theme['primary_color']} !important;
    }}
    
    [data-testid="baseButton-secondary"]:hover {{
        background: rgba(37, 99, 235, 0.1) !important;
    }}
    
    /* Professional Container Spacing */
    .block-container {{
        padding-top: 3rem;
        padding-bottom: 3rem;
    }}
    
    /* Improve Overall Layout - Keep navigation visible */
    footer {{
        visibility: hidden;
    }}
    
    /* Only hide Streamlit's default hamburger menu if present */
    [data-testid="stHeader"] {{
        background: transparent;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def handle_file_upload() -> None:
    """Render uploader widget and validate CSV files."""

    st.markdown("### 📤 Upload Dataset")
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        help="Only .csv files are supported. Ensure the file is well-formatted and not password protected.",
    )

    if not uploaded_file:
        update_dataset(None)
        st.info("Awaiting CSV upload to begin analysis.")
        return

    try:
        df = pd.read_csv(uploaded_file)
    except pd.errors.EmptyDataError:
        update_dataset(None)
        st.error("The uploaded file is empty. Please provide a CSV with data.")
        return
    except pd.errors.ParserError:
        update_dataset(None)
        st.error("Unable to parse the CSV file. Please check the file structure and try again.")
        return
    except Exception as exc:  # pylint: disable=broad-exception-caught
        update_dataset(None)
        st.error(f"An unexpected error occurred while reading the file: {exc}")
        return

    if df.empty:
        update_dataset(None)
        st.error("The dataset contains no rows after loading. Please upload a non-empty CSV.")
        return

    if df.columns.empty:
        update_dataset(None)
        st.error("The dataset contains no columns. Please verify the CSV content.")
        return

    update_dataset(df, name=uploaded_file.name)
    st.success(f"Loaded dataset '{uploaded_file.name}' with {len(df)} rows and {len(df.columns)} columns.")

    with st.expander("Dataset snapshot", expanded=False):
        st.dataframe(df.head(), use_container_width=True)
        st.write(
            {
                "Rows": len(df),
                "Columns": len(df.columns),
                "Missing values": int(df.isna().sum().sum()),
            }
        )


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric column names."""

    return df.select_dtypes(include=["number"]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return categorical-like column names."""

    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def render_data_summary(df: pd.DataFrame) -> None:
    """Display dataset overview, structure, and summary statistics."""

    st.markdown("### 📈 Dataset Overview")
    st.markdown("---")

    total_rows = len(df)
    total_columns = len(df.columns)
    total_missing = int(df.isna().sum().sum())

    theme = st.session_state.get("theme_config", THEME_PRESETS[DEFAULT_THEME_NAME])
    primary = theme['primary_color']
    secondary = theme.get('secondary_color', primary)
    accent = theme.get('accent_color', primary)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, {primary} 0%, {secondary} 100%); 
                        border-radius: 20px; padding: 2rem; color: white; text-align: center;
                        box-shadow: 0 10px 40px rgba(0,0,0,0.2); transition: transform 0.3s ease;">
                <div style="font-size: 3rem; font-weight: 800; margin-bottom: 0.5rem; font-family: 'Poppins', sans-serif;">{total_rows:,}</div>
                <div style="font-size: 0.9rem; opacity: 0.95; text-transform: uppercase; letter-spacing: 2px; font-weight: 600;">Rows</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, {secondary} 0%, {accent} 100%); 
                        border-radius: 20px; padding: 2rem; color: white; text-align: center;
                        box-shadow: 0 10px 40px rgba(0,0,0,0.2); transition: transform 0.3s ease;">
                <div style="font-size: 3rem; font-weight: 800; margin-bottom: 0.5rem; font-family: 'Poppins', sans-serif;">{total_columns:,}</div>
                <div style="font-size: 0.9rem; opacity: 0.95; text-transform: uppercase; letter-spacing: 2px; font-weight: 600;">Columns</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, {accent} 0%, {primary} 100%); 
                        border-radius: 20px; padding: 2rem; color: white; text-align: center;
                        box-shadow: 0 10px 40px rgba(0,0,0,0.2); transition: transform 0.3s ease;">
                <div style="font-size: 3rem; font-weight: 800; margin-bottom: 0.5rem; font-family: 'Poppins', sans-serif;">{total_missing:,}</div>
                <div style="font-size: 0.9rem; opacity: 0.95; text-transform: uppercase; letter-spacing: 2px; font-weight: 600;">Missing Values</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 📋 Column Types")
    st.markdown("---")
    column_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing": df.isna().sum().values,
    })
    st.dataframe(column_info, use_container_width=True, height=300)

    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_categorical_columns(df)

    if not numeric_columns:
        st.warning("No numeric columns identified. Some statistics and charts may be unavailable.")

    if not categorical_columns:
        st.warning("No categorical columns identified. Bar charts may be unavailable.")

    st.markdown("### 📊 Descriptive Statistics")
    st.markdown("---")

    if numeric_columns:
        numeric_df = df[numeric_columns]
        stats_table = pd.DataFrame(
            {
                "Mean": np.round(np.mean(numeric_df.values, axis=0), 3),
                "Median": np.round(np.median(numeric_df.values, axis=0), 3),
                "Std Dev": np.round(np.std(numeric_df.values, axis=0, ddof=1), 3),
                "Min": np.round(np.min(numeric_df.values, axis=0), 3),
                "Max": np.round(np.max(numeric_df.values, axis=0), 3),
            },
            index=numeric_columns,
        )
        st.dataframe(stats_table, use_container_width=True)
    else:
        st.info("Upload a dataset with numeric columns to view descriptive statistics.")

    st.markdown("### 📑 Summary by Column")
    st.markdown("---")
    with st.expander("📖 View detailed summary statistics", expanded=False):
        try:
            desc = df.describe(include="all", datetime_is_numeric=True)
        except TypeError:
            desc = df.describe(include="all")
        st.write(desc.transpose())


def build_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing value counts and percentages per column."""

    summary = pd.DataFrame({
        "Missing Count": df.isna().sum(),
        "Missing %": (df.isna().mean() * 100).round(2),
    })
    summary = summary.sort_values("Missing %", ascending=False)
    summary.index.name = "Column"
    return summary


def apply_cleaning_action(df: pd.DataFrame, action: str) -> pd.DataFrame:
    """Apply the selected missing data cleaning action."""

    if action == "Drop rows with missing values":
        cleaned = df.dropna().reset_index(drop=True)
        st.session_state.cleaning_actions.append("Dropped rows with missing values")
    elif action == "Fill numeric columns with mean":
        cleaned = df.copy()
        numeric_cols = get_numeric_columns(cleaned)
        if not numeric_cols:
            st.warning("No numeric columns available to fill with mean values.")
            return df
        missing_numeric = cleaned[numeric_cols].isna().sum().sum()
        if missing_numeric == 0:
            st.info("Numeric columns do not contain missing values; no changes applied.")
            return df
        means = cleaned[numeric_cols].mean()
        cleaned[numeric_cols] = cleaned[numeric_cols].fillna(means)
        st.session_state.cleaning_actions.append("Filled numeric columns with their mean")
    else:
        cleaned = df

    return cleaned


def generate_summary_report(df: pd.DataFrame) -> str:
    """Create a CSV report combining descriptive stats and missing-value summary."""

    try:
        descriptive = df.describe(include="all", datetime_is_numeric=True)
    except TypeError:
        descriptive = df.describe(include="all")
    descriptive = descriptive.transpose()
    missing_summary = build_missing_summary(df)

    buffer = StringIO()
    buffer.write("# Descriptive Statistics\n")
    descriptive.to_csv(buffer)
    buffer.write("\n# Missing Value Summary\n")
    missing_summary.to_csv(buffer)

    return buffer.getvalue()


def detect_outliers(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, float, float]:
    """Detect outliers using IQR method."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound


def calculate_data_quality_score(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate overall data quality score."""
    total_cells = df.size
    missing_cells = df.isna().sum().sum()
    completeness = (1 - missing_cells / total_cells) * 100
    
    numeric_cols = get_numeric_columns(df)
    duplicate_rows = df.duplicated().sum()
    uniqueness = (1 - duplicate_rows / len(df)) * 100 if len(df) > 0 else 100
    
    quality_score = (completeness + uniqueness) / 2
    
    return {
        "score": round(quality_score, 2),
        "completeness": round(completeness, 2),
        "uniqueness": round(uniqueness, 2),
        "missing_cells": int(missing_cells),
        "duplicate_rows": int(duplicate_rows),
    }


def generate_smart_insights(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Generate automatic insights from the dataset."""
    insights = []
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)
    
    if numeric_cols:
        corr_matrix = df[numeric_cols].corr()
        np.fill_diagonal(corr_matrix.values, 0)
        max_corr = corr_matrix.max().max()
        if max_corr > 0.7:
            row, col = corr_matrix.stack().idxmax()
            insights.append({
                "type": "Strong Correlation",
                "message": f"Strong positive correlation ({max_corr:.2f}) between {row} and {col}",
            })
        elif max_corr < -0.7:
            row, col = corr_matrix.stack().idxmin()
            insights.append({
                "type": "Strong Correlation",
                "message": f"Strong negative correlation ({max_corr:.2f}) between {row} and {col}",
            })
    
    for col in numeric_cols[:5]:
        outliers, _, _ = detect_outliers(df, col)
        if len(outliers) > len(df) * 0.05:
            insights.append({
                "type": "Outliers Detected",
                "message": f"{col} has {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}% of data)",
            })
    
    missing_summary = build_missing_summary(df)
    high_missing = missing_summary[missing_summary["Missing %"] > 50]
    if not high_missing.empty:
        for col in high_missing.index:
            insights.append({
                "type": "Data Quality",
                "message": f"{col} has {high_missing.loc[col, 'Missing %']:.1f}% missing values - consider review",
            })
    
    if categorical_cols:
        for col in categorical_cols[:3]:
            value_counts = df[col].value_counts()
            if len(value_counts) == 1:
                insights.append({
                    "type": "Data Quality",
                    "message": f"{col} has only one unique value - may not be useful for analysis",
                })
            elif value_counts.iloc[0] / len(df) > 0.9:
                insights.append({
                    "type": "Imbalance",
                    "message": f"{col} is highly imbalanced - {value_counts.iloc[0]/len(df)*100:.1f}% are {value_counts.index[0]}",
                })
    
    return insights


def build_sidebar() -> str:
    """Render the sidebar and return the selected section key."""

    theme = st.session_state.get("theme_config", THEME_PRESETS[DEFAULT_THEME_NAME])
    sidebar_text_color = theme.get('sidebar_text', '#ffffff')
    border_color_sidebar = f"rgba(255,255,255,0.2)" if sidebar_text_color == "#ffffff" else f"rgba(0,0,0,0.2)"
    
    st.sidebar.markdown(
        f"""
        <div style="text-align: center; padding: 1rem 0; border-bottom: 2px solid {border_color_sidebar}; margin-bottom: 1.5rem;">
            <h1 style="font-size: 1.5rem; margin: 0; color: {sidebar_text_color};">🧭 Navigation</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.markdown(f"<p style='font-size: 0.9rem; color: {sidebar_text_color}; opacity: 0.9; margin-bottom: 1.5rem;'>Use the sections below to explore your dataset.</p>", unsafe_allow_html=True)

    st.sidebar.markdown("### 🎨 Appearance")
    theme_options = list(THEME_PRESETS.keys())
    current_theme = st.session_state.get("theme_choice", DEFAULT_THEME_NAME)
    default_index = theme_options.index(current_theme) if current_theme in theme_options else 0
    theme_choice = st.sidebar.selectbox(
        "Color theme",
        theme_options,
        index=default_index,
        label_visibility="collapsed"
    )
    if theme_choice != current_theme:
        apply_theme(theme_choice)
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()

    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.markdown("### 📍 Quick Access")

    section = st.sidebar.radio(
        "",
        (
            "📊 Data Summary",
            "💡 Smart Insights",
            "📈 Visualizations",
            "🔍 Missing Data",
            "🤖 Mini ML Model",
            "💾 Export Report",
            "ℹ️ About",
        ),
        label_visibility="collapsed"
    )
    
    # Extract section name without emoji
    section_map = {
        "📊 Data Summary": "Data Summary",
        "💡 Smart Insights": "Smart Insights",
        "📈 Visualizations": "Visualizations",
        "🔍 Missing Data": "Missing Data",
        "🤖 Mini ML Model": "Mini ML Model",
        "💾 Export Report": "Export Report",
        "ℹ️ About": "About",
    }
    
    return section_map.get(section, section)


def render_header() -> None:
    """Render the main page title and intro copy."""
    
    theme = st.session_state.get("theme_config", THEME_PRESETS[DEFAULT_THEME_NAME])
    text_color = theme['text_color']
    # Use a slightly muted version of text color for subtitle
    text_color_muted = f"{text_color}99" if len(text_color) == 7 else f"{text_color}cc"

    st.markdown(
        f"""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="margin-bottom: 0.5rem;">📊 Data Analysis & Visualization Dashboard</h1>
            <p style="font-size: 1.1rem; color: {text_color_muted}; margin-top: 0;">Upload, explore, visualize, and analyze your data with powerful tools</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_section_placeholder(section: str) -> None:
    """Fallback helper when a section is not configured."""

    st.info(f"Section '{section}' is not available in this build.")


def render_about_section(_: pd.DataFrame) -> None:
    st.markdown("### ℹ️ About This Dashboard")
    st.markdown("---")
    
    st.write(
        """
        **Data Analysis and Visualization Dashboard** is a comprehensive Python-based analytics tool 
        that provides end-to-end data exploration capabilities in an interactive, user-friendly interface.
        """
    )
    
    st.markdown("### ✨ Key Features")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        **📊 Data Exploration**
        - Upload and preview CSV datasets
        - Comprehensive summary statistics
        - Column type analysis
        - Descriptive statistics
        
        **🎨 Visualizations**
        - Matplotlib & Seaborn static charts
        - Multiple chart types (Histogram, Bar, Box, Heatmap, Scatter, Pairplot)
        - Customizable plot options
        
        **🤖 Machine Learning**
        - Linear Regression
        - Random Forest Classification
        - K-Means Clustering
        - Model evaluation metrics
        """)
    
    with col2:
        st.write("""
        **💡 Smart Insights**
        - Automatic pattern detection
        - Outlier identification
        - Correlation analysis
        - Data quality scoring
        
        **🔧 Data Management**
        - Missing data handling
        - Data cleaning actions
        - Multiple export formats (CSV, Excel, JSON)
        
        **📈 Advanced Analytics**
        - Statistical summaries
        - Feature importance
        - Model performance metrics
        """)
    
    st.markdown("### 🛠️ Technologies Used")
    st.markdown("---")
    st.write(
        """
        - **Streamlit** - Interactive web app framework
        - **Pandas** - Data manipulation and analysis
        - **NumPy** - Numerical computations
        - **Matplotlib & Seaborn** - Static visualizations
        - **Plotly** - Interactive visualizations
        - **Scikit-learn** - Machine learning models
        """
    )
    
    st.markdown("### 📋 How to Use")
    st.markdown("---")
    st.write(
        """
        1. **Upload** a CSV file using the upload widget
        2. **Explore** your data in the Data Summary section
        3. **Generate** Smart Insights for automatic pattern detection
        4. **Visualize** data using Matplotlib & Seaborn charts
        5. **Handle** missing data with cleaning tools
        6. **Train** ML models for predictions or clustering
        7. **Export** reports in CSV, Excel, or JSON formats
        """
    )
    
    st.caption("Built with Python, Streamlit, Pandas, NumPy, Matplotlib, Seaborn, Plotly, and Scikit-learn.")


def render_visualizations(df: pd.DataFrame) -> None:
    """Provide interactive visualization options."""

    theme = st.session_state.get("theme_config", THEME_PRESETS[DEFAULT_THEME_NAME])
    # Set seaborn theme - handle both list and string palettes
    palette_val = theme["palette"]
    sns.set_theme(style=theme["style"], palette=palette_val)
    
    text_color = theme['text_color']
    text_color_muted = f"{text_color}99" if len(text_color) == 7 else f"{text_color}cc"
    
    st.markdown("### 📈 Visualization Explorer")
    st.markdown(f"<p style='font-size: 1rem; color: {text_color_muted}; margin-bottom: 1.5rem;'>Select a plot type and configure options to explore your data visually</p>", unsafe_allow_html=True)
    st.markdown("---")

    plot_type = st.selectbox(
        "Choose a visualization",
        (
            "Histogram",
            "Bar Chart",
            "Box Plot",
            "Heatmap",
            "Scatter Plot",
            "Pairplot",
        ),
    )

    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)

    if plot_type == "Histogram":
        if not numeric_cols:
            st.warning("Histogram requires at least one numeric column.")
            return

        column = st.selectbox("Numeric column", numeric_cols)
        bins = st.slider("Number of bins", min_value=5, max_value=60, value=20)
        fig, ax = plt.subplots()
        sns.histplot(df[column].dropna(), bins=bins, kde=True, ax=ax, color=theme["primary_color"])
        ax.set_title(f"Histogram of {column}")
        st.pyplot(fig)
        plt.close(fig)

    elif plot_type == "Bar Chart":
        if not categorical_cols:
            st.warning("Bar charts require at least one categorical column.")
            return

        column = st.selectbox("Categorical column", categorical_cols)
        normalized = st.checkbox("Show proportions instead of counts", value=False)
        counts = df[column].dropna().value_counts(normalize=normalized)
        fig, ax = plt.subplots()
        sns.barplot(x=counts.index, y=counts.values, ax=ax, palette=theme["palette"])
        ax.set_xlabel(column)
        ax.set_ylabel("Proportion" if normalized else "Count")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_title(f"Bar Chart of {column}")
        st.pyplot(fig)
        plt.close(fig)

    elif plot_type == "Box Plot":
        if not numeric_cols:
            st.warning("Box plots require numeric columns.")
            return

        value_column = st.selectbox("Numeric column", numeric_cols)
        group_column = st.selectbox("Group by (optional)", ["None"] + categorical_cols)
        fig, ax = plt.subplots()
        if group_column == "None":
            sns.boxplot(x=df[value_column].dropna(), ax=ax, color=theme["primary_color"])
            ax.set_xlabel(value_column)
            ax.set_ylabel("Value")
        else:
            sns.boxplot(x=df[group_column], y=df[value_column], ax=ax, palette=theme["palette"])
            ax.set_xlabel(group_column)
            ax.set_ylabel(value_column)
        ax.set_title("Box Plot")
        st.pyplot(fig)
        plt.close(fig)

    elif plot_type == "Heatmap":
        if len(numeric_cols) < 2:
            st.warning("Heatmap requires at least two numeric columns.")
            return

        corr_method = st.selectbox("Correlation method", ("pearson", "spearman", "kendall"))
        corr = df[numeric_cols].corr(method=corr_method)
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap=theme["heatmap_cmap"], ax=ax)
        ax.set_title(f"Correlation Heatmap ({corr_method.title()})")
        st.pyplot(fig)
        plt.close(fig)

    elif plot_type == "Scatter Plot":
        if len(numeric_cols) < 2:
            st.warning("Scatter plots require at least two numeric columns.")
            return

        x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
        y_choices = [col for col in numeric_cols if col != x_col]
        if not y_choices:
            st.warning("Select a different X column to unlock Y options.")
            return
        y_col = st.selectbox("Y-axis", y_choices, key="scatter_y")
        hue_col = st.selectbox("Color by (optional)", ["None"] + categorical_cols, key="scatter_hue")

        fig, ax = plt.subplots()
        sns.scatterplot(
            data=df,
            x=x_col,
            y=y_col,
            hue=None if hue_col == "None" else hue_col,
            ax=ax,
            palette=theme["palette"],
        )
        ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
        st.pyplot(fig)
        plt.close(fig)

    elif plot_type == "Pairplot":
        if len(numeric_cols) < 2:
            st.warning("Pairplot requires at least two numeric columns.")
            return

        default_selection = numeric_cols[:4]
        selected_columns = st.multiselect(
            "Select up to six columns",
            numeric_cols,
            default=default_selection,
            max_selections=min(6, len(numeric_cols)),
        )
        if len(selected_columns) < 2:
            st.warning("Choose at least two columns for the pairplot.")
            return

        plot_df = df[selected_columns].dropna()
        if plot_df.empty:
            st.warning("No data available after removing missing values for the selected columns.")
            return

        max_slider_rows = min(len(plot_df), 5000)
        if max_slider_rows > 200:
            sample_size = st.slider(
                "Sample rows (for performance)",
                min_value=200,
                max_value=max_slider_rows,
                value=min(1000, max_slider_rows),
            )
            if len(plot_df) > sample_size:
                plot_df = plot_df.sample(sample_size, random_state=42)

        fig = sns.pairplot(plot_df, diag_kind="kde", corner=True, palette=theme["palette"])
        st.pyplot(fig.fig)
        plt.close(fig.fig)


def render_missing_data(df: pd.DataFrame) -> None:
    """Highlight missing data patterns and provide cleaning tools."""

    theme = st.session_state.get("theme_config", THEME_PRESETS[DEFAULT_THEME_NAME])
    # Set seaborn theme - handle both list and string palettes
    palette_val = theme["palette"]
    sns.set_theme(style=theme["style"], palette=palette_val)

    st.markdown("### 🔍 Missing Data Overview")
    st.markdown("---")
    summary = build_missing_summary(df)
    st.dataframe(summary, use_container_width=True, height=300)

    if summary["Missing Count"].sum() == 0:
        st.success("No missing values detected in the dataset.")
        return

    st.markdown("### 🌡️ Missing Data Heatmap")
    st.markdown("---")
    missing_cols = summary[summary["Missing Count"] > 0].index.tolist()
    heatmap_df = df[missing_cols].isna().astype(int)
    if len(heatmap_df) > 150:
        heatmap_df = heatmap_df.head(150)
        st.caption("Showing first 150 rows for readability.")
    fig, ax = plt.subplots(figsize=(min(12, len(missing_cols) * 1.2), 6))
    sns.heatmap(heatmap_df.transpose(), cmap=theme["heatmap_cmap"], cbar=False, ax=ax)
    ax.set_xlabel("Row index")
    ax.set_ylabel("Column")
    ax.set_title("Missing Value Heatmap")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### 🛠️ Handle Missing Values")
    st.markdown("---")
    action = st.radio(
        "Select an action",
        (
            "No action",
            "Drop rows with missing values",
            "Fill numeric columns with mean",
        ),
    )

    confirm = st.checkbox("I confirm that I want to apply this action", value=False)

    if st.button("Apply action"):
        if action == "No action":
            st.info("No changes applied.")
        elif not confirm:
            st.warning("Please confirm the action before applying changes.")
        else:
            cleaned_df = apply_cleaning_action(df, action)
            if cleaned_df is df:
                st.info("No changes were necessary for the selected action.")
            else:
                update_dataset(
                    cleaned_df,
                    name=st.session_state.dataset_name,
                    reset_history=False,
                )
                st.success("Action applied successfully. Dataset updated.")
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()


def render_smart_insights(df: pd.DataFrame) -> None:
    """Display automatic insights and data quality metrics."""
    
    st.markdown("### 💡 Smart Insights")
    theme = st.session_state.get("theme_config", THEME_PRESETS[DEFAULT_THEME_NAME])
    text_color = theme['text_color']
    st.markdown(f"<p style='font-size: 1rem; color: {text_color}99; margin-bottom: 1.5rem;'>Automatically generated insights about your dataset</p>", unsafe_allow_html=True)
    
    quality = calculate_data_quality_score(df)
    
    primary = theme['primary_color']
    secondary = theme.get('secondary_color', primary)
    accent = theme.get('accent_color', primary)
    success_color = theme.get('success_color', '#10b981')
    warning_color = theme.get('warning_color', '#f59e0b')
    error_color = theme.get('error_color', '#ef4444')
    
    quality_color = success_color if quality['score'] >= 80 else warning_color if quality['score'] >= 60 else error_color
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, {quality_color} 0%, {quality_color}dd 100%); 
                        border-radius: 20px; padding: 2rem; color: white; text-align: center;
                        box-shadow: 0 10px 40px rgba(0,0,0,0.2);">
                <div style="font-size: 3rem; font-weight: 800; margin-bottom: 0.5rem; font-family: 'Poppins', sans-serif;">{quality['score']}%</div>
                <div style="font-size: 0.9rem; opacity: 0.95; text-transform: uppercase; letter-spacing: 2px; font-weight: 600;">Quality Score</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, {primary} 0%, {secondary} 100%); 
                        border-radius: 20px; padding: 2rem; color: white; text-align: center;
                        box-shadow: 0 10px 40px rgba(0,0,0,0.2);">
                <div style="font-size: 3rem; font-weight: 800; margin-bottom: 0.5rem; font-family: 'Poppins', sans-serif;">{quality['completeness']:.1f}%</div>
                <div style="font-size: 0.9rem; opacity: 0.95; text-transform: uppercase; letter-spacing: 2px; font-weight: 600;">Completeness</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, {accent} 0%, {primary} 100%); 
                        border-radius: 20px; padding: 2rem; color: white; text-align: center;
                        box-shadow: 0 10px 40px rgba(0,0,0,0.2);">
                <div style="font-size: 3rem; font-weight: 800; margin-bottom: 0.5rem; font-family: 'Poppins', sans-serif;">{quality['uniqueness']:.1f}%</div>
                <div style="font-size: 0.9rem; opacity: 0.95; text-transform: uppercase; letter-spacing: 2px; font-weight: 600;">Uniqueness</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    
    insights = generate_smart_insights(df)
    
    if insights:
        for insight in insights[:10]:
            if insight["type"] == "Strong Correlation":
                st.success(f"🔗 {insight['message']}")
            elif insight["type"] == "Outliers Detected":
                st.warning(f"⚠️ {insight['message']}")
            elif insight["type"] == "Data Quality":
                st.error(f"❌ {insight['message']}")
            elif insight["type"] == "Imbalance":
                st.info(f"⚖️ {insight['message']}")
    else:
        st.info("No significant insights detected. Your data looks clean!")
    
    st.markdown("### 📊 Outlier Analysis")
    st.markdown("---")
    numeric_cols = get_numeric_columns(df)
    if numeric_cols:
        selected_col = st.selectbox("Select column for outlier analysis", numeric_cols)
        outliers, lower, upper = detect_outliers(df, selected_col)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Outliers Found", len(outliers))
        col2.metric("Lower Bound", f"{lower:.2f}")
        col3.metric("Upper Bound", f"{upper:.2f}")
        
        if len(outliers) > 0:
            st.dataframe(outliers[[selected_col]].head(20), use_container_width=True)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[selected_col],
                mode='markers',
                name='Normal',
                marker=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=outliers.index,
                y=outliers[selected_col],
                mode='markers',
                name='Outliers',
                marker=dict(color='red', size=8)
            ))
            fig.add_hline(y=lower, line_dash="dash", line_color="orange", annotation_text="Lower Bound")
            fig.add_hline(y=upper, line_dash="dash", line_color="orange", annotation_text="Upper Bound")
            fig.update_layout(title=f"Outlier Detection: {selected_col}", height=400)
            st.plotly_chart(fig, use_container_width=True)


def render_ml_model(df: pd.DataFrame) -> None:
    """Build and evaluate simple machine learning models."""
    
    theme = st.session_state.get("theme_config", THEME_PRESETS[DEFAULT_THEME_NAME])
    text_color = theme['text_color']
    text_color_muted = f"{text_color}99" if len(text_color) == 7 else f"{text_color}cc"
    
    st.markdown("### 🤖 Mini ML Model")
    st.markdown(f"<p style='font-size: 1rem; color: {text_color_muted}; margin-bottom: 1.5rem;'>Train simple predictive models on your dataset</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    numeric_cols = get_numeric_columns(df)
    if len(numeric_cols) < 2:
        st.warning("ML models require at least 2 numeric columns.")
        return
    
    model_type = st.selectbox("Select model type", ["Regression", "Classification", "Clustering"])
    
    if model_type == "Regression":
        st.write("**Linear Regression**: Predict a numeric target using numeric features.")
        
        target_col = st.selectbox("Target column (what to predict)", numeric_cols)
        feature_cols = [col for col in numeric_cols if col != target_col]
        
        if not feature_cols:
            st.warning("No feature columns available.")
            return
        
        selected_features = st.multiselect("Select feature columns", feature_cols, default=feature_cols[:3])
        
        if not selected_features or target_col in selected_features:
            st.warning("Please select valid feature columns.")
            return
        
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                clean_df = df[selected_features + [target_col]].dropna()
                if len(clean_df) < 10:
                    st.error("Not enough data after removing missing values.")
                    return
                
                X = clean_df[selected_features]
                y = clean_df[target_col]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("R² Score", f"{r2:.3f}")
                col2.metric("RMSE", f"{rmse:.2f}")
                col3.metric("Training Samples", len(X_train))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions'))
                fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], 
                                        mode='lines', name='Perfect Prediction', line=dict(dash='dash')))
                fig.update_layout(title="Actual vs Predicted", xaxis_title="Actual", yaxis_title="Predicted")
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("**Feature Importance (Coefficients):**")
                coef_df = pd.DataFrame({
                    "Feature": selected_features,
                    "Coefficient": model.coef_,
                }).sort_values("Coefficient", key=abs, ascending=False)
                st.dataframe(coef_df, use_container_width=True)
    
    elif model_type == "Classification":
        st.write("**Random Forest Classifier**: Classify categorical targets.")
        
        categorical_cols = get_categorical_columns(df)
        if not categorical_cols:
            st.warning("Classification requires at least one categorical column.")
            return
        
        target_col = st.selectbox("Target column (what to classify)", categorical_cols)
        unique_values = df[target_col].nunique()
        
        if unique_values > 20:
            st.warning(f"Target has {unique_values} unique values. Consider selecting a different column.")
            return
        
        feature_cols = [col for col in numeric_cols]
        selected_features = st.multiselect("Select feature columns", feature_cols, default=feature_cols[:5])
        
        if not selected_features:
            st.warning("Please select at least one feature column.")
            return
        
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                clean_df = df[selected_features + [target_col]].dropna()
                if len(clean_df) < 10:
                    st.error("Not enough data after removing missing values.")
                    return
                
                le = LabelEncoder()
                X = clean_df[selected_features]
                y = le.fit_transform(clean_df[target_col])
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                col1, col2 = st.columns(2)
                col1.metric("Accuracy", f"{accuracy:.3f}")
                col2.metric("Classes", unique_values)
                
                feature_importance = pd.DataFrame({
                    "Feature": selected_features,
                    "Importance": model.feature_importances_,
                }).sort_values("Importance", ascending=False)
                st.write("**Feature Importance:**")
                st.dataframe(feature_importance, use_container_width=True)
    
    elif model_type == "Clustering":
        st.write("**K-Means Clustering**: Group similar data points.")
        
        selected_features = st.multiselect("Select feature columns for clustering", numeric_cols, 
                                          default=numeric_cols[:3])
        if len(selected_features) < 2:
            st.warning("Select at least 2 features for clustering.")
            return
        
        n_clusters = st.slider("Number of clusters", 2, 10, 3)
        
        if st.button("Run Clustering"):
            with st.spinner("Clustering data..."):
                clean_df = df[selected_features].dropna()
                if len(clean_df) < n_clusters:
                    st.error("Not enough data for clustering.")
                    return
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(clean_df[selected_features])
                
                clean_df = clean_df.copy()
                clean_df["Cluster"] = clusters
                
                st.metric("Clusters Created", n_clusters)
                
                if len(selected_features) >= 2:
                    fig = px.scatter(
                        clean_df,
                        x=selected_features[0],
                        y=selected_features[1],
                        color="Cluster",
                        title="K-Means Clustering Visualization",
                        labels={selected_features[0]: selected_features[0], 
                               selected_features[1]: selected_features[1]}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.write("**Cluster Sizes:**")
                cluster_sizes = pd.Series(clusters).value_counts().sort_index()
                st.dataframe(cluster_sizes.to_frame("Count"), use_container_width=True)


def render_download_section(df: pd.DataFrame) -> None:
    """Provide downloadable reports in multiple formats."""
    
    st.markdown("### 💾 Export Reports & Data")
    st.markdown("---")
    
    export_type = st.selectbox(
        "Select export format",
        ["CSV Report (Statistics)", "Full Dataset (CSV)", "Excel Workbook", "JSON Format"],
    )
    
    if export_type == "CSV Report (Statistics)":
        st.write("Generate a CSV report containing descriptive statistics and missing value summary.")
        report_csv = generate_summary_report(df)
        st.download_button(
            label="Download CSV report",
            data=report_csv,
            file_name="data_summary_report.csv",
            mime="text/csv",
        )
    
    elif export_type == "Full Dataset (CSV)":
        st.write("Download the complete dataset as a CSV file.")
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="Download Full Dataset",
            data=csv_data,
            file_name=f"{st.session_state.dataset_name or 'dataset'}.csv",
            mime="text/csv",
        )
    
    elif export_type == "Excel Workbook":
        st.write("Download dataset and summary statistics as an Excel workbook with multiple sheets.")
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name="Data", index=False)
            try:
                desc = df.describe(include="all", datetime_is_numeric=True)
            except TypeError:
                desc = df.describe(include="all")
            desc.transpose().to_excel(writer, sheet_name="Statistics")
            build_missing_summary(df).to_excel(writer, sheet_name="Missing Data")
        buffer.seek(0)
        st.download_button(
            label="Download Excel Workbook",
            data=buffer,
            file_name=f"{st.session_state.dataset_name or 'dataset'}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    
    elif export_type == "JSON Format":
        st.write("Download the dataset in JSON format.")
        json_data = df.to_json(orient="records", indent=2)
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name=f"{st.session_state.dataset_name or 'dataset'}.json",
            mime="application/json",
        )
    
    st.write("---")
    
    if st.session_state.cleaning_actions:
        st.subheader("Applied Cleaning Actions")
        for idx, action in enumerate(st.session_state.cleaning_actions, start=1):
            st.write(f"{idx}. {action}")


def main() -> None:
    """App entry point."""

    st.set_page_config(page_title="Data Analysis Dashboard", layout="wide")
    init_session_state()
    apply_theme(st.session_state.theme_choice)

    render_header()
    handle_file_upload()

    if st.session_state.dataset is None:
        st.warning("Upload a dataset to unlock the dashboard features.")
        return

    section = build_sidebar()

    section_renderers: Dict[str, Callable[[pd.DataFrame], None]] = {
        "Data Summary": render_data_summary,
        "Smart Insights": render_smart_insights,
        "Visualizations": render_visualizations,
        "Missing Data": render_missing_data,
        "Mini ML Model": render_ml_model,
        "Export Report": render_download_section,
        "About": render_about_section,
    }

    renderer = section_renderers.get(section)

    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    with st.container():
        if renderer:
            renderer(st.session_state.dataset)
        else:
            render_section_placeholder(section)
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

