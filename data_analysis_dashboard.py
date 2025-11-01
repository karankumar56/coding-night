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


DEFAULT_THEME_NAME = "Streamlit Light"


THEME_PRESETS: Dict[str, Dict[str, Any]] = {
    "Streamlit Light": {
        "style": "whitegrid",
        "palette": "deep",
        "primary_color": "#FF4B4B",
        "background": "linear-gradient(135deg, #ffffff 0%, #f5f7fb 100%)",
        "sidebar_bg": "#eef2f9",
        "text_color": "#1f2a44",
        "heatmap_cmap": "coolwarm",
    },
    "Midnight": {
        "style": "darkgrid",
        "palette": "rocket",
        "primary_color": "#8ecae6",
        "background": "#0b132b",
        "sidebar_bg": "#1c2541",
        "text_color": "#f1f5f9",
        "heatmap_cmap": "mako",
    },
    "Ocean": {
        "style": "white",
        "palette": ["#006494", "#247ba0", "#1b98e0", "#76c7f4", "#bedfed"],
        "primary_color": "#006494",
        "background": "linear-gradient(135deg, #e6f1ff 0%, #ffffff 100%)",
        "sidebar_bg": "#d9e8ff",
        "text_color": "#002855",
        "heatmap_cmap": "crest",
    },
    "Sunset": {
        "style": "ticks",
        "palette": ["#ff7f51", "#ffae5d", "#ffd166", "#f6bd60", "#ff9f1c"],
        "primary_color": "#ff7f51",
        "background": "linear-gradient(160deg, #fff4e6 0%, #ffffff 100%)",
        "sidebar_bg": "#ffe8d6",
        "text_color": "#4a2b16",
        "heatmap_cmap": "flare",
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

    sns.set_theme(style=theme["style"], palette=theme["palette"])

    css = f"""
    <style>
    .stApp {{
        background: {theme['background']};
        color: {theme['text_color']};
    }}
    body, h1, h2, h3, h4, h5, h6, label, p, span {{
        color: {theme['text_color']} !important;
    }}
    [data-testid="stSidebar"] {{
        background: {theme['sidebar_bg']};
    }}
    .stMetric, .stMetricLabel, .stMetricDelta {{
        color: {theme['text_color']} !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def handle_file_upload() -> None:
    """Render uploader widget and validate CSV files."""

    st.subheader("1. Upload Dataset")
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

    st.subheader("Dataset Overview")

    total_rows = len(df)
    total_columns = len(df.columns)
    total_missing = int(df.isna().sum().sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{total_rows:,}")
    col2.metric("Columns", f"{total_columns:,}")
    col3.metric("Missing values", f"{total_missing:,}")

    st.subheader("Column Types")
    column_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing": df.isna().sum().values,
    })
    st.dataframe(column_info, use_container_width=True)

    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_categorical_columns(df)

    if not numeric_columns:
        st.warning("No numeric columns identified. Some statistics and charts may be unavailable.")

    if not categorical_columns:
        st.warning("No categorical columns identified. Bar charts may be unavailable.")

    st.subheader("Descriptive Statistics")

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

    st.subheader("Summary by Column")
    with st.expander("View summary statistics", expanded=False):
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

    st.sidebar.title("Navigation")
    st.sidebar.write("Use the sections below to explore the dataset.")

    st.sidebar.subheader("Appearance")
    theme_options = list(THEME_PRESETS.keys())
    current_theme = st.session_state.get("theme_choice", DEFAULT_THEME_NAME)
    default_index = theme_options.index(current_theme) if current_theme in theme_options else 0
    theme_choice = st.sidebar.selectbox(
        "Color theme",
        theme_options,
        index=default_index,
    )
    if theme_choice != current_theme:
        apply_theme(theme_choice)
        st.experimental_rerun()

    section = st.sidebar.radio(
        "Go to",
        (
            "Data Summary",
            "Smart Insights",
            "Visualizations",
            "Missing Data",
            "Mini ML Model",
            "Export Report",
            "About",
        ),
    )

    return section


def render_header() -> None:
    """Render the main page title and intro copy."""

    st.title("Data Analysis and Visualization Dashboard")
    st.caption("Upload CSV datasets, explore structure, visualize patterns, and handle missing values.")


def render_section_placeholder(section: str) -> None:
    """Fallback helper when a section is not configured."""

    st.info(f"Section '{section}' is not available in this build.")


def render_about_section(_: pd.DataFrame) -> None:
    st.subheader("About This Dashboard")
    
    st.write(
        """
        **Data Analysis and Visualization Dashboard** is a comprehensive Python-based analytics tool 
        that provides end-to-end data exploration capabilities in an interactive, user-friendly interface.
        """
    )
    
    st.subheader("✨ Key Features")
    
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
    
    st.subheader("🛠️ Technologies Used")
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
    
    st.subheader("📋 How to Use")
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
    sns.set_theme(style=theme["style"], palette=theme["palette"])

    st.subheader("Visualization Explorer")
    st.write("Select a plot type and configure options to explore the dataset visually.")

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
    sns.set_theme(style=theme["style"], palette=theme["palette"])

    st.subheader("Missing Data Overview")
    summary = build_missing_summary(df)
    st.dataframe(summary, use_container_width=True)

    if summary["Missing Count"].sum() == 0:
        st.success("No missing values detected in the dataset.")
        return

    st.subheader("Missing Data Heatmap")
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

    st.subheader("Handle Missing Values")
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
                st.experimental_rerun()


def render_smart_insights(df: pd.DataFrame) -> None:
    """Display automatic insights and data quality metrics."""
    
    st.subheader("Smart Insights")
    st.write("Automatically generated insights about your dataset:")
    
    quality = calculate_data_quality_score(df)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Data Quality Score", f"{quality['score']}%", delta=f"{quality['score']-50:.1f}%")
    col2.metric("Completeness", f"{quality['completeness']}%")
    col3.metric("Uniqueness", f"{quality['uniqueness']}%")
    
    st.write("---")
    
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
    
    st.subheader("Outlier Analysis")
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
    
    st.subheader("Mini ML Model")
    st.write("Train simple predictive models on your dataset.")
    
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
    
    st.subheader("Export Reports & Data")
    
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

    with st.container():
        if renderer:
            renderer(st.session_state.dataset)
        else:
            render_section_placeholder(section)


if __name__ == "__main__":
    main()

