# In modules/eda.py

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from .cache_utils import (
    DataCache, 
    cached_correlation_matrix, 
    cached_missing_analysis, 
    cached_statistical_summary,
    with_progress_cache
)


def create_eda_report(df: pd.DataFrame):
    """Generates and displays a comprehensive EDA report using Streamlit and Plotly."""
    try:
        # Get data hash for caching
        df_hash = DataCache.get_data_hash(df)
        
        st.subheader("📊 Dataset Overview")

        # Basic info (quick calculations, no caching needed)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            # Use cached missing analysis
            missing_info = with_progress_cache(
                "Analyzing missing values",
                cached_missing_analysis,
                df_hash, df
            )
            st.metric("Missing Values", missing_info['total_missing'])
        with col4:
            st.metric("Duplicates", df.duplicated().sum())

        # Data types with enhanced information
        st.subheader("📋 Data Types")
        
        try:
            # Use cached statistical summary
            stats_info = with_progress_cache(
                "Computing statistical summary",
                cached_statistical_summary,
                df_hash, df
            )
            
            # Create dtype dataframe with proper alignment
            columns_list = df.columns.tolist()
            dtype_data = []
            
            for col in columns_list:
                row_data = {
                    "Column": col,
                    "Data Type": str(df.dtypes[col]),
                    "Non-Null Count": df[col].count(),
                    "Missing Count": missing_info['missing_count'].get(col, 0),
                    "Missing %": round(missing_info['missing_percentage'].get(col, 0.0), 2),
                    "Unique Values": stats_info['nunique'].get(col, df[col].nunique()),
                    "Memory (KB)": round((stats_info['memory_usage'].get(col, df[col].memory_usage(deep=True))) / 1024, 2)
                }
                dtype_data.append(row_data)
            
            dtype_df = pd.DataFrame(dtype_data)
            st.dataframe(dtype_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating data types summary: {e}")
            # Fallback to basic data types
            basic_dtype_df = pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str),
                "Non-Null Count": df.count(),
                "Missing Count": df.isnull().sum(),
                "Missing %": (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(basic_dtype_df, use_container_width=True)

        # Missing values visualization
        if df.isnull().sum().sum() > 0:
            st.subheader("🔍 Missing Values Analysis")
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

            if len(missing_data) > 0:
                fig = px.bar(
                    x=missing_data.index,
                    y=missing_data.values,
                    title="Missing Values by Column",
                    labels={"x": "Columns", "y": "Missing Count"},
                )
                st.plotly_chart(fig, use_container_width=True)

        # Numerical columns analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.subheader("📈 Numerical Data Analysis")

            # Statistical summary (use cached version)
            st.write("**Statistical Summary:**")
            st.dataframe(stats_info['describe'], use_container_width=True)

            # Correlation matrix
            if len(numeric_cols) > 1:
                st.write("**Correlation Matrix:**")
                corr_matrix = with_progress_cache(
                    "Computing correlation matrix",
                    cached_correlation_matrix,
                    df_hash, df
                )
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Correlation Matrix",
                    color_continuous_scale="RdBu_r",
                )
                st.plotly_chart(fig, use_container_width=True)

            # Distribution plots for numerical columns
            st.write("**Distribution Plots:**")
            for col in numeric_cols[:6]:  # Limit to first 6 columns
                fig = px.histogram(
                    df, x=col, title=f"Distribution of {col}", marginal="box"
                )
                st.plotly_chart(fig, use_container_width=True)

        # Categorical columns analysis
        categorical_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()
        if categorical_cols:
            st.subheader("📊 Categorical Data Analysis")

            # Value counts for categorical columns
            for col in categorical_cols[:5]:  # Limit to first 5 columns
                if df[col].nunique() <= 20:  # Only show if not too many unique values
                    st.write(f"**{col} - Value Counts:**")
                    value_counts = df[col].value_counts()

                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(value_counts, use_container_width=True)

                    with col2:
                        fig = px.pie(
                            values=value_counts.values,
                            names=value_counts.index,
                            title=f"Distribution of {col}",
                        )
                        st.plotly_chart(fig, use_container_width=True)

        # Sample dataF
        st.subheader("📋 Sample Data")
        st.write("**First 10 rows:**")
        st.dataframe(df.head(10), use_container_width=True)

        return True

    except Exception as e:
        st.error(f"Error creating EDA report: {e}")
        return False
