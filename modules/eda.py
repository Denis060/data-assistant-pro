# In modules/eda.py

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def create_eda_report(df: pd.DataFrame):
    """Generates and displays a comprehensive EDA report using Streamlit and Plotly."""
    try:
        st.subheader("📊 Dataset Overview")
        
        # Basic info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Duplicates", df.duplicated().sum())
        
        # Data types
        st.subheader("📋 Data Types")
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Missing Count': df.isnull().sum(),
            'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(dtype_df, use_container_width=True)
        
        # Missing values visualization
        if df.isnull().sum().sum() > 0:
            st.subheader("🔍 Missing Values Analysis")
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if len(missing_data) > 0:
                fig = px.bar(x=missing_data.index, y=missing_data.values,
                            title="Missing Values by Column",
                            labels={'x': 'Columns', 'y': 'Missing Count'})
                st.plotly_chart(fig, use_container_width=True)
        
        # Numerical columns analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.subheader("📈 Numerical Data Analysis")
            
            # Statistical summary
            st.write("**Statistical Summary:**")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            
            # Correlation matrix
            if len(numeric_cols) > 1:
                st.write("**Correlation Matrix:**")
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(corr_matrix, 
                               text_auto=True,
                               aspect="auto",
                               title="Correlation Matrix",
                               color_continuous_scale='RdBu_r')
                st.plotly_chart(fig, use_container_width=True)
            
            # Distribution plots for numerical columns
            st.write("**Distribution Plots:**")
            for col in numeric_cols[:6]:  # Limit to first 6 columns
                fig = px.histogram(df, x=col, title=f'Distribution of {col}',
                                 marginal="box")
                st.plotly_chart(fig, use_container_width=True)
        
        # Categorical columns analysis
        categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
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
                        fig = px.pie(values=value_counts.values, 
                                   names=value_counts.index,
                                   title=f'Distribution of {col}')
                        st.plotly_chart(fig, use_container_width=True)
        
        # Sample dataF
        st.subheader("📋 Sample Data")
        st.write("**First 10 rows:**")
        st.dataframe(df.head(10), use_container_width=True)
        
        return True
        
    except Exception as e:
        st.error(f"Error creating EDA report: {e}")
        return False