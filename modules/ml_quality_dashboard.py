"""
Data Quality Dashboard
Comprehensive assessment and visualization of data quality metrics
Specifically designed for ML readiness evaluation
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

def create_ml_quality_dashboard(df: pd.DataFrame, target_column: str = None):
    """
    Create comprehensive ML-focused data quality dashboard.
    
    Args:
        df: DataFrame to analyze
        target_column: Target variable for ML analysis
    """
    
    st.markdown("## 🎯 ML Data Quality Assessment")
    
    # Overall quality metrics
    quality_metrics = calculate_ml_quality_metrics(df, target_column)
    
    # Display overall score
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score = quality_metrics['overall_score']
        if score >= 90:
            st.success(f"🏆 **Quality Score**\n\n{score:.0f}%")
        elif score >= 75:
            st.info(f"✅ **Quality Score**\n\n{score:.0f}%")
        elif score >= 60:
            st.warning(f"⚠️ **Quality Score**\n\n{score:.0f}%")
        else:
            st.error(f"❌ **Quality Score**\n\n{score:.0f}%")
    
    with col2:
        completeness = quality_metrics['completeness']
        st.metric("📊 Completeness", f"{completeness:.1f}%")
    
    with col3:
        consistency = quality_metrics['consistency']
        st.metric("🔄 Consistency", f"{consistency:.1f}%")
    
    with col4:
        ml_readiness = quality_metrics['ml_readiness']
        st.metric("🎯 ML Readiness", f"{ml_readiness:.1f}%")
    
    # Detailed quality analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Missing Data", "📈 Distribution", "🎯 ML Features"])
    
    with tab1:
        show_quality_overview(df, quality_metrics)
    
    with tab2:
        show_missing_data_analysis(df)
    
    with tab3:
        show_distribution_analysis(df, target_column)
    
    with tab4:
        show_ml_feature_analysis(df, target_column)

def calculate_ml_quality_metrics(df: pd.DataFrame, target_column: str = None) -> dict:
    """Calculate comprehensive data quality metrics for ML."""
    
    metrics = {}
    
    # Basic statistics
    metrics['total_rows'] = len(df)
    metrics['total_columns'] = len(df.columns)
    metrics['memory_usage_mb'] = df.memory_usage(deep=True).sum() / 1024**2
    
    # Missing data analysis
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    metrics['missing_percentage'] = (missing_cells / total_cells) * 100
    metrics['completeness'] = 100 - metrics['missing_percentage']
    
    # Duplicate analysis
    duplicate_rows = df.duplicated().sum()
    metrics['duplicate_percentage'] = (duplicate_rows / len(df)) * 100
    
    # Data type distribution
    dtype_counts = df.dtypes.value_counts()
    metrics['numeric_columns'] = len(df.select_dtypes(include=[np.number]).columns)
    metrics['categorical_columns'] = len(df.select_dtypes(include=['object']).columns)
    metrics['datetime_columns'] = len(df.select_dtypes(include=['datetime64']).columns)
    
    # Consistency score
    consistency_score = 100
    
    # Penalize high missing data
    consistency_score -= min(metrics['missing_percentage'] * 2, 40)
    
    # Penalize duplicates
    consistency_score -= min(metrics['duplicate_percentage'], 20)
    
    # Penalize inconsistent data types
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        # Check if numeric data is stored as text
        try:
            numeric_conversion = pd.to_numeric(df[col], errors='coerce')
            numeric_pct = numeric_conversion.notna().sum() / df[col].notna().sum()
            if numeric_pct > 0.8:  # Mostly numeric but stored as object
                consistency_score -= 5
        except:
            pass
    
    metrics['consistency'] = max(0, consistency_score)
    
    # ML Readiness score
    ml_score = 100
    
    # Data size considerations
    if len(df) < 100:
        ml_score -= 30  # Too small for ML
    elif len(df) < 1000:
        ml_score -= 15  # Small but workable
    
    # Feature considerations
    if metrics['numeric_columns'] == 0:
        ml_score -= 20  # No numeric features
    
    if metrics['categorical_columns'] > metrics['numeric_columns'] * 2:
        ml_score -= 10  # Too many categorical features
    
    # Missing data impact on ML
    ml_score -= min(metrics['missing_percentage'] * 1.5, 25)
    
    # Constant features (bad for ML)
    constant_features = 0
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_features += 1
    
    if constant_features > 0:
        ml_score -= constant_features * 5
    
    metrics['ml_readiness'] = max(0, ml_score)
    
    # Overall score (weighted average)
    metrics['overall_score'] = (
        metrics['completeness'] * 0.3 +
        metrics['consistency'] * 0.3 +
        metrics['ml_readiness'] * 0.4
    )
    
    return metrics

def show_quality_overview(df: pd.DataFrame, metrics: dict):
    """Show overall data quality overview."""
    
    st.markdown("### 📊 Data Quality Overview")
    
    # Key statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📋 Basic Information**")
        st.write(f"• **Rows**: {metrics['total_rows']:,}")
        st.write(f"• **Columns**: {metrics['total_columns']:,}")
        st.write(f"• **Memory Usage**: {metrics['memory_usage_mb']:.1f} MB")
        st.write(f"• **Missing Data**: {metrics['missing_percentage']:.1f}%")
        st.write(f"• **Duplicates**: {metrics['duplicate_percentage']:.1f}%")
    
    with col2:
        st.markdown("**🔢 Feature Types**")
        st.write(f"• **Numeric**: {metrics['numeric_columns']}")
        st.write(f"• **Categorical**: {metrics['categorical_columns']}")
        st.write(f"• **DateTime**: {metrics['datetime_columns']}")
        
        # Data type distribution chart
        if metrics['total_columns'] > 0:
            type_data = {
                'Type': ['Numeric', 'Categorical', 'DateTime'],
                'Count': [metrics['numeric_columns'], metrics['categorical_columns'], metrics['datetime_columns']]
            }
            type_df = pd.DataFrame(type_data)
            type_df = type_df[type_df['Count'] > 0]
            
            if not type_df.empty:
                fig = px.pie(type_df, values='Count', names='Type', title="Feature Type Distribution")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

def show_missing_data_analysis(df: pd.DataFrame):
    """Show detailed missing data analysis."""
    
    st.markdown("### 🔍 Missing Data Analysis")
    
    missing_summary = df.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)
    
    if len(missing_summary) == 0:
        st.success("🎉 No missing data found!")
        return
    
    # Missing data by column
    missing_df = pd.DataFrame({
        'Column': missing_summary.index,
        'Missing Count': missing_summary.values,
        'Missing %': (missing_summary.values / len(df) * 100).round(2)
    })
    
    # Color code by severity
    def color_missing_percentage(val):
        if val >= 50:
            return 'background-color: #ffcccc'  # Red for high missing
        elif val >= 20:
            return 'background-color: #ffffcc'  # Yellow for medium missing
        else:
            return 'background-color: #ccffcc'  # Green for low missing
    
    styled_df = missing_df.style.applymap(color_missing_percentage, subset=['Missing %'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Missing data visualization
    if len(missing_summary) > 0:
        fig = px.bar(
            missing_df, 
            x='Column', 
            y='Missing %',
            title='Missing Data Percentage by Column',
            color='Missing %',
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Missing data recommendations
    st.markdown("**💡 Recommendations:**")
    high_missing = missing_df[missing_df['Missing %'] >= 50]
    medium_missing = missing_df[(missing_df['Missing %'] >= 20) & (missing_df['Missing %'] < 50)]
    low_missing = missing_df[missing_df['Missing %'] < 20]
    
    if len(high_missing) > 0:
        st.warning(f"⚠️ **High missing data** ({len(high_missing)} columns): Consider dropping these columns or using advanced imputation")
    
    if len(medium_missing) > 0:
        st.info(f"ℹ️ **Medium missing data** ({len(medium_missing)} columns): Use appropriate imputation methods")
    
    if len(low_missing) > 0:
        st.success(f"✅ **Low missing data** ({len(low_missing)} columns): Simple imputation should work well")

def show_distribution_analysis(df: pd.DataFrame, target_column: str = None):
    """Show distribution analysis for features."""
    
    st.markdown("### 📈 Feature Distribution Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    if len(numeric_cols) == 0:
        st.warning("No numeric columns found for distribution analysis.")
        return
    
    # Select column for analysis
    selected_col = st.selectbox("Select column for detailed analysis:", numeric_cols)
    
    if selected_col:
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot
            fig = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        with stats_col1:
            st.metric("Mean", f"{df[selected_col].mean():.2f}")
        with stats_col2:
            st.metric("Median", f"{df[selected_col].median():.2f}")
        with stats_col3:
            st.metric("Std Dev", f"{df[selected_col].std():.2f}")
        with stats_col4:
            st.metric("Skewness", f"{df[selected_col].skew():.2f}")

def show_ml_feature_analysis(df: pd.DataFrame, target_column: str = None):
    """Show ML-specific feature analysis."""
    
    st.markdown("### 🎯 ML Feature Analysis")
    
    # Feature importance for categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)
    
    # High cardinality features
    high_cardinality = []
    for col in categorical_cols:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.5:  # More than 50% unique values
            high_cardinality.append({
                'Column': col,
                'Unique Values': df[col].nunique(),
                'Unique Ratio': f"{unique_ratio:.1%}"
            })
    
    if high_cardinality:
        st.warning("⚠️ **High Cardinality Features** (may need special handling):")
        high_card_df = pd.DataFrame(high_cardinality)
        st.dataframe(high_card_df, use_container_width=True)
    
    # Constant features
    constant_features = []
    for col in df.columns:
        if col != target_column and df[col].nunique() <= 1:
            constant_features.append(col)
    
    if constant_features:
        st.error(f"❌ **Constant Features** (should be removed): {', '.join(constant_features)}")
    
    # Nearly constant features
    nearly_constant = []
    for col in df.columns:
        if col != target_column and df[col].nunique() > 1:
            mode_frequency = df[col].value_counts().iloc[0] / len(df)
            if mode_frequency > 0.95:  # More than 95% same value
                nearly_constant.append({
                    'Column': col,
                    'Dominant Value Frequency': f"{mode_frequency:.1%}"
                })
    
    if nearly_constant:
        st.warning("⚠️ **Nearly Constant Features** (low information value):")
        nearly_const_df = pd.DataFrame(nearly_constant)
        st.dataframe(nearly_const_df, use_container_width=True)
    
    # Memory optimization suggestions
    st.markdown("**💾 Memory Optimization Opportunities:**")
    
    memory_savings = []
    for col in df.select_dtypes(include=[np.number]).columns:
        current_memory = df[col].memory_usage(deep=True)
        
        if df[col].dtype == 'float64':
            # Check if can be downcast to float32
            if df[col].min() >= np.finfo(np.float32).min and df[col].max() <= np.finfo(np.float32).max:
                potential_memory = df[col].astype(np.float32).memory_usage(deep=True)
                savings = current_memory - potential_memory
                memory_savings.append({
                    'Column': col,
                    'Current Type': 'float64',
                    'Suggested Type': 'float32',
                    'Memory Savings': f"{savings / 1024:.1f} KB"
                })
        
        elif df[col].dtype == 'int64':
            # Check if can be downcast to int32
            if df[col].min() >= np.iinfo(np.int32).min and df[col].max() <= np.iinfo(np.int32).max:
                potential_memory = df[col].astype(np.int32).memory_usage(deep=True)
                savings = current_memory - potential_memory
                memory_savings.append({
                    'Column': col,
                    'Current Type': 'int64',
                    'Suggested Type': 'int32',
                    'Memory Savings': f"{savings / 1024:.1f} KB"
                })
    
    if memory_savings:
        memory_df = pd.DataFrame(memory_savings)
        st.dataframe(memory_df, use_container_width=True)
        total_savings = sum([float(x['Memory Savings'].replace(' KB', '')) for x in memory_savings])
        st.info(f"💡 Total potential memory savings: {total_savings:.1f} KB")
    else:
        st.success("✅ Memory usage is already optimized!")
